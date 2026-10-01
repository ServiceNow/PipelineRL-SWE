#!/usr/bin/env python3
"""Collect a frozen disjoint evaluation sample; resume safely and record billed usage."""
import argparse,asyncio,fcntl,json,time
from pathlib import Path
import aiohttp
from collect_math_pool import ROUTES,PROMPT,call,_grade_task


async def run(a):
    plan=json.loads(Path(a.plan).read_text())
    tasks=[json.loads(l) for l in Path(a.tasks).read_text().splitlines()]
    expected=plan['estimates'][a.dataset]['new_problems']
    if len(tasks)!=expected or len({t['problem_id'] for t in tasks})!=expected:
        raise ValueError('Sample size or ID uniqueness does not match frozen plan')
    import hashlib
    if hashlib.sha256(Path(a.tasks).read_bytes()).hexdigest()!=plan['estimates'][a.dataset]['sample_sha256']:
        raise ValueError('Sample hash does not match frozen plan')
    out=Path(a.out)/a.dataset;out.mkdir(parents=True,exist_ok=True)
    lockfile=(out/'collector.lock').open('w')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    complete=out/'COMPLETE.json'
    if complete.exists():print('Already complete',flush=True);return
    key=Path(a.api_key_file).read_text().strip()
    sem=asyncio.Semaphore(a.concurrency)
    spent=0.;donecount=0;errors=0;reserved=0.;stopped=False
    condition=asyncio.Condition()
    handles={};queue=asyncio.Queue()
    expected_calls=len(tasks)*sum(plan['draws'].values())
    byid={t['problem_id']:t for t in tasks}
    def capped_cost(row,route):
        if row.get('usage_cost') is not None:return float(row['usage_cost'])
        pin,pout=plan['price_caps_per_million'][route]
        return (row.get('prompt_tokens',0)*pin+row.get('completion_tokens',0)*pout)/1e6
    existing={}
    for route,draws in plan['draws'].items():
        for draw in range(draws):
            path=out/f'{route}_d{draw}.jsonl';done=set()
            if path.exists():
                # Recover an incomplete final line after preemption. All earlier
                # records must parse; silently dropping earlier corruption is unsafe.
                raw=path.read_bytes();lines=raw.splitlines(keepends=True);offset=0
                for i,line in enumerate(lines):
                    try:row=json.loads(line)
                    except json.JSONDecodeError:
                        if i==len(lines)-1 and not line.endswith(b'\n'):
                            with path.open('r+b') as f:f.truncate(offset)
                            break
                        raise
                    offset+=len(line)
                    spent+=capped_cost(row,route)
                    if row.get('finish_reason')!='error':done.add(row['problem_id'])
            existing[route,draw]=done
            donecount+=len(done)
            handles[route,draw]=path.open('a')
            for task in tasks:
                if task['problem_id'] not in done:queue.put_nowait((route,draw,task))
    metadata=[{k:t[k] for k in ['problem_id','answer','difficulty','subject']}|{'problem_statement':t['problem']} for t in tasks]
    (out/'problems.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    (out/'collection_plan.json').write_text(json.dumps(plan,indent=2))
    started=time.time()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def balance():
            async with session.get('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+key},timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status();d=(await response.json())['data']
            return d.get('limit_remaining')
        remaining=await balance()
        if remaining is not None and remaining<min(a.budget_usd-spent,5.):
            raise RuntimeError('Insufficient remaining key allowance')
        print(json.dumps({'event':'start','dataset':a.dataset,'todo_calls':queue.qsize(),'existing_calls':donecount,'existing_spend_usd':spent,'budget_usd':a.budget_usd,'key_allowance_remaining_usd':remaining}),flush=True)
        async def worker():
            nonlocal spent,donecount,errors,reserved,stopped
            while not queue.empty() and not stopped:
                route,draw,task=queue.get_nowait()
                prompt=task.get('prompt') or PROMPT.format(problem=task['problem'])
                pin,pout=plan['price_caps_per_million'][route]
                # Reserve enough for four attempts at the configured maximum.
                # Unknown charges from ambiguous network failures are not observable.
                reserve=4*((len(prompt.encode())+512)*pin+a.max_tokens*pout)/1e6
                async with condition:
                    while spent+reserved+reserve>a.budget_usd and reserved>0 and not stopped:
                        await condition.wait()
                    if stopped or spent+reserved+reserve>a.budget_usd:
                        stopped=True;condition.notify_all();queue.task_done();return
                    reserved+=reserve
                row=await call(session,key,route,prompt,sem,a.max_tokens,provider_max_price={'prompt':pin,'completion':pout})
                row.update(problem_id=task['problem_id'],dataset=a.dataset,route_label=route,model=ROUTES[route][0],draw=draw,difficulty=task['difficulty'],resolved=_grade_task(task,row))
                handles[route,draw].write(json.dumps(row)+'\n');handles[route,draw].flush()
                async with condition:
                    spent+=capped_cost(row,route);reserved-=reserve
                    if row['finish_reason']=='error':errors+=1
                    else:donecount+=1
                    condition.notify_all()
                queue.task_done()
                if (donecount+errors)%100==0:
                    progress={'valid_calls':donecount,'expected_calls':expected_calls,'error_rows_this_run':errors,'observed_spend_usd':spent,'elapsed_seconds':time.time()-started}
                    (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
                if errors>=max(20,int(.05*(donecount+errors))):
                    stopped=True
                    async with condition:condition.notify_all()
        try:await asyncio.gather(*[worker() for _ in range(a.concurrency)])
        finally:
            for f in handles.values():f.close()
    result={'valid_calls':donecount,'expected_calls':expected_calls,'error_rows_this_run':errors,'observed_spend_usd':spent,'budget_usd':a.budget_usd,'budget_or_error_stop':stopped,'elapsed_seconds':time.time()-started}
    (out/'progress.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if donecount!=expected_calls:raise RuntimeError('Incomplete collection; inspect progress and resume explicitly')
    complete.write_text(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--tasks',required=True)
    ap.add_argument('--dataset',choices=['mmlupro','omni500'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--budget-usd',type=float,required=True)
    ap.add_argument('--concurrency',type=int,default=48);ap.add_argument('--max-tokens',type=int,default=64000)
    ap.add_argument('--api-key-file',default='/home/toolkit/.secrets/openrouter_api_key')
    args=ap.parse_args()
    if args.budget_usd<=0 or args.concurrency<=0:ap.error('budget and concurrency must be positive')
    asyncio.run(run(args))
if __name__=='__main__':main()
