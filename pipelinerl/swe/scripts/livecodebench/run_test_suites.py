"""Execute generated test-suites against stored pool candidates (test-writer smoke test).

For each (problem, test-writer): run every stored candidate with each suite case's input on stdin,
compare stdout with the suite's expected_output, and record per-case verdicts against the REAL
grader label (draw_records.jsonl:final_outcome). Output: verdicts_{writer}.jsonl with one row per
(problem, writer, candidate).

Executed locally: subprocess in a fresh temp dir per candidate, timeout per case, no network.
Resumable: skips (writer, slot, draw_index) rows already in the output file.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def norm(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def run_case(candidate_path: str, workdir: str, inp: str, expected: str, timeout: float):
    try:
        proc = subprocess.run(
            [sys.executable, candidate_path],
            input=inp.encode(),
            capture_output=True,
            timeout=timeout,
            cwd=workdir,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir,
                 "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": f"exec_error:{type(e).__name__}"}
    if proc.returncode != 0:
        return {"status": "crash", "stderr_tail": proc.stderr.decode(errors="replace")[-200:]}
    return {"status": "ran", "pass": norm(proc.stdout.decode(errors="replace")) == norm(expected)}


def run_candidate(suite: dict, candidate: dict, tmp_root: str, timeout: float, max_cases: int):
    cases = suite["cases"][:max_cases]
    workdir = tempfile.mkdtemp(prefix="twsmoke_", dir=tmp_root)
    sol_path = os.path.join(workdir, "sol.py")
    try:
        with open(sol_path, "w") as f:
            f.write(candidate["code"])
        per_case, err = [], None
        for case in cases:
            r = run_case(sol_path, workdir, case["input"], case["expected_output"], timeout)
            if r["status"] == "ran":
                per_case.append(bool(r["pass"]))
            elif r["status"].startswith("exec_error"):
                per_case.append(None)
                err = err or r["status"]
            else:
                # crash/timeout: the candidate failed this case; the suite still ran it
                per_case.append(False)
        n_pass = sum(1 for c in per_case if c is True)
        n_ran = sum(1 for c in per_case if c is not None)
        return {
            "problem_id": suite["problem_id"],
            "writer": suite["writer"],
            "slot": candidate["slot"],
            "draw_index": candidate["draw_index"],
            "truth": candidate["truth"],
            "suite_ok": n_ran == len(cases) and n_ran > 0,
            "suite_err": err,
            "n_cases": len(cases),
            "n_ran": n_ran,
            "n_case_pass": n_pass,
            "per_case": per_case,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def load_candidates(pool_dir: Path, cap_per_problem: int, seed: int):
    """Round-robin across model slots per problem so no single rung dominates the sample."""
    by_slot = {}
    for line in open(pool_dir / "draw_records.jsonl"):
        d = json.loads(line)
        by_slot.setdefault(d["problem_id"], {}).setdefault(d["model_slot"], []).append(
            {"slot": d["model_slot"], "draw_index": d["draw_index"],
             "code": d["code"], "truth": bool(d["final_outcome"])}
        )
    out = {}
    for pid, slots in sorted(by_slot.items()):
        rng = random.Random(f"{seed}:{pid}")
        for s in slots:
            rng.shuffle(slots[s])
        picked, order = [], sorted(slots)
        while len(picked) < cap_per_problem and any(slots[s] for s in order):
            for s in order:
                if slots[s] and len(picked) < cap_per_problem:
                    picked.append(slots[s].pop())
        out[pid] = picked
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--suites-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--writers", default="qwen4b,oss20lo,oss20md,dsv4f,oss120md")
    ap.add_argument("--cap-per-problem", type=int, default=12)
    ap.add_argument("--max-cases", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = str(out_dir / "tmp_exec")
    Path(tmp_root).mkdir(exist_ok=True)

    cand_by_problem = load_candidates(Path(args.pool_dir), args.cap_per_problem, args.seed)

    for writer in args.writers.split(","):
        writer = writer.strip()
        suites_path = Path(args.suites_dir) / f"suites_{writer}.jsonl"
        out_path = out_dir / f"verdicts_{writer}.jsonl"
        done = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        done.add((d["problem_id"], d["slot"], d["draw_index"]))
                    except Exception:
                        continue
        tasks, missing = [], 0
        for line in open(suites_path):
            s = json.loads(line)
            if not s.get("cases"):
                continue
            cands = cand_by_problem.get(s["problem_id"], [])
            if not cands:
                continue
            for c in cands:
                if (s["problem_id"], c["slot"], c["draw_index"]) in done:
                    continue
                tasks.append((s, c))
        print(f"[{writer}] {len(tasks)} candidate executions to run")
        with open(out_path, "a") as fout, ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_candidate, s, c, tmp_root, args.timeout, args.max_cases)
                    for s, c in tasks]
            n = 0
            for fut in as_completed(futs):
                row = fut.result()
                fout.write(json.dumps(row) + "\n")
                n += 1
                if n % 500 == 0:
                    fout.flush()
                    print(f"[{writer}] {n}/{len(tasks)}")
            fout.flush()
            print(f"[{writer}] done: {n} rows -> {out_path}")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()