#!/usr/bin/env python3
"""Track B, S1 with abstention: generate once, test once, submit only if the test passes.

Utility per instance: +1 correct submission, -lambda wrong submission, 0 abstain. A test that is not
VALID (label-free: passes on the unpatched repo) gives no information; the pair then submits anyway.
Compares, per lambda: always-submit for each generator (no test) vs every (generator, tester) pair,
open models only. Reports utility, coverage and cost; paired bootstrap CI for the best tested pair vs
the best no-test policy (both chosen on the full set -> optimistic for both; a held-out choice follows
when the 369-instance data lands).
"""
import argparse, json
import numpy as np

SELF = {"oss20": "oss20", "oss120": "oss120", "qwen30": "qcoder30"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass-matrix", required=True)
    ap.add_argument("--gens", default="oss20,qwen30,oss120")
    ap.add_argument("--writers", default="oss20,qcoder30,dsv4f,oss120,devstral")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = [json.loads(l) for l in open(a.pass_matrix)]
    G, W = a.gens.split(","), a.writers.split(",")
    res = {}   # policy -> (correct_submitted, wrong_submitted, cost) arrays
    for g in G:
        cs, ws, co = [], [], []
        for r in recs:
            c = next((x for x in r["candidates"] if x["gen"] == g), None)
            if c is None:                      # patch did not apply: counts as a wrong submission
                cs.append(0); ws.append(1); co.append(0.0); continue
            cs.append(int(c["correct"])); ws.append(int(not c["correct"])); co.append(c["gen_cost_c"])
        res[f"{g} (no test)"] = tuple(map(np.array, (cs, ws, co)))
        for w in W:
          for strict in (False, True):   # strict: an INVALID test also means abstain
            cs, ws, co = [], [], []
            for r in recs:
                c = next((x for x in r["candidates"] if x["gen"] == g), None)
                t = next((x for x in r["tests"] if x["writer"] == w), None)
                if c is None:
                    cs.append(0); ws.append(0); co.append(0.0); continue   # nothing to submit: abstain
                wc = t["write_cost_c"] if t and np.isfinite(t["write_cost_c"]) else 0.0
                if t is None or t["valid"] is False:
                    submit = not strict
                else:
                    submit = t["passes"].get(c["cid"], False)
                cs.append(int(submit and c["correct"])); ws.append(int(submit and not c["correct"]))
                co.append(c["gen_cost_c"] + wc + r["run_cost_c"])
            tag = (" [self]" if SELF.get(g) == w else "") + (" strict" if strict else "")
            res[f"{g} + {w} test{tag}"] = tuple(map(np.array, (cs, ws, co)))
    res["always abstain"] = (np.zeros(len(recs)), np.zeros(len(recs)), np.zeros(len(recs)))
    print(f"{len(recs)} instances; open generators {G}; testers {W}")
    print(f"{'lambda':>7} {'best no-test policy':<24}{'U':>7}   {'best tested pair':<34}{'U':>7}{'cov':>6}{'cost':>8}   gain [95% CI]")
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
        U = {k: v[0] - lam * v[1] for k, v in res.items()}
        nt = max((k for k in U if "no test" in k or k == "always abstain"), key=lambda k: U[k].mean())
        tt = max((k for k in U if "test" in k and "no test" not in k), key=lambda k: U[k].mean())
        d = U[tt] - U[nt]; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        cov = (res[tt][0] + res[tt][1]).mean()
        print(f"{lam:>7} {nt:<24}{U[nt].mean():>7.3f}   {tt:<34}{U[tt].mean():>7.3f}{cov*100:>5.0f}%{res[tt][2].mean():>8.3f}   "
              f"{d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    print("\n(strict rule: submit only if the test is valid AND passes)")
    print("self-test vs best other tester, per generator (precision of submitted answers / coverage):")
    for g in G:
        rows = {k: v for k, v in res.items() if k.startswith(g + " +")}
        def pc(v):
            sub = v[0] + v[1]; return (v[0].sum() / max(sub.sum(), 1), sub.mean())
        s = next(k for k in rows if "[self] strict" in k); o = max((k for k in rows if "[self]" not in k and "strict" in k), key=lambda k: pc(rows[k])[0])
        print(f"   {g:<8} self: {pc(rows[s])[0]*100:4.0f}% prec / {pc(rows[s])[1]*100:3.0f}% cov   best other ({o.split('+ ')[1]}): "
              f"{pc(rows[o])[0]*100:4.0f}% / {pc(rows[o])[1]*100:3.0f}%   | no test: {res[g + ' (no test)'][0].mean()*100:.0f}% correct")


if __name__ == "__main__":
    main()
