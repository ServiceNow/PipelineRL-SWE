"""Effective billed $/token per model from the fresh collection's usage_cost (least squares); shared by the fresh-set analyses."""
import glob, json, collections
import numpy as np
from decompose import R


def billed_rates():
    by = collections.defaultdict(list); byp = collections.defaultdict(list)
    for f in glob.glob(f"{R}/math_expand_20261001/*/*.jsonl"):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("usage_cost") is None or r.get("finish_reason") == "error": continue
            m = "oss120" if "120" in r["route_label"] else ("oss20" if r["route_label"].startswith("oss20") else "dsv4f")
            x = (r["prompt_tokens"], r["completion_tokens"], r["usage_cost"]); by[m].append(x); byp[(m, r["provider"])].append(x)
    fit = lambda rows: np.linalg.lstsq(np.array([[a, b] for a, b, _ in rows], float), np.array([c for *_, c in rows]), rcond=None)[0]
    return {m: fit(v) for m, v in by.items()}, {k: fit(v) for k, v in byp.items() if len(v) >= 30}


RATE, PRATE = billed_rates()
