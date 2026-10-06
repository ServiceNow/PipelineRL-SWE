"""Shadow root with deepseek-v4-flash PINNED (NEW_PATH 4.A.59). Everything under the real root is symlinked EXCEPT:
  - each pool's tensors dir: rebuilt with ONLY the dsv4f slot replaced by the pinned draws (same problems, order, splits, draw
    indices); only features/embeddings (.npy, .npz other than tensors/zr_) are symlinked, so nothing stale can be read;
    feature / embedding files are symlinked;
  - expanded_eval_20261001/<ds>: same treatment (original rows from math_pool_pinned, fresh rows from math_expand_pinned);
  - math_expand_20261001/<ds>/dsv4f_d0.jsonl: replaced by the pinned fresh rows (billed rates, realized costs).
Validity follows each pool's own builder: math = finish_reason != error; bcb/cc = also non-empty full_output; lcb = pool_v2 infra rule.
Run analysis with REASON_ROOT=<shadow> (decompose.R honours it). Usage: python build_pinned_root.py [pool ...]
"""
import json, os, re, shutil, sys
from pathlib import Path
import numpy as np

REAL = Path("/mnt/llmd/results/exps/aristides/reason"); PIN = "StreamLake"
ANCHOR = "--anchor" in sys.argv       # unchanged tensors in reason_anchor/: regenerated predictions must reproduce the archived ones
SH = REAL.parent / ("reason_anchor" if ANCHOR else "reason_pinned")
POOLS = {   # tensors dir: (pinned source spec, validity rule)
    "pool_v2_tensors_5rung": ("split", REAL / "pool_v2_lcb_pinned", "lcb"),
    "bcb_tensors_5r": ("split", REAL / "pool_v2_bcb_pinned", "bcb"),
    "cc_tensors": ("split", REAL / "cc_pool" / "full_pinned", "bcb"),
    "omni500_tensors": ("math", REAL / "math_pool_pinned" / "omni500", "math"),
    "mmlupro_tensors": ("math", REAL / "math_pool_pinned" / "mmlupro", "math"),
    "aime_tensors": ("math", REAL / "math_pool_pinned" / "aime", "math"),
    "apps_tensors": ("apps", REAL / "provider_pilot_20261002" / "apps" / f"dsv4f_pin_{PIN}.jsonl", "math"),
}
KEEP_FILES = ("tensors.npz", "split_manifest.json", "problems.jsonl", "prices.json", "mmlupro_meta.json")


def link(dst, src):
    """Symlink, tolerating a concurrent pool job that created it first."""
    try:
        dst.symlink_to(src)
    except FileExistsError:
        pass


def ok(r, rule):
    if r.get("finish_reason") == "error" or r.get("error"):
        return False
    if rule == "bcb":
        return bool(str(r.get("full_output") or "").strip())
    if rule == "lcb":
        if str(r.get("full_output") or "").strip() or r.get("finish_reason") == "length":
            return True
        msg = str((r.get("eval_metadata") or {}).get("error_message", ""))
        return bool(r.get("provider")) and "PayloadError" not in msg and "ClientConnection" not in msg
    return bool(r.get("completion_tokens") or r.get("content") or r.get("reasoning"))


def pinned_rows(kind, src):
    out = {}
    if kind == "split":
        files = [(f, int(re.search(r"_d(\d+)\.jsonl$", f.name).group(1))) for f in src.glob("dsv4f_*_d*.jsonl")]
    elif kind == "math":
        files = [(f, int(re.search(r"_d(\d+)\.jsonl$", f.name).group(1))) for f in src.glob("dsv4f_d*.jsonl")]
    else:
        files = [(src, 0)]
    for f, d in files:
        for l in open(f):
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                out[(str(r["problem_id"]), d)] = r
    return out


def swap(t, rows, rule, offset=0, n_rows=None, draws=None):
    """Replace the dsv4f slot for rows [offset, offset+n_rows) from pinned rows keyed (problem_id, draw)."""
    if ANCHOR:
        arr = {k: np.array(t[k]).copy() for k in t}; j = list(map(str, arr["model_slots"])).index("dsv4f")
        n = int(arr["valid"][offset:offset + (n_rows or len(arr["problem_ids"])), j].sum()); return arr, n, n
    arr = {k: np.array(t[k]).copy() for k in t}; slots = list(map(str, arr["model_slots"])); j = slots.index("dsv4f")
    ids = list(map(str, arr["problem_ids"])); K = arr["valid"].shape[2]; n_rows = n_rows or len(ids)
    old_valid = int(arr["valid"][offset:offset + n_rows, j].sum())
    for k_ in ("final_outcome", "execution_outcome", "weak_verifier_outcome", "valid"):
        if k_ in arr:
            arr[k_][offset:offset + n_rows, j] = False
    for k_ in ("prompt_tokens", "completion_tokens"):
        arr[k_][offset:offset + n_rows, j] = 0
    filled = 0
    for i in range(offset, offset + n_rows):
        for d in range(draws if draws is not None else K):
            r = rows.get((ids[i], d))
            if r is None or not ok(r, rule):
                continue
            res = bool(r.get("resolved"))
            for k_ in ("final_outcome", "execution_outcome", "weak_verifier_outcome"):
                if k_ in arr:
                    arr[k_][i, j, d] = res
            arr["valid"][i, j, d] = True; filled += 1
            arr["prompt_tokens"][i, j, d] = float(r.get("prompt_tokens", 0)); arr["completion_tokens"][i, j, d] = float(r.get("completion_tokens", 0))
    return arr, old_valid, filled


def make_dir(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        d = dst_dir / f.name
        if d.exists() or d.is_symlink():
            continue
        if f.name in KEEP_FILES:
            if f.name != "tensors.npz":
                shutil.copy(f, d)
        elif f.suffix == ".npy" or (f.suffix == ".npz" and f.name != "tensors.npz" and not f.name.startswith("zr_")):
            link(d, f)                    # features / embeddings only; every label-derived file is regenerated


def report(name, t_old, arr, j, old_valid, filled, rows_slice=slice(None)):
    v0, v1 = t_old["valid"][rows_slice, j], arr["valid"][rows_slice, j]
    a0 = t_old["final_outcome"][rows_slice, j][v0].mean(); a1 = arr["final_outcome"][rows_slice, j][v1].mean()
    l0 = t_old["completion_tokens"][rows_slice, j][v0].mean(); l1 = arr["completion_tokens"][rows_slice, j][v1].mean()
    info = dict(valid_before=old_valid, valid_after=filled, acc_before=float(a0), acc_after=float(a1), mean_out_before=float(l0), mean_out_after=float(l1))
    print(f"{name:<32} dsv4f valid {old_valid} -> {filled}; acc {a0:.3f} -> {a1:.3f}; mean out {l0:.0f} -> {l1:.0f}", flush=True)
    return info


def main():
    only = [a for a in sys.argv[1:] if a != "--anchor"] or list(POOLS) + ["expanded_mmlupro", "expanded_omni500"]
    SH.mkdir(exist_ok=True); man = {}
    for f in REAL.iterdir():                                  # symlink the untouched world
        d = SH / f.name
        if f.name in POOLS or f.name in ("expanded_eval_20261001", "math_expand_20261001") or d.exists() or d.is_symlink():
            continue
        link(d, f)
    for name in [p for p in only if p in POOLS]:
        kind, src, rule = POOLS[name]; t = np.load(REAL / name / "tensors.npz", allow_pickle=True)
        make_dir(REAL / name, SH / name); arr, ov, fl = swap({k: t[k] for k in t.files}, pinned_rows(kind, src), rule)
        np.savez(SH / name / "tensors.npz", **arr)
        man[name] = report(name, t, arr, list(map(str, t["model_slots"])).index("dsv4f"), ov, fl)
    for ds in [p.split("_", 1)[1] for p in only if p.startswith("expanded_")]:
        src_dir = REAL / "expanded_eval_20261001" / ds; dst = SH / "expanded_eval_20261001" / ds; make_dir(src_dir, dst)
        t = np.load(src_dir / "tensors.npz", allow_pickle=True)
        n_old = len(np.load(REAL / f"{ds}_tensors" / "tensors.npz", allow_pickle=True)["problem_ids"])
        arr, ov, fl = swap({k: t[k] for k in t.files}, pinned_rows("math", REAL / "math_pool_pinned" / ds), "math", 0, n_old, draws=3)
        arr2, ov2, fl2 = swap(arr, pinned_rows("math", REAL / "math_expand_pinned_20261005" / ds), "math", n_old, len(arr["problem_ids"]) - n_old, draws=1)
        np.savez(dst / "tensors.npz", **arr2)
        j = list(map(str, t["model_slots"])).index("dsv4f")
        man[f"expanded_{ds}_original_rows"] = report(f"expanded/{ds} original rows", t, arr2, j, ov, fl, slice(0, n_old))
        man[f"expanded_{ds}_fresh_rows"] = report(f"expanded/{ds} fresh rows", t, arr2, j, ov2, fl2, slice(n_old, None))
        me_src, me_dst = REAL / "math_expand_20261001" / ds, SH / "math_expand_20261001" / ds; me_dst.mkdir(parents=True, exist_ok=True)
        for f in me_src.iterdir():
            d = me_dst / f.name
            if f.name == "dsv4f_d0.jsonl" and not ANCHOR:
                if d.is_symlink() or d.exists():
                    d.unlink()
                shutil.copy(REAL / "math_expand_pinned_20261005" / ds / "dsv4f_d0.jsonl", d)
            elif not (d.exists() or d.is_symlink()):
                link(d, f)
    for d in ("expanded_eval_20261001", "math_expand_20261001"):            # remaining children of the two partial dirs
        for f in (REAL / d).iterdir():
            dd = SH / d / f.name
            if not (dd.exists() or dd.is_symlink()) and f.name not in ("mmlupro", "omni500"):
                (SH / d).mkdir(exist_ok=True); link(dd, f)
    for k, v in man.items():                                                  # one manifest per pool: no write races
        (SH / f"pinned_manifest_{k}.json").write_text(json.dumps(v, indent=1))
    print(f"shadow root: {SH}")


if __name__ == "__main__":
    main()
