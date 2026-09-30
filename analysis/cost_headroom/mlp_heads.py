"""Controlled joint-MLP vs linear heads on frozen rich prefill features.

Run on a GPU via launch_mlp_heads.sh. No API calls or encoder inference.
Configuration and epoch selection use calibration only. All four arms share
features, splits, prices, labels, and routing evaluation. Published predictions
are retained separately to audit reconstruction of the linear baseline.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from baseline_cost_heads import rich
from decompose import MK, R, cost_at, hull

POOLS = {
    "LCB": ("pool_v2_tensors_5rung", "pv2_scout_prefill_1756715297/scout.npz", "cost_preds_probe.jsonl", [0.70, 0.75, 0.80]),
    "Omni": ("omni500_tensors", "omni500_probe/thinking.npz", "cost_preds_probe_thinking.jsonl", [0.60, 0.65, 0.70]),
}
CONFIGS = [dict(width=64, dropout=0.2, weight_decay=0.1),
           dict(width=128, dropout=0.2, weight_decay=0.1)]
SEEDS = [0, 1, 2]
VS = np.geomspace(1e-5, 100, 300)


def read_preds(path, ids, key):
    rows = {str(d["problem_id"]): d[key] for d in map(json.loads, path.open())}
    return np.array([rows[p] for p in ids], dtype=float)


def fit_mlp(X, y, weight, tr, ca, task, config, seed, epochs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], config["width"]),
                              torch.nn.GELU(), torch.nn.Dropout(config["dropout"]),
                              torch.nn.Linear(config["width"], y.shape[1])).cuda()
    if task == "success":
        prior = (y[tr] * weight[tr]).sum(0) / weight[tr].sum(0)
        with torch.no_grad():
            net[-1].bias.copy_(torch.tensor(np.log(np.clip(prior, 1e-4, 1-1e-4) / np.clip(1-prior, 1e-4, 1)), device="cuda", dtype=torch.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=config["weight_decay"])
    yt = torch.tensor(y, device="cuda", dtype=torch.float32)
    wt = torch.tensor(weight, device="cuda", dtype=torch.float32)

    def loss(z, ii):
        element = torch.nn.functional.binary_cross_entropy_with_logits(z, yt[ii], reduction="none") if task == "success" else (z - yt[ii]).square()
        return ((element * wt[ii]).sum(0) / wt[ii].sum(0).clamp_min(1)).mean()

    best_loss, best_epoch, state = float("inf"), -1, None
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        train_loss = loss(net(X[tr]), tr)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss(net(X[ca]), ca))
        if val < best_loss - 1e-6:
            best_loss, best_epoch = val, ep
            state = copy.deepcopy(net.state_dict())
        if ep - best_epoch >= 50:
            break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        pred = net(X).cpu().numpy()
    print(f"{task} {config} seed={seed} epoch={best_epoch} calibration_loss={best_loss:.6f}", flush=True)
    return pred, dict(seed=seed, best_epoch=best_epoch, calibration_loss=best_loss)


def calibrate_success(logits, ok, valid, ca):
    p = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    # Match the existing logistic head's first-draw Platt calibration.
    for m in range(p.shape[1]):
        ii = ca[valid[ca, m, 0]]
        y = ok[ii, m, 0].astype(int)
        if len(ii) >= 20 and len(np.unique(y)) == 2:
            model = LogisticRegression(C=1e6, max_iter=2000).fit(logits[ii, m, None], y)
            p[:, m] = model.predict_proba(logits[:, m, None])[:, 1]
    return p


def output_cost(pred_log, target_log, outm, inp, pin, pout, tr, available):
    tokens = np.exp(np.clip(pred_log, -10, 20))
    for m in range(tokens.shape[1]):
        ii = tr[available[tr, m]]
        tokens[:, m] *= np.exp(np.clip(target_log[ii, m] - pred_log[ii, m], -20, 20)).mean()
        tokens[:, m] *= outm[ii, m].mean() / tokens[ii, m].mean()
    return inp * pin + tokens * pout


def decisions(P, C, Q, Cr, available, ii):
    choice = np.where(available[ii][None], P[ii][None] * VS[:, None, None] - C[ii][None], -np.inf).argmax(2)
    rows = np.arange(len(ii))[None]
    return Q[ii][rows, choice], Cr[ii][rows, choice]


def compare_frontiers(curves, bootstrap=500):
    def summary(ii):
        hs = {k: hull(zip(c[:, ii].mean(1), a[:, ii].mean(1))) for k, (a, c) in curves.items()}
        low = max(h[0][1] for h in hs.values())
        high = min(h[-1][1] for h in hs.values())
        if high <= low:
            return None
        targets = np.linspace(low + .05 * (high-low), high - .05 * (high-low), 12)
        base = np.array([cost_at(hs["linear"], t) for t in targets])
        gains = {k: float(1 - np.exp(np.log(np.array([cost_at(h, t) for t in targets]) / base).mean())) for k, h in hs.items()}
        return gains, targets.tolist()
    n = next(iter(curves.values()))[0].shape[1]
    point = summary(np.arange(n))
    if point is None:
        raise ValueError("No shared test accuracy band")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(bootstrap):
        result = summary(rng.integers(n, size=n))
        if result is not None:
            boots.append(result[0])
    return dict(accuracy_targets=point[1], bootstrap_valid=len(boots),
                arms={k: dict(cost_saved_vs_linear=g, ci95=np.percentile([b[k] for b in boots], [2.5, 97.5]).tolist()) for k, g in point[0].items()})


def deployable(curves_cal, curves_test, targets):
    """Choose mixtures on calibration once; report achieved test accuracy/spend.

    Test outcomes never select the operating points or mixture weights.
    """
    result = {}
    for target in targets:
        arms = {}
        for name, (a, c) in curves_cal.items():
            ma, mc = a.mean(1), c.mean(1)
            h = hull(zip(mc, ma))
            if not h[0][1] <= target <= h[-1][1]:
                arms[name] = dict(reachable_on_calibration=False)
                continue
            left, right = h[0], h[0]
            for x, y in zip(h, h[1:]):
                if x[1] <= target <= y[1]:
                    left, right = x, y
                    break
            i = int(np.argmin((mc-left[0])**2 + (ma-left[1])**2))
            j = int(np.argmin((mc-right[0])**2 + (ma-right[1])**2))
            w = float((target-left[1]) / (right[1]-left[1])) if right[1] > left[1] else 0.0
            at, ct = curves_test[name]
            ai, ci = (1-w)*at[i]+w*at[j], (1-w)*ct[i]+w*ct[j]
            arms[name] = dict(reachable_on_calibration=True, V=[float(VS[i]), float(VS[j])], mixture_weight=w,
                              test_accuracy=float(ai.mean()), test_cost_cents=float(ci.mean()),
                              per_problem_accuracy=ai.tolist(), per_problem_cost_cents=ci.tolist())
        if arms["linear"].get("reachable_on_calibration"):
            rng = np.random.default_rng(0)
            base = arms["linear"]
            for name, arm in arms.items():
                if not arm.get("reachable_on_calibration"):
                    continue
                da = np.array(arm["per_problem_accuracy"]) - base["per_problem_accuracy"]
                dc = np.array(arm["per_problem_cost_cents"]) - base["per_problem_cost_cents"]
                ii = rng.integers(len(da), size=(500, len(da)))
                arm["accuracy_difference_vs_linear"] = float(da.mean())
                arm["accuracy_difference_ci95"] = np.percentile(da[ii].mean(1), [2.5, 97.5]).tolist()
                arm["cost_difference_vs_linear_cents"] = float(dc.mean())
                arm["cost_difference_ci95_cents"] = np.percentile(dc[ii].mean(1), [2.5, 97.5]).tolist()
        result[str(target)] = arms
    return result


def train_one(label, output, epochs, task, config_index, seed):
    """One independent GPU job per dataset/head/configuration/seed."""
    name, act, _, _ = POOLS[label]
    folder = R / name
    dest = output / label / "training"
    dest.mkdir(parents=True, exist_ok=True)
    data = np.load(folder / "tensors.npz", allow_pickle=True)
    ids = list(map(str, data["problem_ids"]))
    pi = {p: i for i, p in enumerate(ids)}
    sp = json.loads((folder / "split_manifest.json").read_text())
    tr, ca = [np.array([pi[str(p)] for p in sp[k+"_problem_ids"]]) for k in ["train", "calibration"]]
    valid = data["valid"].astype(bool)
    n = valid.sum(2)
    available = n > 0
    Q = (data["final_outcome"].astype(bool) & valid).sum(2) / np.maximum(n, 1)
    outm = np.where(valid, data["completion_tokens"], 0).sum(2) / np.maximum(n, 1)
    ylog = np.log(np.maximum(outm, 1))
    center = np.array([ylog[tr[available[tr, m]], m].mean() for m in range(Q.shape[1])])
    scale = np.array([max(ylog[tr[available[tr, m]], m].std(), 1e-6) for m in range(Q.shape[1])])
    Xraw = rich(R / act, ids)
    X = StandardScaler().fit(Xraw[tr]).transform(Xraw).astype(np.float32)
    y = Q if task == "success" else (ylog-center)/scale
    weight = n if task == "success" else available.astype(float)
    z, metadata = fit_mlp(torch.tensor(X, device="cuda"), y, weight, tr, ca, task, CONFIGS[config_index], seed, epochs)
    stem = dest / f"{task}_config{config_index}_seed{seed}"
    np.savez_compressed(str(stem)+".npz", logits=z, problem_ids=ids)
    # Completion marker is written last, atomically, for the aggregation job.
    temp = Path(str(stem)+".json.tmp")
    temp.write_text(json.dumps(metadata, indent=2))
    temp.replace(Path(str(stem)+".json"))
    print("TRAINING DONE", stem, flush=True)


def run(label, output, epochs, aggregate=False):
    name, act, current_cost, targets = POOLS[label]
    folder = R / name
    dest = output / label
    dest.mkdir(parents=True, exist_ok=True)
    data = np.load(folder / "tensors.npz", allow_pickle=True)
    ids = list(map(str, data["problem_ids"]))
    slots = list(map(str, data["model_slots"]))
    sp = json.loads((folder / "split_manifest.json").read_text())
    pi = {p: i for i, p in enumerate(ids)}
    tr, ca, te = [np.array([pi[str(p)] for p in sp[k+"_problem_ids"]]) for k in ["train", "calibration", "test"]]
    assert not (set(tr) & set(ca) or set(tr) & set(te) or set(ca) & set(te))
    valid = data["valid"].astype(bool)
    ok = data["final_outcome"].astype(bool) & valid
    n = valid.sum(2)
    available = n > 0
    assert available.any(1).all()
    Q = ok.sum(2) / np.maximum(n, 1)
    outm = np.where(valid, data["completion_tokens"], 0).sum(2) / np.maximum(n, 1)
    inp = np.where(valid, data["prompt_tokens"], 0).sum(2) / np.maximum(n, 1)
    pin = np.array([MK[s][0] for s in slots]) * 100/1e6
    pout = np.array([MK[s][1] for s in slots]) * 100/1e6
    Cr = np.where(available, inp * pin + outm * pout, 1e9)
    Xraw = rich(R / act, ids)
    X = StandardScaler().fit(Xraw[tr]).transform(Xraw).astype(np.float32)
    Xt = None if aggregate else torch.tensor(X, device="cuda")
    ylog = np.log(np.maximum(outm, 1))
    # Reconstruct the linear success arm with the same rich feature matrix.
    subprocess.run([sys.executable, "pipelinerl/swe/scripts/livecodebench/activation_content_preds.py",
                    "--activations", str(R / act), "--rich", "--select-C", "--tensors-dir", str(folder),
                    "--out", str(dest / "linear_success.jsonl")], check=True)
    Plinear = read_preds(dest / "linear_success.jsonl", ids, "p_successes")
    yh = np.zeros_like(ylog)
    for m in range(len(slots)):
        ii = tr[available[tr, m]]
        yh[:, m] = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(X[ii], ylog[ii, m]).predict(X)
    Clinear = output_cost(yh, ylog, outm, inp, pin, pout, tr, available)
    selections = {}
    preds = {}
    for task in ["success", "cost"]:
        center = np.array([ylog[tr[available[tr, m]], m].mean() for m in range(len(slots))])
        scale = np.array([max(ylog[tr[available[tr, m]], m].std(), 1e-6) for m in range(len(slots))])
        y = Q if task == "success" else (ylog-center)/scale
        weight = n if task == "success" else available.astype(float)
        options = []
        for config_index, config in enumerate(CONFIGS):
            if aggregate:
                runs = []
                for seed in SEEDS:
                    stem = dest / "training" / f"{task}_config{config_index}_seed{seed}"
                    saved = np.load(str(stem)+".npz", allow_pickle=True)
                    assert list(map(str, saved["problem_ids"])) == ids
                    runs.append((saved["logits"], json.loads(Path(str(stem)+".json").read_text())))
            else:
                runs = [fit_mlp(Xt, y, weight, tr, ca, task, config, seed, epochs) for seed in SEEDS]
            options.append((np.mean([meta["calibration_loss"] for _, meta in runs]), config, runs))
        _, chosen, runs = min(options, key=lambda v: v[0])
        selections[task] = dict(config=chosen, runs=[meta for _, meta in runs],
                                candidates=[dict(config=c, mean_calibration_loss=float(v)) for v, c, _ in options])
        if task == "success":
            preds[task] = [calibrate_success(z, ok, valid, ca) for z, _ in runs]
        else:
            preds[task] = [output_cost(z*scale+center, ylog, outm, inp, pin, pout, tr, available) for z, _ in runs]
    Pmlp = np.mean(preds["success"], axis=0)
    Cmlp = np.mean(preds["cost"], axis=0)
    arms = {"linear": (Plinear, Clinear), "mlp_success": (Pmlp, Clinear),
            "mlp_cost": (Plinear, Cmlp), "mlp_both": (Pmlp, Cmlp)}
    test = {k: decisions(p, c, Q, Cr, available, te) for k, (p, c) in arms.items()}
    cal = {k: decisions(p, c, Q, Cr, available, ca) for k, (p, c) in arms.items()}
    result = dict(pool=label, slots=slots, splits=dict(train=len(tr), calibration=len(ca), test=len(te)),
                  feature_dimension=X.shape[1], activation_file=act, selections=selections,
                  matched_accuracy=compare_frontiers(test), deployable=deployable(cal, test, targets),
                  caveat="Bootstrap conditions on fitted heads and calibration selections; frontier comparisons use test hulls. Deployable rows report achieved accuracy, not test-matched accuracy.")
    result["individual_seeds"] = {}
    for i, seed in enumerate(SEEDS):
        seed_arms = {"linear": arms["linear"], "mlp_success": (preds["success"][i], Clinear),
                     "mlp_cost": (Plinear, preds["cost"][i]), "mlp_both": (preds["success"][i], preds["cost"][i])}
        result["individual_seeds"][str(seed)] = compare_frontiers({k: decisions(p, c, Q, Cr, available, te) for k, (p, c) in seed_arms.items()}, bootstrap=200)
    oldP = read_preds(folder / "content_preds.jsonl", ids, "p_successes")
    oldC = read_preds(folder / current_cost, ids, "expected_costs") * 100
    result["published_baseline_audit"] = dict(success_max_absolute_difference=float(np.abs(oldP-Plinear).max()),
                                             cost_max_absolute_difference_cents=float(np.abs(oldC-Clinear).max()),
                                             comparison=compare_frontiers({"linear": test["linear"], "published": decisions(oldP, oldC, Q, Cr, available, te)}))
    result["prediction_metrics"] = {}
    for name_, p, c in [("linear", Plinear, Clinear), ("mlp", Pmlp, Cmlp)]:
        route_metrics = {}
        for m, s in enumerate(slots):
            ii = te[available[te, m]]
            prob = np.clip(p[ii, m], 1e-6, 1-1e-6)
            route_metrics[s] = dict(binomial_log_loss=float(-(Q[ii,m]*np.log(prob)+(1-Q[ii,m])*np.log(1-prob)).mean()),
                                   cost_dollar_r2=float(1-((c[ii,m]-Cr[ii,m])**2).sum()/max(((Cr[ii,m]-Cr[ii,m].mean())**2).sum(),1e-12)))
            jj = te[valid[te, m, 0]]
            if len(np.unique(ok[jj,m,0])) == 2:
                route_metrics[s]["first_draw_auc"] = float(roc_auc_score(ok[jj,m,0], p[jj,m]))
        result["prediction_metrics"][name_] = route_metrics
    np.savez_compressed(dest / "predictions.npz", problem_ids=ids, Plinear=Plinear, Clinear=Clinear,
                        Pmlp=Pmlp, Cmlp=Cmlp, Pseeds=preds["success"], Cseeds=preds["cost"])
    (dest / "results.json").write_text(json.dumps(result, indent=2))
    print(label, json.dumps(result["matched_accuracy"]), flush=True)
    del Xt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--pool", choices=list(POOLS))
    parser.add_argument("--task", choices=["success", "cost"])
    parser.add_argument("--config-index", type=int, choices=range(len(CONFIGS)), default=0)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=7200)
    a = parser.parse_args()
    if not a.aggregate:
        assert torch.cuda.is_available(), "Use an eai GPU job"
    torch.set_num_threads(8)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.task:
        if not a.pool:
            parser.error("--task requires --pool")
        train_one(a.pool, a.out, a.epochs, a.task, a.config_index, a.seed)
        return
    labels = [a.pool] if a.pool else list(POOLS)
    (a.out / "protocol.json").write_text(json.dumps(dict(configs=CONFIGS, seeds=SEEDS, epochs=a.epochs,
        learning_rate=1e-4, patience=50, pools=POOLS, success_loss="binomial BCE", cost_loss="train-standardized log-output MSE",
        selection="mean calibration loss across three seeds; epoch selected per seed", ensemble="average all three seeds",
        features="train-standardized concatenated mean and last readouts, all eight stored layers; no PCA"), indent=2))
    if a.aggregate:
        expected = [a.out / label / "training" / f"{task}_config{ci}_seed{seed}.json"
                    for label in labels for task in ["success", "cost"] for ci in range(len(CONFIGS)) for seed in SEEDS]
        deadline = time.monotonic() + a.wait_seconds
        while any(not path.exists() for path in expected):
            if time.monotonic() >= deadline:
                raise TimeoutError("Missing training results: " + str([str(p) for p in expected if not p.exists()]))
            print(f"Waiting: {sum(p.exists() for p in expected)}/{len(expected)} independent runs complete", flush=True)
            time.sleep(30)
    result = {}
    for label in labels:
        result[label] = run(label, a.out, a.epochs, aggregate=a.aggregate)
        (a.out / "results.json").write_text(json.dumps(result, indent=2))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
