"""Figures for the cost-routing paper (PAPER_OUTLINE.md). Writes PNGs to analysis/figures/.
Numbers either read from the analysis JSONs (path in each function) or, where a figure collects results reported across
several NEW_PATH.md sections, copied from those sections (cited inline).
Usage: python analysis/figures/make_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; A = H.parent / "cost_headroom"
INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C_OURS, C_BASE, C_ALT, C_ORACLE, C_WARN = "#2563eb", "#9ca3af", "#d97706", "#059669", "#dc2626"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    p = H / name; fig.savefig(p); plt.close(fig); print(p)


def fig_headroom():
    # NEW_PATH 4.A.4, 4.A.7, 4.A.9, 4.A.14, 4.A.19 (market prices)
    rows = [("MMLU-Pro", 65.1, 55.2, 71.2, "reasoning"), ("LCB", 47.3, 39.7, 52.9, "reasoning"), ("Omni-MATH", 44.2, 31.8, 53.5, "reasoning"),
            ("TACO", 36.1, 24.9, 46.0, "reasoning"), ("CodeContests", 21.8, 13.5, 29.1, "reasoning"), ("BCB", 20.2, 9.9, 29.8, "reasoning"),
            ("SWE-Smith", 14.2, -2.1, 29.6, "reasoning"), ("RouterBench (chat)", 10.5, 7.4, 12.8, "chat"),
            ("Agentic, 7 open models", 25.7, 6.3, 40.6, "agentic"), ("Agentic, all 13 (~190x prices)", 8.5, -5.0, 17.8, "agentic")]
    col = {"reasoning": C_OURS, "chat": C_BASE, "agentic": C_ALT}
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); y = np.arange(len(rows))[::-1]
    for yi, (n, v, lo, hi, g) in zip(y, rows):
        ax.barh(yi, v, color=col[g], height=0.62); ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=INK, lw=1, capsize=2)
        ax.text(max(hi, v) + 1.2, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel("Headroom: cost saved at matched accuracy by perfect per-query cost (%)")
    ax.set_xlim(-6, 80); ax.axvline(0, color=MUTED, lw=0.8)
    for g, lab in (("reasoning", "one-shot reasoning pools"), ("chat", "non-reasoning chat"), ("agentic", "agentic SWE")):
        ax.barh([], [], color=col[g], label=lab)
    ax.legend(loc="lower right", frameon=False); ax.set_title("Per-query cost is worth 20-65% for reasoning pools, ~10% for chat", loc="left", fontsize=10.5)
    save(fig, "fig1_headroom.png")


def fig_capture():
    d = json.load(open(A / "decompose_all.json")); want = {"pool_v2_tensors_5rung [market]": "LCB", "cc_tensors [market]": "CodeContests",
                                                           "taco_tensors_ha [market]": "TACO", "bcb_tensors_5r [market]": "BCB"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0)); cols = [C_OURS, C_ALT, C_WARN, C_ORACLE]
    for (key, lab), c in zip(want.items(), cols):
        r = next(x for x in d if x["pool"] == key and x.get("curve"))
        cv = sorted([z for z in r["curve"] if isinstance(z["rho"], (int, float))], key=lambda z: z["rho"]); x = [z["rho"] ** 2 for z in cv]; yv = [z["gain"] / r["headroom"] * 100 for z in cv]
        ax.plot(x, yv, "-o", color=c, ms=3, lw=1.6, label=f"{lab} (synthetic head)")
        r2 = np.mean([v["test_r2"] for v in r["routes"].values() if v.get("test_r2") is not None])
        ax.plot([r2], [r["learned_gain"] / r["headroom"] * 100], marker="*", ms=13, color=c, mec=INK, mew=0.6, ls="none")
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlabel("Cost-predictor quality (log-length R$^2$)"); ax.set_ylabel("Capture: share of headroom realised (%)")
    ax.plot([], [], marker="*", ms=11, color=MUTED, ls="none", label="real 4B-prefill head"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Convex capture: little gain below R$^2$ ~ .3-.5", loc="left", fontsize=10.5)
    save(fig, "fig2_capture_curve.png")


def fig_level_vs_diff():
    # NEW_PATH 4.A.17 (true components) and captured-by-predictor numbers
    pools = ["LCB", "Omni", "CodeContests", "BCB", "TACO"]; full = [46.0, 44.2, 21.8, 20.2, 36.1]
    level = [35.6, 21.0, 12.7, 9.8, -1.9]; diff = [36.5, 41.1, 19.3, 15.2, 30.0]
    x = np.arange(len(pools)); w = 0.26; fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x - w, full, w, color=C_ORACLE, label="full true cost"); ax.bar(x, level, w, color=C_OURS, label="true shared level only")
    ax.bar(x + w, diff, w, color=C_ALT, label="true between-model differences only")
    ax.set_xticks(x); ax.set_xticklabels(pools); ax.set_ylabel("Headroom vs paper rule (%)"); ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5); ax.set_title("'Long for everyone' (the level) is worth 10-36% on its own", loc="left", fontsize=10.5)
    save(fig, "fig3_level_vs_differences.png")


def fig_prereg():
    # NEW_PATH 4.A.9, 4.A.12, 4.A.13, 4.A.19: screen R2 (decision variable) and realised gain where the full pool ran
    calls = [("Omni", 0.67, 21.9, 8.9, 33.5), ("MMLU-Pro", 0.65, 30.7, 16.3, 43.2)]
    pending = [("APPS, K&K", 0.61), ("SuperGPQA", 0.58), ("OlympiadBench", 0.48), ("BBEH", 0.28), ("AIME", 0.13)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axvspan(0, 0.35, color="#fee2e2", alpha=0.6, lw=0); ax.axvspan(0.50, 0.8, color="#dbeafe", alpha=0.6, lw=0)
    ax.text(0.175, 44, "predict NO GAIN", ha="center", color=C_WARN, fontsize=9); ax.text(0.65, 44, "predict GAIN (>=10%, CI>0)", ha="center", color=C_OURS, fontsize=9)
    ax.axhline(10, color=C_OURS, lw=0.8, ls="--")
    for n, r2, g, lo, hi in calls:
        ax.errorbar(r2, g, yerr=[[g - lo], [hi - g]], fmt="o", color=C_OURS, ms=7, capsize=3); ax.text(r2 + 0.01, g + 1.5, f"{n} (confirmed)", fontsize=9, color=INK)
    for i, (n, r2) in enumerate(pending):
        ax.plot(r2, -4, marker="v", color=MUTED, ms=7); ax.text(r2 + (0.035 if n.startswith("APPS") else -0.035 if n == "SuperGPQA" else 0), -8.5 - 3.4 * (i % 2), n, ha="center", fontsize=8, color=MUTED)
    ax.set_xlim(0.05, 0.8); ax.set_ylim(-16, 48); ax.set_xlabel("Screen: probe CV log-length R$^2$ on gpt-oss-20b-low (committed before full pools)")
    ax.set_ylabel("Realised gain vs paper rule (%)"); ax.set_title("Pre-registered calls: 2 confirmed, 6 pending (triangles; APPS running)", loc="left", fontsize=10.5)
    save(fig, "fig4_preregistration.png")


def fig_onboarding():
    names = {"pool_v2_tensors_5rung": "LCB", "omni500_tensors": "Omni", "mmlupro_tensors": "MMLU-Pro"}
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, (p, lab) in zip(axs, names.items()):
        d = json.load(open(A / f"onboard_vs_zr_{p}.json")); routes = sorted({k.split("|")[0] for k in d if "|" in k}); ks = [5, 10, 20, 50, 200]
        for arm, c, nm in (("ours", C_OURS, "ours (1+2 params vs the latent)"), ("zr", C_ALT, "ZeroRouter-style (1-D IRT)"), ("naive", C_BASE, "naive (median + base rate)")):
            y = [np.mean([d[f"{r}|{k}"][arm][0] for r in routes if f"{r}|{k}" in d]) * 100 for k in ks]
            ax.plot(ks, y, "-o", color=c, ms=4, lw=1.7, label=nm)
        ax.axhline(d["full"][0] * 100, color=C_ORACLE, ls="--", lw=1, label="all models fully trained")
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_title(lab, loc="left", fontsize=10.5); ax.set_xlabel("labelled examples for the new model (k)")
    axs[0].set_ylabel("Gain vs paper rule (%), mean over held-out models"); axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig5_onboarding.png")


def fig_zerorouter():
    # zr_repro.json (D=1, K=10, seed 0) and the pools' own full-head numbers
    d = json.load(open(A / "zr_repro.json")); pools = ["LCB", "Omni", "MMLU-Pro"]
    arms = [("ours", "ours", C_OURS), ("zr succ + our cost", "their success + our cost", "#60a5fa"),
            ("our succ + zr cost", "our success + their pricing", C_ALT), ("zr", "ZeroRouter (full)", C_WARN)]
    x = np.arange(3); w = 0.2; fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, (k, lab, c) in enumerate(arms):
        v = [(d[p]["ours"] if k == "ours" else d[p]["D1|s0|K10"][k]) * 100 for p in pools]
        ax.bar(x + (i - 1.5) * w, v, w, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(pools); ax.set_ylabel("Gain vs paper rule (%)"); ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, ncol=2); ax.set_title("ZeroRouter on a 5-model pool: the gap is its pricing", loc="left", fontsize=10.5)
    save(fig, "fig6_zerorouter_component_swap.png")


def fig_budget():
    d = json.load(open(A / "budget_cap.json")); fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, p in zip(axs, ["LCB", "Omni", "MMLU-Pro"]):
        hmap = d[p]["hard"]; X = sorted(float(k) for k in hmap)
        for arm, c, lab in (("oracle", C_ORACLE, "oracle length"), ("ours", C_OURS, "ours (per-query length)"), ("constant", C_BASE, "constant per model"), ("median", C_ALT, "median rule")):
            ax.plot(X, [hmap[str(x) if str(x) in hmap else repr(x)][arm][0] * 100 for x in X], "-", color=c, lw=1.7, label=lab)
        ax.set_xscale("log"); ax.set_title(p, loc="left", fontsize=10.5); ax.set_xlabel("per-query budget X (cents, enforced)")
    axs[0].set_ylabel("Test accuracy (%)"); axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig7_per_query_budget.png")


def fig_cascade():
    j = json.load(open(A / "judge_cascade.json"))["matched_test_hull"]["0.0"]
    arms = [("paper", "paper rule"), ("cascade[4B]", "cascade, 4B judge"), ("cascade[137M]", "cascade, FrugalGPT-style 137M judge"),
            ("cascade[oracle]", "cascade, PERFECT judge"), ("hybrid[4B]", "router + 4B judge"), ("hybrid[137M]", "router + 137M judge")]
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); y = np.arange(len(arms))[::-1]
    for yi, (k, lab) in zip(y, arms):
        v, lo, hi = [x * 100 for x in j[k]]; c = C_ORACLE if "oracle" in k else (C_OURS if "hybrid" in k else (C_BASE if k == "paper" else C_ALT))
        ax.barh(yi, v, color=c, height=0.6); ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=INK, lw=1, capsize=2)
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(y); ax.set_yticklabels([a[1] for a in arms])
    ax.set_xlabel("Cost saved vs the one-shot prefill router at matched accuracy (%)")
    ax.set_title("LCB, no verifier, one submission: nothing beats the router", loc="left", fontsize=10.5)
    save(fig, "fig8_cascades.png")


def fig_mmlupro_drivers():
    # NEW_PATH 4.A.28 (mean over routes of test R2 of log length)
    labs = ["success-head\ndifficulty", "TRUE\ndifficulty", "subject", "difficulty\n+ subject", "+ source\n+ options", "4B cost\nprobe"]
    per = [[.14, .25, .25, .21, .36], [.02, .19, .27, .12, .33], [.22, .17, .16, .25, .20], [.36, .37, .30, .43, .42], [.39, .39, .31, .47, .42], [.51, .58, .20, .66, .50]]
    fig, ax = plt.subplots(figsize=(6.6, 3.3)); x = np.arange(len(labs))
    ax.bar(x, [np.mean(p) for p in per], color=[C_BASE, C_BASE, C_ALT, C_ALT, C_ALT, C_OURS], width=0.62)
    for i, p in enumerate(per):
        ax.scatter([i] * len(p), p, color=INK, s=9, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylabel("Test R$^2$ of log output length")
    ax.set_title("MMLU-Pro: length is 'work required', not difficulty", loc="left", fontsize=10.5)
    save(fig, "fig9_mmlupro_length_drivers.png")


if __name__ == "__main__":
    for f in (fig_headroom, fig_capture, fig_level_vs_diff, fig_prereg, fig_onboarding, fig_zerorouter, fig_budget, fig_cascade, fig_mmlupro_drivers):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__}: FAILED {type(e).__name__}: {e}")


def fig_headline_zerorouter():
    """Headline, two panels. Left: cost saved vs median-length routing (averaged over the accuracy range both reach) for ours,
    ZeroRouter with its configuration chosen on CALIBRATION (4B reader), and ZeroRouter with its own DistilBERT encoder; paired
    ours - ZeroRouter [95% CI] annotated (zr_best_ci.json). Right: ZeroRouter's cost relative to ours at each accuracy, with a
    95% paired-bootstrap band (zr_ratio_curve.json); above 1 = ZeroRouter more expensive."""
    ci = json.load(open(A / "zr_best_ci.json")); enc = json.load(open(A / "zr_encoder_eval.json")); rc = json.load(open(A / "zr_ratio_curve.json"))
    pools = ["LCB", "Omni", "MMLU-Pro"]; pc = {"LCB": C_OURS, "Omni": C_ORACLE, "MMLU-Pro": C_WARN}
    ours = [ci[p]["calibration-chosen"]["ours"] * 100 for p in pools]; zr = [ci[p]["calibration-chosen"]["zr"] * 100 for p in pools]
    own = [max(max(v[str(K)] for K in (5, 10, 20)) for k, v in enc[p].items() if "DistilBERT" in k) * 100 for p in pools]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 4.1), gridspec_kw={"width_ratios": [1.05, 1]})
    x = np.arange(3); w = 0.26
    ax.bar(x - w, ours, w, color=C_OURS, label="Ours (frozen 4B prefill)")
    ax.bar(x, zr, w, color=C_ALT, label="ZeroRouter, tuned on calibration")
    ax.bar(x + w, own, w, color="#fbbf24", label="ZeroRouter, own DistilBERT encoder")
    for i, p in enumerate(pools):
        for dx, val, bold in ((-w, ours[i], True), (0, zr[i], False), (w, own[i], False)):
            ax.text(x[i] + dx, val + 0.8, f"{val:.0f}%", ha="center", fontsize=8.5, color=INK, weight="bold" if bold else "normal")
        d = ci[p]["calibration-chosen"]; lo, hi = d["ci"]
        ax.text(x[i], max(ours[i], zr[i], own[i]) + 6.0, f"{d['diff']:+.1f} pt\n[{lo:+.1f}, {hi:+.1f}]" + ("" if lo > 0 else "\nn.s."),
                ha="center", fontsize=8.3, color=INK if lo > 0 else MUTED)
    ax.set_xticks(x); ax.set_xticklabels(pools, fontsize=10.5); ax.set_ylim(0, 56)
    ax.set_ylabel("Cost saved vs median-length routing (%)"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(a) Averaged over the accuracy range", loc="left", fontsize=10.5)
    for p in pools:
        r = rc[p]; a_ = np.array(r["acc"]) * 100
        bx.fill_between(a_, r["lo"], r["hi"], color=pc[p], alpha=0.15, lw=0); bx.plot(a_, r["ratio"], "-", color=pc[p], lw=2, label=p)
    bx.axhline(1, color=INK, lw=1); bx.set_xlabel("Test accuracy (%)"); bx.set_ylabel("ZeroRouter cost / our cost")
    bx.text(0.02, 0.97, "above 1: ZeroRouter more expensive", transform=bx.transAxes, fontsize=8, color=MUTED, va="top")
    bx.legend(frameon=False, fontsize=8.5, loc="upper right"); bx.set_title("(b) At each accuracy (95% paired band)", loc="left", fontsize=10.5)
    fig.suptitle("Ours vs ZeroRouter (reproduced, tuned) on a 5-model reasoning pool", x=0.01, ha="left", fontsize=11.5)
    save(fig, "fig0_headline_vs_zerorouter.png")


if __name__ == "__main__":
    fig_headline_zerorouter()
