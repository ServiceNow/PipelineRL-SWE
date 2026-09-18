#!/usr/bin/env python3
"""Trajectory figures: where each policy spends, and when it gives up.

Reads the per-episode dump of one replay (--dump-trajectories) and its replay_results.json. Each
method is shown at ONE deterministic operating point: the cheapest one whose accuracy reaches
gpt-oss-120b's own accuracy on this seed (the Figure 1 target). Our method is the union of the
price arm and the two-constraint arm, as in the tables.

  traj_giveup  -- among problems NOT solved: number of draws taken before stopping, stacked by the
                  model of the last draw, with the mean spend on those problems. For baselines,
                  stopping means the cap bound or the draws ran out; for ours it is mostly the
                  give-up action. Like for like: every unsolved problem ends with no answer.
  traj_flow    -- the weighted graph: draw 1 -> draw 2 -> ...; nodes are the model drawn, or the
                  ending (solved / stopped); band width = number of episodes.
  traj_draws   -- draws per model per problem, per method.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

INK, INK2, RULE = "#1d2129", "#5b616e", "#c9ccd2"
ROUTE = ["scout (4B)", "gpt-oss-20b", "gpt-oss-120b"]
RCOL = ["#8fbff0", "#2a78d6", "#123f78"]           # one hue, light -> dark = cheap -> dear
SOLVED, STOPPED = "#1baf7a", "#eb6834"
METHODS = [  # label, policies
    ("ours", ("content_decay_qcost_value", "content_decay_qcost_cappedvalue")),
    ("count-belief greedy (RoR v1)", ("counts",)),
    ("random allocation", ("random_allocation",)),
]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE,
    "grid.linewidth": 0.5,
})


def pick_points(results, target):
    """Cheapest deterministic operating point at or above the target accuracy, per method."""
    chosen = {}
    for label, pols in METHODS:
        cand = [r for r in results if r["policy"] in pols and r["correctness"] >= target]
        if not cand:
            continue
        r = min(cand, key=lambda x: x["mean_realized_cost"])
        chosen[label] = (r["policy"], r.get("budget"), r.get("value_of_correct"),
                         r["correctness"], r["mean_realized_cost"])
    return chosen


def key(policy, budget, voc):
    f = lambda x: None if x is None else round(float(x), 12)
    return (policy, f(budget), f(voc))


def load_episodes(path, wanted):
    eps = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            e = json.loads(line)
            k = key(e["policy"], e.get("budget"), e.get("value_of_correct"))
            if k in wanted:
                eps[wanted[k]].append(e)
    return eps


def fig_giveup(eps, chosen, out):
    labels = [m for m, _ in METHODS if m in eps]
    fig, axes = plt.subplots(1, len(labels), figsize=(7.4, 2.5), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    maxd = 8
    for ax, lab in zip(np.atleast_1d(axes), labels):
        fail = [e for e in eps[lab] if not e["correct"]]
        n = len(eps[lab])
        H = np.zeros((maxd + 1, 3))
        for e in fail:
            seq = e["route_sequence"] or []
            d = min(len(seq), maxd)
            if seq:
                H[d, seq[-1]] += 1
        bottom = np.zeros(maxd + 1)
        x = np.arange(maxd + 1)
        for m in range(3):
            ax.bar(x, 100 * H[:, m] / n, bottom=bottom, color=RCOL[m], width=0.8,
                   label=ROUTE[m], edgecolor="white", linewidth=0.4)
            bottom += 100 * H[:, m] / n
        spent = np.mean([e["realized_spend"] for e in fail]) if fail else 0.0
        share = sum(e["realized_spend"] for e in fail) / max(1e-12, sum(e["realized_spend"] for e in eps[lab]))
        _name = {"count-belief greedy (RoR v1)": "RoR v1 (count beliefs)"}.get(lab, lab)
        ax.set_title(f"{_name}\n{100*share:.0f}% of spend on unsolved", loc="left", color=INK,
                     fontsize=7.2)
        ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(maxd)] + [f"{maxd}+"])
    fig.supxlabel("draws taken before the episode ended unsolved", fontsize=7.5, y=-0.04)
    np.atleast_1d(axes)[0].set_ylabel("% of all problems")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, [f"last draw: {x}" for x in l], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.1))
    fig.savefig(out / "traj_giveup.pdf"); fig.savefig(out / "traj_giveup.svg"); fig.savefig(out / "traj_giveup.png", dpi=160)
    plt.close(fig)


def _band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.45):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig_flow(eps, out, stages=6):
    """Weighted graph of trajectories. Column k holds the episodes still running at draw k, split
    by the model drawn; episodes that ended flow to SOLVED / STOPPED nodes at the far right."""
    labels = [m for m, _ in METHODS if m in eps]
    fig, axes = plt.subplots(len(labels), 1, figsize=(6.75, 1.55 * len(labels)),
                             gridspec_kw={"hspace": 0.55})
    for ax, lab in zip(np.atleast_1d(axes), labels):
        E = eps[lab]; n = len(E)
        # node(k, state): state in 0..2 (model drawn at step k) ; terminal nodes after
        trans = Counter()
        for e in E:
            seq = (e["route_sequence"] or [])[:stages]
            end = "solved" if e["correct"] else "stopped"
            full = e["route_sequence"] or []
            for k in range(len(seq) - 1):
                trans[(k, seq[k], seq[k + 1])] += 1
            if seq:
                last_k = len(seq) - 1
                if len(full) > stages:       # still running past the last column
                    trans[(last_k, seq[-1], "more")] += 1
                else:
                    trans[(last_k, seq[-1], end)] += 1
            else:
                trans[(-1, None, end)] += 1
        gap = 0.02 * n
        # node sizes per column
        col = [np.zeros(3) for _ in range(stages)]
        for (k, a, b), c in trans.items():
            if k >= 0:
                col[k][a] += c
        term = Counter()
        for (k, a, b), c in trans.items():
            if b in ("solved", "stopped", "more"):
                term[b] += c
        # layout: y positions (stacked, top-down)
        def stack(sizes):
            ys, y = [], 0.0
            for s in sizes:
                ys.append((y, y + s)); y += s + (gap if s > 0 else 0)
            return ys
        pos = [stack(col[k]) for k in range(stages)]
        tpos = dict(zip(("solved", "stopped", "more"), stack([term["solved"], term["stopped"], term["more"]])))
        X = list(range(stages)) + [stages + 0.6]
        w = 0.16
        # bands: consume node capacity top-down
        out_off = [np.array([p[0] for p in pos[k]]) for k in range(stages)]
        in_off = [np.array([p[0] for p in pos[k]]) for k in range(stages)]
        t_in = {t: tpos[t][0] for t in tpos}
        for k in range(stages):
            for a in range(3):
                for b in (0, 1, 2, "solved", "stopped", "more"):
                    c = trans.get((k, a, b), 0)
                    if not c:
                        continue
                    y0a = out_off[k][a]; y0b = y0a + c; out_off[k][a] = y0b
                    if b in (0, 1, 2) and k + 1 < stages:
                        y1a = in_off[k + 1][b]; y1b = y1a + c; in_off[k + 1][b] = y1b
                        _band(ax, X[k] + w / 2, y0a, y0b, X[k + 1] - w / 2, y1a, y1b, RCOL[b])
                    else:
                        y1a = t_in[b]; y1b = y1a + c; t_in[b] = y1b
                        colr = SOLVED if b == "solved" else (STOPPED if b == "stopped" else "#b4b8bf")
                        _band(ax, X[k] + w / 2, y0a, y0b, X[-1] - w / 2, y1a, y1b, colr, alpha=0.30)
        for k in range(stages):
            for a in range(3):
                y0, y1 = pos[k][a]
                if y1 > y0:
                    ax.add_patch(plt.Rectangle((X[k] - w / 2, y0), w, y1 - y0, color=RCOL[a], lw=0))
        for t, colr in (("solved", SOLVED), ("stopped", STOPPED), ("more", "#b4b8bf")):
            y0, y1 = tpos[t]
            if y1 > y0:
                ax.add_patch(plt.Rectangle((X[-1] - w / 2, y0), w, y1 - y0, color=colr, lw=0))
                ax.text(X[-1] + w, (y0 + y1) / 2,
                        f"{'still drawing' if t == 'more' else t} {100*(y1-y0)/n:.0f}%",
                        va="center", fontsize=7, color=INK)
        ax.set_xlim(-0.4, X[-1] + 1.3)
        ax.set_ylim(max(max(p[-1][1] for p in pos), tpos["more"][1]) + gap, -gap)
        ax.set_xticks(X); ax.set_xticklabels([f"draw {k+1}" for k in range(stages)] + ["end"])
        ax.set_yticks([]); ax.grid(False)
        for sp_ in ("left",):
            ax.spines[sp_].set_visible(False)
        _sol = 100 * np.mean([e["correct"] for e in E])
        _abs = 100 * np.mean([e.get("abstained", False) for e in E])
        ax.set_title(f"{lab}: {_sol:.0f}% solved in total, {_abs:.0f}% gave up by choice, "
                     f"{100 - _sol - _abs:.0f}% ran out", loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in RCOL]
    fig.legend(handles, ROUTE, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / "traj_flow.pdf"); fig.savefig(out / "traj_flow.svg"); fig.savefig(out / "traj_flow.png", dpi=160)
    plt.close(fig)


def fig_draws(eps, out):
    labels = [m for m, _ in METHODS if m in eps]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.1), sharey=True, gridspec_kw={"wspace": 0.1})
    cols = ["#2a78d6", "#eb6834", "#eda100", "#8a9099"]
    x = np.arange(7); w = 0.8 / len(labels)
    for m, ax in enumerate(axes):
        for i, lab in enumerate(labels):
            cnt = Counter(min(6, (e["route_sequence"] or []).count(m)) for e in eps[lab])
            n = len(eps[lab])
            ax.bar(x + (i - (len(labels) - 1) / 2) * w, [100 * cnt[j] / n for j in x], w,
                   color=cols[i % len(cols)], label=lab)
        ax.set_title(ROUTE[m], loc="left", color=INK)
        ax.set_xticks(x); ax.set_xlabel("draws on this model per problem")
    axes[0].set_ylabel("% of problems")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(out / "traj_draws.pdf"); fig.savefig(out / "traj_draws.svg"); fig.savefig(out / "traj_draws.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--results", required=True, help="replay_results.json of the same run")
    ap.add_argument("--out", default="overleaf/figures")
    ap.add_argument("--target", type=float, default=None,
                    help="accuracy target; default gpt-oss-120b's own accuracy in this run")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())["results"]
    target = a.target
    if target is None:
        target = next(r["correctness"] for r in res if r["policy"] == "single_oss120")
    chosen = pick_points(res, target)
    print(f"target accuracy {target:.4f}")
    for lab, c in chosen.items():
        print(f"  {lab:32s} {c[0]:34s} budget={c[1]} R={c[2]} acc={c[3]:.4f} cost=${c[4]:.5f}")
    wanted = {key(c[0], c[1], c[2]): lab for lab, c in chosen.items()}
    eps = load_episodes(a.episodes, wanted)
    for lab in chosen:
        print(f"  {lab:32s} {len(eps.get(lab, []))} episodes")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_giveup(eps, chosen, out)
    fig_flow(eps, out)
    fig_draws(eps, out)
    print("wrote traj_giveup / traj_flow / traj_draws ->", out)


if __name__ == "__main__":
    main()
