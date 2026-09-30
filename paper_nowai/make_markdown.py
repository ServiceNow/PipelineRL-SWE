"""Generate a readable Markdown companion from this paper's LaTeX source."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
tex = (ROOT / "main.tex").read_text()
title = re.search(r"\\title\{([^}]+)\}", tex)[1]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)[1].strip()
body = tex.split(r"\end{abstract}", 1)[1].split(r"\balance", 1)[0]


def figure(match):
    block = match[0]
    name = re.search(r"figures/([^}]+)\.pdf", block)[1]
    caption = block.split(r"\caption{", 1)[1].split(r"\label{", 1)[0].strip().removesuffix("}").strip()
    return f"\n\n![{name.replace('_', ' ')}](figures/{name}.png)\n\n{caption}\n\n"


body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", figure, body, flags=re.S)
table = """

| Baseline (its own gain: LCB / Omni / MMLU-Pro) | LCB | Omni | MMLU-Pro |
|---|---:|---:|---:|
| Ours (gain vs. median rule) | 35.6 [27.1, 41.9] | 21.9 [8.9, 33.5] | 30.7 [16.3, 43.2] |
| Mean per-model constant (1.3 / -1.4 / 1.2) | +34.3† | +23.3† | +29.5† |
| Single best model (-3.7 / -20.1 / 5.4) | +39.3† | +42.0† | +25.3† |
| MixLLM-style embedding ensemble (14.1 / 12.5 / 4.0) | +21.5† | +9.4 [-4.4, 24.2] | +26.7 [12.4, 40.2] |
| Prompt-feature GBM (12.3 / 6.6 / 9.8) | +23.3† | +15.2 [-1.4, 34.1] | +20.9 [3.8, 33.3] |
| Cost from the success head (34.1 / 21.0 / 9.2) | +1.5 [-1.3, 4.3] | +0.8 [-7.2, 9.0] | +21.5 [8.5, 30.3] |
| ZeroRouter-style bins on our difficulty, K = 5 (32.2 / 21.7 / 10.9) | +3.7 [0.1, 7.4] | -0.1 [-12.1, 10.7] | +19.6 [7.8, 32.7] |
| ZeroRouter-style bins on our difficulty, K = 10 (34.0 / 17.9 / 13.3) | +1.8 [-1.6, 5.1] | +3.5 [-8.2, 14.9] | +19.1 [7.6, 31.6] |
| ZeroRouter faithful, tuned on calibration (18.5 / 20.0 / 21.6) | +17.6 [9.8, 24.9] | +3.0 [-10.7, 16.6] | +9.3 [-2.6, 21.5] |
| ZeroRouter faithful, own DistilBERT encoder (12.7 / 13.0 / 12.1) | +23.2† | +9.8† | +18.8† |

**Table 1.** Canonical baselines at matched accuracy (test). Cells are ours minus the baseline's cost saved versus the median-length rule (the baseline's own gain in parentheses); brackets are paired 95% bootstrap intervals; † marks unpaired differences of separately reported gains. The two ZeroRouter families differ: "bins on our difficulty" is bin-lookup pricing driven by our shared-difficulty signal; "faithful" reproduces its IRT stage with configuration chosen on calibration. ZR-family rows use the ZeroRouter comparison's shared band (ours 36.0 / 22.7 / 30.9 within it).

"""
body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", lambda _: table, body, flags=re.S)
refs = {
    "prefill": ("Prefill router", "https://arxiv.org/abs/2603.20895"),
    "mixllm": ("MixLLM", "https://arxiv.org/abs/2502.18482"),
    "carrot": ("CARROT", "https://arxiv.org/abs/2502.03261"),
    "zerorouter": ("ZeroRouter", "https://arxiv.org/abs/2601.06220"),
    "lcb": ("LiveCodeBench", "https://arxiv.org/abs/2403.07974"),
    "omni": ("Omni-MATH", "https://arxiv.org/abs/2410.07985"),
    "mmlupro": ("MMLU-Pro", "https://arxiv.org/abs/2406.01574"),
    "routerbench": ("RouterBench", "https://arxiv.org/abs/2403.12031"),
    "codecontests": ("AlphaCode / CodeContests", "https://arxiv.org/abs/2203.07814"),
    "taco": ("TACO", "https://arxiv.org/abs/2312.14852"),
    "bigcodebench": ("BigCodeBench", "https://arxiv.org/abs/2406.15877"),
    "egtp": ("Entropy-guided length prediction", "https://arxiv.org/abs/2602.11812"),
    "alps": ("ALPS", "https://doi.org/10.5281/zenodo.19078431"),
}
link = lambda k: f"[{refs[k][0]}]({refs[k][1]})"
md = f"# {title}\n\n## Abstract\n\n{abstract}\n\n{body.strip()}"
md = re.sub(r"\\section\{([^}]+)\}", r"## \1\n", md)
md = re.sub(r"\\paragraph\{([^}]+)\}", r"### \1\n\n", md)
md = re.sub(r"\\(emph|textbf)\{([^{}]+)\}", lambda m: ("*" if m[1] == "emph" else "**") + m[2] + ("*" if m[1] == "emph" else "**"), md)
md = re.sub(r"~?\\cite\{([^}]+)\}", lambda m: " (" + ", ".join(link(k) for k in m[1].split(",")) + ")", md)
md = re.sub(r"\\[Cc]ref\{([^}]+)\}", lambda m: {"fig:headroom": "Figure 1", "fig:cost": "Figure 2", "tab:main": "Table 1"}[m[1]], md)
md = md.replace(r"\begin{equation}", "\n$$").replace(r"\end{equation}", "$$\n")
md = re.sub(r"\\label\{[^}]+\}\n?", "", md)
md = md.replace(r"\%", "%").replace("~", " ").replace("--", "–").replace(r"\!", "")
md = md.replace("|–-|–-:|–-:|–-:|", "|---|---:|---:|---:|").replace("|–-|–-:|–-:|", "|---|---:|---:|")
md = re.sub(r"\n{3,}", "\n\n", md)
md += "\n\n## References\n\n" + "\n".join("- " + link(k) for k in refs) + "\n"
(ROOT / "PAPER_DRAFT.md").write_text(md)
