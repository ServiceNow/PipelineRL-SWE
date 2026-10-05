"""Generate a readable Markdown companion from this paper's LaTeX source."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
tex = (ROOT / "main.tex").read_text()
tex = re.sub(r"\\input\{([^}]+)\}", lambda m: (ROOT / (m[1] + '.tex')).read_text(), tex)
tex = re.sub(r"(?<!\\)%.*", "", tex)                 # drop LaTeX comments (TODO slots), keep escaped \%
title = re.search(r"\\title\{([^}]+)\}", tex)[1]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)[1].strip()
body = tex.split(r"\end{abstract}", 1)[1].split(r"\balance", 1)[0]


def figure(match):
    block = match[0]
    name = re.search(r"figures/([^}]+)\.pdf", block)[1]
    caption = block.split(r"\caption{", 1)[1].split(r"\label{", 1)[0].strip().removesuffix("}").strip()
    return f"\n\n![{name.replace('_', ' ')}](figures/{name}.png)\n\n{caption}\n\n"


body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", figure, body, flags=re.S)
table_number = 0
def table(match):
    global table_number
    table_number += 1
    block = match[0]
    block = re.sub(r'\\shortstack\{([^{}]+)\}', lambda m: m[1].replace(r'\\', ' '), block)
    tabular = re.search(r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", block, re.S)[1]
    rows = []
    for line in tabular.splitlines():
        line = line.strip()
        if "&" not in line:
            continue
        line = re.sub(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{([^}]*)\}", lambda m: m[2] + " &" * (int(m[1]) - 1), line)
        cells = [cell.strip().removesuffix(r"\\").strip() for cell in line.split("&")]
        rows.append("| " + " | ".join(cells) + " |")
    rows.insert(1, "| " + " | ".join(["---"] + ["---:"] * (len(rows[0].split("|")) - 3)) + " |")
    caption = block.split(r"\caption{", 1)[1].split(r"\label{", 1)[0].strip().removesuffix("}").strip()
    return "\n\n" + "\n".join(rows) + f"\n\n**Table {table_number}.** " + caption + "\n\n"

body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", table, body, flags=re.S)
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
    "samemodel": ("Same model, not the same service", "https://arxiv.org/abs/2605.02821"),
    "gateways": ("API gateway consistency", "https://arxiv.org/abs/2604.21083"),
    "modelequality": ("Model equality testing", "https://arxiv.org/abs/2410.20247"),
    "facet": ("Market-aware provider routing (FACET)", "https://arxiv.org/abs/2609.37902"),
}
link = lambda k: f"[{refs[k][0]}]({refs[k][1]})"
md = f"# {title}\n\n## Abstract\n\n{abstract}\n\n{body.strip()}"
md = re.sub(r"\\section\{([^}]+)\}", r"## \1\n", md)
md = re.sub(r"\\paragraph\{([^}]+)\}", r"### \1\n\n", md)
md = re.sub(r"\\(emph|textbf)\{([^{}]+)\}", lambda m: ("*" if m[1] == "emph" else "**") + m[2] + ("*" if m[1] == "emph" else "**"), md)
md = re.sub(r"~?\\cite\{([^}]+)\}", lambda m: " (" + ", ".join(link(k) for k in m[1].split(",")) + ")", md)
md = re.sub(r"\\[Cc]ref\{([^}]+)\}", lambda m: ", ".join({"fig:overview": "Figure 1", "fig:headroom": "Supplementary Figure 1", "tab:fresh": "Table 1", "tab:policies": "Table 2", "tab:main": "Table 2", "tab:grid": "Table 3", "tab:size": "Table 4"}[k] for k in m[1].split(",")), md)
md = md.replace(r'\Delta', 'Δ')
md = md.replace(r"\begin{equation}", "\n$$").replace(r"\end{equation}", "$$\n")
md = re.sub(r"\\label\{[^}]+\}\n?", "", md)
md = md.replace("\\ ", " ").replace("$-$", "−").replace(r"\%", "%").replace("~", " ").replace("--", "–").replace(r"\!", "")
md = md.replace("|–-|–-:|–-:|–-:|", "|---|---:|---:|---:|").replace("|–-|–-:|–-:|", "|---|---:|---:|")
md = re.sub(r"(?m)^\|[ |:–-]+\|$", lambda m: m[0].replace("–", "--"), md)
md = re.sub(r"\n{3,}", "\n\n", md)
md += "\n\n## References\n\n" + "\n".join("- " + link(k) for k in refs) + "\n"
(ROOT / "PAPER_DRAFT.md").write_text(md)
