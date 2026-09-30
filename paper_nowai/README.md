# One Prefill Prices a Pool of Reasoning Models

A four-page academic manuscript in the uploaded NOWAI template, including references. Authors are placeholders.

## Read and edit

- [Compiled four-page paper](main.pdf)
- [Filled Markdown draft](PAPER_DRAFT.md)
- [LaTeX source](main.tex)
- [Original project outline](PAPER_OUTLINE.md)

The workshop draft centers on one frozen prefill predicting costs across a pool of reasoning models. It compares our cost heads with adapted embedding and prompt-feature estimators, then uses headroom and success-derived pricing to explain where the approach helps. It omits the MLP comparison, onboarding, cascades, and pending experiments. The original outline is preserved separately.

## Figures

- [Available versus captured savings](figures/headroom_and_capture.png): [vector PDF](figures/headroom_and_capture.pdf)
- [Cost signal and estimator comparisons](figures/cost_signal_ablation.png): [vector PDF](figures/cost_signal_ablation.pdf)
- [Computed values and provenance](figures/data_manifest.json)

Regenerate figures with Python, NumPy, and Matplotlib:

```bash
python make_plots.py
```

The `data/` directory contains the selected rows from the project's saved `analysis/cost_headroom/` analyses, including bootstrap draws used for the paired cost-from-success differences. These snapshots reproduce the plotted summaries; they are not the raw benchmark generations or a complete experiment rerun. The script also supports using the repository analyses when no bundled data directory exists.

## Compile

Upload the source ZIP to Overleaf, or compile locally:

```bash
latexmk -pdf main.tex
```

Alternatively, use `tectonic main.tex`. The PDF was compiled with Tectonic 0.17.0 and visually checked at four pages. Required template files and figure PDFs are included. `PaperForReview.tex` is a compatibility entry point that includes `main.tex`.

## Reporting choices

- LiveCodeBench's 46.0% oracle result uses the same accuracy band as its 35.6% plain-ridge result.
- Main savings are descriptive, matched-accuracy test-frontier comparisons, with 500 paired problem bootstrap resamples.
- ZeroRouter is identified as a paper-based deployment-pool reimplementation. Its configuration is selected on calibration; significance is claimed only where paired intervals support it. Component-swap bars have no significance claim.
- Encoder overhead is excluded from generation-spend savings and stated explicitly in the manuscript.
- Bibliographic metadata was checked against arXiv. `references.bib` abbreviates long author lists for the page limit; `references_full.bib` preserves full metadata.


- The estimator baselines are adaptations, not complete MixLLM or CARROT reproductions. Omni estimator differences are inconclusive; no uniform superiority claim is made.
- Activation-based length prediction is credited to ALPS and EGTP; the contribution is shared-prefill cross-model pricing and its controlled routing evaluation.
