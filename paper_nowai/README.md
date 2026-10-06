# Predicting Reasoning-Model Costs from Shared Prefill Activations

A four-page academic manuscript (excluding references) in the uploaded NOWAI template. Authors are placeholders.

## Read and edit

- [Compiled four-page paper](main.pdf)
- [Filled Markdown draft](PAPER_DRAFT.md)
- [LaTeX source](main.tex)
- [Original project outline](PAPER_OUTLINE.md)

The workshop draft centers on one frozen prefill predicting costs across a pool of reasoning models. It compares our cost heads with adapted embedding and prompt-feature estimators, then uses headroom and success-derived pricing to explain where the approach helps. The main paper omits the MLP comparison, onboarding, cascades, and pending collection results. The old two-page supplement (unpinned, list prices) is archived in archive/ and is not part of the submission. The original outline is preserved separately.

## Figures

- [Figure 1: shared-prefill overview and controlled savings](figures/shared_prefill_overview.png): [vector PDF](figures/shared_prefill_overview.pdf)
- [Cost signal and estimator comparisons](figures/cost_signal_ablation.png): [vector PDF](figures/cost_signal_ablation.pdf)
- [Computed values and provenance](figures/data_manifest.json)

Regenerate figures with Python, NumPy, and Matplotlib:

```bash
python make_plots.py
python make_billed_curves.py   # Figure 1(b) data at billed prices (reads analysis/cost_headroom)
python make_overview.py
python make_markdown.py
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
- Figure 1 and Tables 1-3 use billed prices; Table 3 prices every encoder pass. Both arms in Figure 1 share the same encoder pass.
- Bibliographic metadata was checked against arXiv. `references.bib` abbreviates long author lists for the page limit; `references_full.bib` preserves full metadata.


- The estimator baselines are adaptations, not complete MixLLM or CARROT reproductions. Omni estimator differences are inconclusive; no uniform superiority claim is made.
- Activation-based length prediction is credited to ALPS and EGTP; the contribution is shared-prefill cross-model pricing and its controlled routing evaluation.

## Current revision

Revision of 2026-10-05: original-pool savings repriced at billed rates (LCB 26.7,
Omni 25.2, MMLU-Pro 35.3); the drift offset is described as removing the bias on
average, with the single-fit interval; the provider-routing saving is replaced by
"endpoint = model + offset", since pinning the best provider chosen on fitting
problems beats the provider router by about 4%; the refit-bootstrap caveat (Omni vs
difficulty bins not significant) is stated. The extra page adds Table 3
(representation x readout grid, encoders priced), Table 4 (prefill size, Qwen3
0.6B-8B), a second-model-family paragraph and the cross-fitted-rates sentence.
Figure 1(b) now shows billed-price frontiers with matched-accuracy arrows labelled
with the cost saved. LaTeX comments `% TODO(offfamily)`, `% TODO(aime)`,
`% TODO(provider-cc)` and `% TODO(live)` mark slots for pending results. The main
text is four pages; references begin on page 4 and end on page 5.
