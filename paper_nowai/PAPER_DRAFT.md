# Reducing the Cost of LLM Coding and Reasoning Models with Routers

## Abstract

Routers typically choose a model per query by trading predicted correctness against cost. However, most price each model at a fixed cost, assuming that the number of tokens each model outputs does not vary significantly. We find that the output length of reasoning models varies by an order of magnitude across queries. We predict **both** success probability **and** route-specific output length from the prefill activations of a single small frozen 4B LLM, using a linear readout per route. On 6,500 MMLU-Pro, 1,000 Omni-MATH, and 341 LiveCodeBench tasks, predicted costs reduce spending at matched accuracy by 28.8%, 34.8% and 33.7% relative to median-length pricing. They also beat every alternative estimator we evaluated, with paired intervals excluding zero; pricing from difficulty alone matches them only where output length tracks difficulty. We then ablate across different small LLMs and show that every small LLM prefill we tried (0.6–8B, four model families) can be used as a router in this way.

## Introduction

Routing reasoning models requires predicting both correctness and cost before generation. Token rates are published; the number of tokens is not. A model that is cheap on average can be expensive on a particular problem, and the outputs of the models we study vary 10–90$\times$ across problems.

Prefill routers predict correctness from a small encoder's prompt activations; the prefill router of Varshney et al. ([Prefill router](https://arxiv.org/abs/2603.20895)) explicitly prices outputs at median training length. We read output length from the same activations (Figure 1). Cost prediction adds one linear readout per route to an existing prefill router, without another encoder pass.

We contribute: (i) evidence on held-out test problems at billed prices that prefill cost readouts reduce spending at matched accuracy and outperform mean-length, prompt-feature, text-embedding and ZeroRouter cost estimators; (ii) a mechanism, from ablations that price from our own success readouts: difficulty suffices where output length tracks it and fails where it does not; (iii) controls showing the gain is the prefill representation, holds for every small prefill we tried (0.6–8B, four families) and transfers to two further routed model families; and, as a side note, (iv) the same model's provider endpoints drift, which the readouts can absorb with one offset per endpoint; we pin providers instead.

### Related work.

MixLLM predicts output length from query embeddings ([MixLLM](https://arxiv.org/abs/2502.18482)); CARROT estimates cost with embedding nearest neighbours or a fine-tuned RoBERTa ([CARROT](https://arxiv.org/abs/2502.03261)); ZeroRouter prices queries by item-response difficulty bins ([ZeroRouter](https://arxiv.org/abs/2601.06220)). Activation-based length prediction also supports scheduling ([ALPS](https://doi.org/10.5281/zenodo.19078431), [Entropy-guided length prediction](https://arxiv.org/abs/2602.11812)). Hosted open-weight models vary across providers in accuracy, length and price ([Same model, not the same service](https://arxiv.org/abs/2605.02821), [API gateway consistency](https://arxiv.org/abs/2604.21083), [Model equality testing](https://arxiv.org/abs/2410.20247)); FACET routes among providers of a fixed model per (provider, task) using measured outcomes ([Market-aware provider routing (FACET)](https://arxiv.org/abs/2609.37902)). We instead route per query over endpoints of several models with predicted, endpoint-specific costs.

![shared prefill overview](figures/shared_prefill_overview.png)

**Cost–accuracy frontiers at billed prices.** Prefill cost readouts (blue) versus training-median length (orange), with the same success predictions and the same encoder pass. Dots: the deterministic policy at each $V$; lines: their frontiers; hollow circles: policies selected on calibration for fixed accuracy targets. Arrows run from the median rule's frontier to ours at matched accuracy and give the cost saved; averaged over the shared accuracy band the saving is 34% on LiveCodeBench, 29% on MMLU-Pro and 35% on Omni-MATH.

## Cost Prediction and Routing

For query $x$ and route $m$ (a model at a fixed reasoning effort, served by an endpoint), let $p_m(x)$ be the probability of a correct answer and $c_m(x)$ the expected generation cost. One route is chosen before generation,

$$
m^*(x;V)=\arg\max_m\{V\hat p_m(x)-\hat c_m(x)\},
$$

where $V$ is the value of a correct answer; sweeping $V$ trades accuracy for cost. Every query is answered once, with no verifier or fallback.

### Readouts.

We concatenate mean and last-token activations from eight layers of a frozen Qwen3-4B (Instruct-2507; Thinking-2507 on Omni), standardized on training data. Success readouts are L2-regularized logistic regressions with held-out Platt calibration; cost readouts are ridge regressions of log mean output length, regularized by training-only cross-validation. Predicted length is exponentiated with residual smearing and matched to the route's training mean, then priced: $\hat c_m(x)=r_m^{\rm in}n^{\rm in}_m(x)+r_m^{\rm out}\hat\ell_m(x)$.

## Experimental Protocol

The routes are gpt-oss-20b at low and medium effort, deepseek-v4-flash, and gpt-oss-120b at medium and high effort, called through OpenRouter, with deepseek-v4-flash pinned to one provider (StreamLake) throughout, because its output length depends on the serving provider (see Endpoints). LiveCodeBench ([LiveCodeBench](https://arxiv.org/abs/2403.07974)) has 892 problems split by date into 441 training, 110 calibration and 341 test problems. Omni-MATH ([Omni-MATH](https://arxiv.org/abs/2410.07985)) and MMLU-Pro ([MMLU-Pro](https://arxiv.org/abs/2406.01574)) have 275/75 and 550/150 training/calibration problems and 1,000 and 6,500 disjoint test problems. Training problems have 2–16 draws per route; Omni-MATH and MMLU-Pro test problems have one.

### Prices.

Realized cost is the cost billed for each call (on LCB, realized tokens at the billed rates). Predictions are priced at each model's effective billed rate, fitted on the test calls (USD per million output tokens: 0.13 for gpt-oss-20b, 0.08 for pinned deepseek-v4-flash, 0.26 for gpt-oss-120b). These differ from the list prices by up to 2.3$\times$.

### Evaluation.

All estimators are fitted on training problems only and evaluated on the test sets, with the same success predictions. Each $V$ on a dense grid gives a deterministic policy and an observed (cost, accuracy) point. Comparing two cost estimators $a,b$, we interpolate each one's cost at 12 accuracies spanning the interior 90% of the range both reach and report $1-\exp(\overline{\log C_a/C_b})$, the cost saved by $a$ at matched accuracy. Intervals are percentile intervals from paired problem bootstraps (300 resamples), keeping each problem's routes together. To check that an operating point can be chosen in advance, we also select the cheapest $V$ reaching each accuracy target on calibration and apply it once.

| Cost saved by ours vs. | LCB ($n$=341) | Omni-MATH ($n$=1,000) | MMLU-Pro ($n$=6,500) |
| --- | ---: | ---: | ---: |
| Training-median length | 33.7 [24.1, 39.9] | 34.8 [27.9, 39.8] | 28.8 [26.0, 31.8] |
| Training-mean length | 29.6 [23.6, 34.5] | 35.8 [28.9, 40.5] | 28.6 [25.9, 31.7] |
| Prompt-feature GBM | 26.3 [16.3, 32.9] | 26.5 [18.9, 31.9] | 25.2 [21.0, 28.7] |
| MixLLM-style embeddings | 18.2 [11.2, 25.2] | 26.7 [20.0, 32.5] | 19.2 [15.3, 22.5] |
| ZeroRouter (reimplemented) | 21.1 [13.2, 26.8] | 14.4 [8.4, 20.6] | 29.0 [25.5, 32.0] |

**Table 1.** **Cost saved at matched accuracy, billed costs.** Cost saved (%) by our prefill cost readouts relative to each estimator, with 95% paired bootstrap intervals; same success predictions for every arm except ZeroRouter, which uses its own; all intervals exclude zero. Test sets as in Figure 1. MixLLM-style: jina-embeddings 137M with the MixLLM ensemble head (LCB, code-aware variant). ZeroRouter: item-response latent fitted on the five routes, 4B features, its own success model and bin pricing, configuration chosen on calibration.

## Results

### Cost readouts beat every estimator.

At matched accuracy, prefill cost readouts spend 33.7% less than median-length pricing on LCB, 34.8% less on Omni-MATH and 28.8% less on MMLU-Pro (Table 1). Every external estimator loses with intervals excluding zero: mean length by 29–36 points, a prompt-feature model by 25–27, MixLLM-style embeddings by 18–27 and a full ZeroRouter reimplementation by 14–29. Choosing $V$ in advance costs little: on Omni-MATH and MMLU-Pro, policies selected on calibration spend 0–4% more than our own test frontier at the accuracy they reach (hollow circles in Figure 1).

### Endpoints.

The same model served by different providers behaves like the model plus an offset: pinning deepseek-v4-flash to each of three providers, they agree on correctness for 90–96% of problems and their output lengths correlate at .83–.95, but one writes about half as many tokens as another. Reusing the readouts with one multiplicative length offset per endpoint, estimated from 50 calls, removes the resulting cost bias on average, though a single estimate is noisy. A few dozen calls per provider also identify the cheapest one, and pinning it beats routing over providers; we therefore pin deepseek-v4-flash throughout, which also makes its length more predictable and raises our saving on every large set.

### Other pools.

Savings against median pricing are 14.9% [9.4, 21.4] on AIME 1983–2024 (281 test problems) and 16.8% [5.6, 28.3] on APPS (382).

## Ablations

### Is difficulty enough to price?

Two ablations price each route from our own success readouts instead of a dedicated cost readout: ZeroRouter's bin lookup on mean success logit (ten quantile bins), and a ridge regression of log length on the success logits. Both come within a few points of the cost readout on LCB (ours saves 4.0% [−1.1, 9.0] and −0.8% [−5.6, 4.0]) and on Omni-MATH (6.4% [1.1, 11.6] and −0.1% [−4.6, 4.6]), where output length tracks difficulty, and lose by 22.3% [18.8, 24.7] and 24.1% [21.1, 26.7] on MMLU-Pro. There, even empirical difficulty explains little of log length ($R^2\approx .19$), while subject labels recover roughly 60% of the gap. The prefill encodes length information beyond difficulty, and that information pays where difficulty and length come apart.

### Representation, readout and prefill size.

Table 3 crosses three text encoders with three cost readouts: ridge, CARROT-style $k$-nearest neighbours and a MixLLM-style ensemble, each with our success predictions and with every encoder pass priced. Our readout saves 16–27% against every cell, including Qwen3-Embedding-8B, an encoder twice the prefill's size. On PCA-reduced 4B features the three readouts are within about 7 points of one another, so the gain comes from the representation rather than the regressor. Pricing the encoder pass changes no comparison by more than 3 points. Table 4 varies the prefill model while keeping prompts, layers and both readouts fixed. Every size saves 22–34% against median pricing, and length $R^2$ rises with size. None beats our prefill: Qwen3-1.7B and 8B come within 0.3–4 points (within the intervals on Omni-MATH), while 0.6B trails by 9–11.
The result is not specific to Qwen: prefills from three other families (Phi-4-mini, Granite-3.3-2B, SmolLM2-1.7B) also save 25–31% against median pricing on both sets. Phi-4-mini matches our router on MMLU-Pro (−0.7 [−2.6, 1.3]) but trails by 10 points on Omni-MATH; Granite and SmolLM2 trail by 5–9.

### New model families.

Readouts for Qwen3-32B and GLM-4.7-flash, fitted on the same frozen 4B features, were evaluated on 1,905 MMLU-Pro test problems (log-length $R^2$ .48 and .06). With all seven routes, prefill cost readouts save 30.0% [25.4, 35.3] relative to median pricing, 21.8% [16.3, 26.7] relative to cost from success and 20.6% [15.3, 25.4] relative to difficulty bins. On a pool of the two gpt-oss-20b routes and the two new families, they save 32.2%, 11.6% and 14.8%. Readouts onboarded from ten examples per new route match the full readouts in the seven-route pool (0.1 points) and trail them by 11.9 [−4.2, 21.9] in the new-family pool.

|  | Omni-MATH |  |  | MMLU-Pro |  |  |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder | Ridge | $k$NN | MixLLM | Ridge | $k$NN | MixLLM |
| Qwen3-Embedding-8B | 16.6 | 20.9 | 21.5 | 21.1 | 21.3 | 19.5 |
| jina-embeddings 137M | 23.8 | 25.3 | 26.6 | 17.8 | 20.4 | 18.9 |
| MiniLM-L12 | 24.3 | 25.7 | 25.4 | 18.8 | 19.0 | 16.2 |

**Table 2.** **Representation and readout.** Cost saved (%) by our prefill readouts relative to each encoder–readout cell at matched accuracy on the Omni-MATH and MMLU-Pro test sets; billed costs, every encoder pass priced, same success predictions. Every paired bootstrap interval (200 resamples) excludes zero.

|  | Omni-MATH |  |  | MMLU-Pro |  |  |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | $R^2$ | vs. median | vs. ours | $R^2$ | vs. median | vs. ours |
| Qwen3-0.6B | .62 | 31.9 | $-9.0$ | .33 | 22.0 | $-10.8$ |
| Qwen3-1.7B | .65 | 34.4 | $-2.8$ | .42 | 26.8 | $-3.0$ |
| Qwen3-4B (hybrid) | .67 | 29.3 | $-4.1$ | .45 | 25.8 | $-4.4$ |
| Qwen3-8B | .68 | 31.4 | $-0.3$ | .47 | 26.1 | $-4.1$ |
| Phi-4-mini (3.8B) | .65 | 31.4 | $-10.2$ | .45 | 28.4 | $-0.7$ |
| Granite-3.3 (2.5B) | .62 | 26.6 | $-9.4$ | .39 | 25.8 | $-4.7$ |
| SmolLM2 (1.7B) | .59 | 31.4 | $-7.6$ | .33 | 24.8 | $-4.9$ |
| Qwen3-4B-2507 (ours) | .70 | 34.8 | – | .50 | 28.8 | – |

**Table 3.** **Prefill size.** Each prefill feeds both readouts, refitted with the paper's recipes on the training problems; Omni-MATH and MMLU-Pro test sets, billed costs. $R^2$: test log-length $R^2$ averaged over routes. vs. median: cost saved relative to median pricing with the same row's success predictions. vs. ours: cost saved by the row's router relative to ours (negative: it spends more); its paired intervals (300 resamples) include zero for Qwen3-1.7B and 8B on Omni-MATH and Phi-4-mini on MMLU-Pro.

## Discussion and Limitations

Each route needs output-length labels. The main experiments use five routes from two model families; two further families (Qwen3-32B, GLM-4.7-flash) were tested on MMLU-Pro only. Omni-MATH and MMLU-Pro test problems have one draw per route.
Intervals are pointwise and assume independent problems, and Table 1 conditions on the fitted readouts; resampling the training problems as well and refitting every estimator widens them, but our saving over median pricing stays positive on both large test sets (Omni-MATH [23.2, 38.0], MMLU-Pro [22.9, 30.7]). Only deepseek-v4-flash is pinned to a provider; the gpt-oss routes are not, since their provider changes price but barely length. The prefill comparison covers four model families at up to 8B.
Billed rates were measured over one collection, and prices change within days. The ZeroRouter, MixLLM and CARROT baselines are reimplementations.

## Conclusion

A frozen prefill that already predicts success also predicts cost. On held-out test problems at billed prices, its cost readouts reduce spending at matched accuracy and outperform every alternative estimator we tested; pricing from difficulty alone matches them only where output length tracks difficulty. Endpoints of one model share these readouts up to an offset, so the readouts can absorb provider drift from a few dozen calls, though pinning a provider is simpler.

## References

- [Prefill router](https://arxiv.org/abs/2603.20895)
- [MixLLM](https://arxiv.org/abs/2502.18482)
- [CARROT](https://arxiv.org/abs/2502.03261)
- [ZeroRouter](https://arxiv.org/abs/2601.06220)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [Omni-MATH](https://arxiv.org/abs/2410.07985)
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [RouterBench](https://arxiv.org/abs/2403.12031)
- [AlphaCode / CodeContests](https://arxiv.org/abs/2203.07814)
- [TACO](https://arxiv.org/abs/2312.14852)
- [BigCodeBench](https://arxiv.org/abs/2406.15877)
- [Entropy-guided length prediction](https://arxiv.org/abs/2602.11812)
- [ALPS](https://doi.org/10.5281/zenodo.19078431)
- [Same model, not the same service](https://arxiv.org/abs/2605.02821)
- [API gateway consistency](https://arxiv.org/abs/2604.21083)
- [Model equality testing](https://arxiv.org/abs/2410.20247)
- [Market-aware provider routing (FACET)](https://arxiv.org/abs/2609.37902)
