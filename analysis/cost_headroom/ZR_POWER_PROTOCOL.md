# Free ZeroRouter analyses: protocol recorded before execution

2026-09-30. Exploratory follow-up to already observed fixed-split results;
this is not a prospective preregistration. No new model generations.

1. Reconstruct the published calibration-selected ZeroRouter configuration
   on each original train split and retain the shipped linear predictions.
   Use the existing 250-point utility sweep and 12 interior accuracy targets.
   Primary effect: difference in savings against median-output routing, in
   percentage points. Preserve the existing separate arm/reference bands;
   also report a sensitivity analysis using one band shared by all three arms.
   Report plug-in effects, rather than calling a bootstrap mean the estimate.
2. Fixed-fit uncertainty diagnostic: 1,000 paired problem bootstrap samples,
   1,000 paired within-problem generation bootstrap samples, and a nested
   200 problem samples x 5 generation samples. Resample each route's valid
   draws jointly for correctness and token cost; choices and fitted heads
   stay fixed. Nested variance components describe this resampling scheme,
   not an identified decomposition of true population uncertainty. In
   particular, the problem bootstrap already contains noisy observed means.
3. Five-fold cross-fitting, shuffled KFold seed 17, on all available problems.
   Within each outer training partition, randomly reserve calibration rows
   using the original calibration/(train+calibration) fraction and seed
   1700+fold. Refit both success and cost heads. Preserve the original linear
   C grid, all-draw binomial fit, first-draw Platt calibration, and ridge
   LOOCV cost selection. Select ZeroRouter D in {1,5}, seed in {0,1,2},
   K in {5,10,20} on inner calibration only. PCA is fit on training only.
   Pool held-out predictions, retaining each fold's train median reference.
   Bootstrap pooled problems 1,000 times, conditional on these fitted heads.
   These intervals omit training variability and are not valid universal
   confidence intervals for the learning algorithm. The larger training
   fraction and random rather than original split also change the estimand.
4. Exploratory equal-weight Stouffer combination of three independent-pool
   one-sided centered-bootstrap p estimates from the ORIGINAL fixed split.
   Center bootstrap effects at the plug-in effect; use add-one smoothing.
   This tests pooled positive evidence, not a win on every dataset. No
   multiplicity-adjusted confirmatory claim will be made.
5. Omni equivalence: +/-5 percentage points in savings difference. Use
   the 90% paired percentile interval as an approximate bootstrap TOST
   criterion (must lie strictly inside both bounds). Also report normal
   approximation TOST p values. Failure is inconclusive, not equivalence.

Bootstrap seed 20261001; no stopping on observed significance. Save all
bootstrap draws, fold assignments, selections, and settings. An exact
orthogonal reduction to the training feature row space may accelerate the
linear models; it does not truncate informative training dimensions.
