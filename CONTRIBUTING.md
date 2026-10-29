# Contributing

Read the bilingual [experiment history](EXPERIMENTS.md) for prior attempts,
aggregate results and the limits of the historical evaluations.

Start with the feature baseline and the dataset schema. Pull requests should
include the exact target, split, random seed, dependency versions, aggregate
metrics and a reproducible command. Explain how repeated votes, annotator
scales, weak labels, cluster judgments and revisions were handled.

Good initial projects:

- strictly increasing rater calibrations versus shifts/affine maps;
- ordinal consensus versus averaging on identical held-out pairs;
- grouped label-denoising evaluations and uncertainty from disagreeing raters;
- lightweight regressors on the public 192-dimensional vectors;
- quantization/distillation with accuracy and CPU latency measurements;
- HuBERT fine-tuning experiments using separately permitted recordings.

Compare validation methods on identical cohorts and in meaningful target
units. A compressed score scale can artificially lower error. Do not report
propagated cluster values as independent human listening judgments.

Use issues for proposals and pull requests for code/results. Do not attach
private audio, tokens, personal IDs, or attempts to identify speakers/raters.
Report suspected privacy leaks privately through the maintainer's GitHub
profile contact channels; avoid posting the leaked values publicly.

Contributions to code are under the repository's Apache-2.0 license.
