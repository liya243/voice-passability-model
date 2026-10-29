# Model card: original HuBERT voice-passability regressor

## Architecture and origin

- Backbone: [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960).
- Hidden size 768; masked mean pooling; regression head 768 → 256 → 1
  with GELU, dropout and sigmoid. A parallel 20-bin classification head is
  retained for exact checkpoint compatibility; inference uses the regressor.
- Original fine-tuned checkpoint at step 30,000. This is the original model,
  not one of the subsequent experimental models or weight averages.
- The historical training supervision used administrator-assigned cluster
  scores. These are weak labels shared by recordings within each old group.
  They are not thousands of independent human listening judgments.
- Later individual/cluster ratings are released as research data; their
  existence does not mean the released original model was trained on them.

Safetensors contains only the model tensors, bitwise equal to the original
checkpoint. The JSON config permits offline reconstruction without downloading
another copy of the backbone. UINT8 ONNX is an additional inference artifact;
quantization changes its numerical predictions. Hashes/sizes are in `assets.json`.

## Input and output

Input: mono 16 kHz audio, peak normalization over the whole recording.
Inference windows are 5 s, minimum real tail 3 s; eligible shorter windows
are zero-padded, shorter tails omitted. Mean-absolute-amplitude threshold
1e-4 skips near-silent windows; if every eligible window is skipped, the
first eligible window is used as a legacy-compatible fallback. Therefore
silence can still receive a score; applications should apply their own
speech-presence checks when appropriate.

The output is 100 × sigmoid(regression logit), averaged over windows.
The ONNX graph already emits the 0–100 value; do not multiply it again.
Historically lower values represent more masculine perception, higher values
more feminine perception. It is a subjective perception target, not a
ground-truth label of a speaker's gender identity or a calibrated pass probability.

## Evaluation and limitations

This initial release establishes inference reproducibility, checkpoint parity
and a public research dataset. It does not establish a new independent
speaker-disjoint accuracy estimate. Original and later experiments used
different labels, splits and scales; their MAEs cannot be compared without
matching the evaluated cohort and target.

The bilingual [experiment history](EXPERIMENTS.md) documents those earlier
training, calibration and compression attempts, including unsuccessful variants
and aggregate results. It does not describe a new independent model benchmark.

Raters disagree in scale and in ordering. Recording quality, speech style,
language and population shifts can affect predictions. Closely related
recordings and cluster-level labels create dependence between examples.
The released feature baseline targets mean genuine individual ratings and
uses grouped splits; it is a separate experiment from waveform-model quality.

## Intended work

Voluntary personal voice-training feedback and research into human perception,
label agreement, grouped validation, regression, compression and distillation.
A score is not a basis for deciding someone's identity or access to services.
The model has not been trained with differential privacy and no formal
training-data privacy guarantee is claimed.

## Licensing

Released code and fine-tuned weights: Apache-2.0, with attribution to HuBERT/Meta.
The upstream backbone lists Apache-2.0 on its model card. Feature data has a
separate CC BY 4.0 notice. No source audio redistribution rights are conveyed.
