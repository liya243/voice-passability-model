# Model card: HuBERT and ECAPA voice-passability regressors

## v0.2.0 additions

The original model described below remains available in v0.1.0. v0.2.0 adds
nine selected research checkpoints. See [model_zoo.json](model_zoo.json) for
exact source hashes, parameters and backend assets, and the bilingual
[experiment continuation](EXPERIMENTS_2026-10-30.md) for selection and results.

| Variant | Architecture | Selection role |
|---|---|---|
| expert-hubert-v1 | Full HuBERT, 94.77M | Expert refine_02 teacher; also UINT8 |
| expert-hubert-recording | Full HuBERT, 94.77M | Common-recording validation |
| expert-ecapa-supervised | ECAPA1024, 20.79M | Supervised 2s training / 5s inference |
| expert-ecapa-distilled-human | ECAPA1024, 20.79M | Human validation |
| expert-ecapa-distilled-fidelity | ECAPA1024, 20.79M | Teacher fidelity |
| expert-mini | ECAPA256, 1.78M | Compact expert validation; bot/live trial |
| original-mini | ECAPA256, 1.78M | Original-teacher VAL fidelity; six-model benchmark |
| original-mini-human | ECAPA256, 1.78M | Separate human-selected epoch |
| original-ecapa384 | ECAPA384, 3.44M | Original-teacher fidelity, wider student |

ECAPA uses an 80-band Fbank, per-sentence mean normalization without standard
deviation normalization, a 192-dimensional embedding and LayerNorm → 128 → 1
GELU/dropout/sigmoid head. Widths and exact config accompany each model. The
1024-channel experts derive from [SpeechBrain's VoxCeleb ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb);
256/384-channel students are initialized from scratch. No speaker-classification
head or training-only feature-alignment projector is needed at inference.

Expert supervision: 1,406 real judgments from one vocal expert, with split
989/219/198. Weak recording/cluster propagation remains separate. Original
student training excludes 418 challenge-set windows from teacher targets;
those original-model scores are not reliable ground truth. Later expert data
is not included in the v0.1.0 public feature dataset snapshot.

New variants use **per-window peak normalization**, unlike the legacy original
CLI's whole-recording normalization. The TorchScript mini includes this step
and emits 0–100 directly; do not multiply again. It expects float32
`[batch, 80000]` mono 16 kHz inputs and is checked on batch sizes 1/2/4/8/48.
TorchScript targets CPU; use ECAPA Safetensors for GPU inference or training.
All new variants are released for five-second inference, including the
supervised ECAPA trained on two-second windows. The original tail rule still
applies: omit real tails below three seconds; pad tails of three to five.

The repeated fixed benchmark suggests useful speed/size improvements, but
does not prove independent generalization. Expert-mini versus full expert
TEST MAE difference has a group-bootstrap interval crossing zero. Expert
models remain worse than the original in the >80 human-rating range. All-human
and expert-only distributions differ because raters disagree and have different
recording coverage. Consult the [aggregate benchmark](experiments/benchmark-2026-10-30.json)
and [verification](VERIFICATION_V0.2.md) before comparing these selections.

The expert mini was enabled for a practical bot and live-miniapp trial on
2026-10-30; previous weights remain available for rollback.

## Original v0.1.0 model

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
