# Release verification — v0.1.0

## Model artifacts

- All 219 model tensors are bitwise identical to the original checkpoint.
- Only tensors were exported to Safetensors; optimizer/scheduler/training arguments are absent.
- Offline reconstruction strictly loads all weights.
- Original inference architecture and released full model were compared on two private recordings / six windows: maximum score difference **0 percentage points**.
- Original UINT8 ONNX and metadata-cleaned UINT8 ONNX: maximum output difference **0 percentage points** on the same controls.
- Full-versus-UINT8 mean absolute difference on these controls: **0.363 points**, maximum **0.696 points**. This small check is not a dataset-level accuracy estimate.
- The test audio is not distributed.

## Data

- 9,004 finite unit-normalized 192D vectors, aligned with random record IDs.
- 43 unique random six-digit rater IDs, consistent across tables; original ID mapping remains private.
- Aggregated record means recomputed independently from genuine per-rater judgments.
- Cluster membership and acoustic split blocks remain intact across dataset splits.
- Heard examples match the cluster and its membership revision.
- Public text, tables, Safetensors metadata and ONNX string metadata were checked for original identifiers, original source filenames and server paths.
- No source audio, database dumps, original checkpoint files or private mapping is included in Git or release assets.
- These checks verify the export; they do not constitute a formal anonymity guarantee.

## Reproducible fixed-feature baseline

Ridge regression, alpha=1 selected on validation among 0.1/1/10/100;
target = mean genuine individual rating per recording, equal rater weight.

| Split | Rated recordings | MAE, points |
|---|---:|---:|
| Train | 3645 | 13.066 |
| Validation | 782 | 13.140 |
| Test | 810 | 14.062 |

These are ECAPA baseline results, not HuBERT accuracy. The grouped split is not verified speaker-disjoint.
Run `voice-passability-baseline --dataset data`; details are in `baseline_result.json`.

## Environment

Tested locally with Python3.13.5, Torch2.11.0+cpu, Transformers4.57.1,
Safetensors0.6.2, ONNX Runtime1.31.1, NumPy2.1.3, SciPy1.15.3,
SoundFile0.13.1, scikit-learn1.6.1. CPU inference uses two threads by default.
The audio-boundary unit tests also run in GitHub Actions.
