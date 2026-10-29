# Voice passability model

Research tools for a HuBERT model trained on human ratings of perceived voice
femininity / passability. Contributions to label quality, evaluation, compact
models and reproducible training are welcome.

[Русское описание](README_RU.md) · [Model card](MODEL_CARD.md) ·
[Dataset card and schema](DATASET_CARD.md) · [Contributing](CONTRIBUTING.md)

The first release contains the **original model used in the bot**, its UINT8
ONNX version, and an ECAPA feature dataset with ratings. Audio is not distributed.
Model files and data are [release assets](https://github.com/liya243/voice-passability-model/releases/tag/v0.1.0),
so cloning the repository stays small.

## Run on your own audio

Python 3.10+:

```bash
git clone https://github.com/liya243/voice-passability-model.git
cd voice-passability-model
python -m venv .venv
# Activate the environment, then:
pip install -e ".[onnx]"
python scripts/download_assets.py --backend onnx
voice-passability your_recording.wav --backend onnx
```

For the full model:

```bash
pip install -e ".[torch]"
python scripts/download_assets.py --backend torch
voice-passability your_recording.wav --backend torch
```

Inference runs locally. Input is mono/resampled to 16 kHz and peak-normalized.
Five-second non-overlapping windows are scored. A final real tail shorter
than 3 seconds is omitted; tails of 3–5 seconds are padded. A recording
shorter than 3 seconds is rejected. The result is the mean window score.

The output is a **learned human-perception rating**, not a calibrated
probability of gender, identity, or how every listener will perceive someone.
Languages, microphones and populations can change the result.

## Work with the public feature dataset

```bash
pip install -e ".[research]"
python scripts/download_assets.py --dataset
voice-passability-baseline --dataset data
```

The dataset contains **9,004 recording-level ECAPA vectors (192 values)**,
8,582 individual ratings, 646 genuine cluster ratings, and separately marked
historical/derived labels. Forty-three raters have random six-digit IDs
consistent across tables. Records and clusters have random identifiers too.
Original names, Telegram IDs, timestamps and source audio are excluded.

**Pseudonymized does not mean anonymous.** ECAPA is a speaker-recognition
encoder; embeddings can be used to link voices. The private identifier mapping
is not published. See the [privacy details](DATASET_CARD.md#privacy).

The public features support regression baselines, calibration of rater scales,
ranking and label analysis. End-to-end HuBERT fine-tuning requires waveforms;
the ECAPA vectors cannot substitute for them. A small local-audio fine-tuning
example is provided in `scripts/finetune.py` for recordings you can use.

The initial data split groups recordings by acoustic similarity. It is not
verified speaker-disjoint. No independent accuracy improvement over the
original model is claimed by this release.

## Reproducibility and contributions

`assets.json` contains sizes and SHA256 hashes. The full model is exported as
Safetensors with all weight tensors unchanged; optimizer state and saved
training arguments are omitted. The ONNX artifact has exporter docstrings
removed. Check results are in `VERIFICATION.md`.

```bash
python -m unittest discover -s tests -v
```

Useful contributions include rater-scale alignment, robust grouped evaluation,
smaller regressors, distillation, and carefully documented training experiments.
Use issues and pull requests to share results. Do not attach private audio or
personal identifiers to issues.

Code/model: [Apache-2.0](LICENSE). Feature dataset/ratings:
[CC BY 4.0](LICENSE_DATA.md). Source recordings are not licensed or supplied
by this release. See [NOTICE](NOTICE) for pretrained-model attribution.
