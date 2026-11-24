# voice-passability-model

A fixed training pipeline for a model that scores how a voice is heard on a feminine–masculine scale (0–100),
used by the Qt_voice Telegram bot. Data keeps growing; the pipeline stays the same, and every backbone goes
through the same stages and the same scorecard.

No audio, ratings or speaker identities are in this repository. Paths point to a data directory (`VPM_DATA`).

## Stages

| Stage | Module | Output in `work/` |
|---|---|---|
| Snapshot of all recordings and rating tables | `vpm.snapshot` | `snapshot/recordings.csv`, `snapshot/ratings/` |
| Voice check per 0.5 s (silero VAD) | `vpm.voicecheck` | `voicecheck/<source>.npz` |
| Cis labels from two gender classifiers, validated first | `vpm.cislabel` | `cislabel/labels.csv` |
| Target: expert-anchored consensus, per-rater monotone scales, pair votes | `vpm.target` | `target/items.csv`, `raters.csv`, `pairs.csv` |
| Split by voice (gold voices always test) | `vpm.split` | — |
| Scorecard windows (fixed per snapshot) | `vpm.scorecard build` | `scorecard/windows.csv` |
| Frozen-feature probe of a backbone | `vpm.probe <backbone>` | `preds/probe_<backbone>.csv` |
| Fine-tuning | `vpm.train` | `runs/<name>/`, `preds/<name>.csv` |
| Any model on the scorecard windows | `vpm.predict` | `preds/<name>.csv` |
| Report | `vpm.scorecard report preds/*.csv` | `scorecard/<name>.json` + table |

```
export VPM_DATA=~/data
python -m vpm.snapshot
python -m vpm.voicecheck
python -m vpm.target
python -m vpm.scorecard build
python -m vpm.predict --model legacy:model_archive_20261121/runs/ft3/best.pt --calib sys_new/pass_onnx/ft3_calibration_half_v1.json --name ft3_bot
python -m vpm.probe mhubert147
python -m vpm.scorecard report work/preds/*.csv
```

## Scorecard

- **gold**: MAE to the consensus on the 500 gold segments, overall, in 40–90 and per 10-point band (the worst band is reported, so one part of the range cannot quietly fail); Spearman; order accuracy on pairs ≥ 10 points apart.
- **heldout**: the same on listened items of test voices outside gold.
- **consistency**: spread of one recording's voiced windows (people: about 4.7 points).
- **language**: TidyVoice speakers recorded in several languages — how far the language moves a speaker's mean, next to the same-language noise floor.
- **cis**: median score of cis women and cis men; share of cis women below 80.

## Loss

Scale (Huber to the consensus) + order (RankNet on consensus pairs and miniapp pair votes) + same voice within a recording (hinge at 5 points) + same speaker across languages (hinge, optional adversarial language head) + cis floors/ceilings.

## Data sources

- Own recordings and ratings (private).
- [TidyVoice](https://datacollective.mozillafoundation.org/) (TidyVoiceX_ASV, derived from Common Voice, CC0): one speaker, several languages.
- Cis voices: own Telegram voice messages, labelled by the classifiers, plus official corpora with gender metadata (`cis/<corpus>/speakers.csv`).
- Classifiers: [JaesungHuh/voice-gender-classifier](https://huggingface.co/JaesungHuh/voice-gender-classifier) (MIT), [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) (CC-BY-NC-SA 4.0).
