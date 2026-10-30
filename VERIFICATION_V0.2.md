# Release verification — v0.2.0

## Weights and inference

- Nine selected checkpoints exported to tensor-only Safetensors. Every exported
  tensor is finite where floating-point and bitwise equal to the corresponding
  source model tensor; strict offline state loading passes for all nine.
- The supervised ECAPA export retains the frontend, normalization, embedding
  encoder and regression head. Unused speaker classifier/global embedding
  normalization are omitted; the source regression path does not use them.
- Public inference code and source architectures were compared on eight private
  real TRAIN windows for each checkpoint: **maximum score delta 0 points**.
  This verifies conversion, not test-set accuracy.
- Expert UINT8 metadata-cleaned ONNX and source graph were compared on 12 real
  TRAIN windows: **maximum score delta 0 points**. Node names/docstrings/metadata
  do not affect the graph's computations. Quantization itself changes scores.
- Both mini TorchScript runtimes include peak normalization, Fbank, sentence
  mean normalization, ECAPA and the percentage conversion. Each was checked
  against its source student on all **1,405 unique expert windows** with maximum
  score delta **0 points** in the local export environment.
- Dynamic batch sizes 1, 2, 4, 8 and 48 pass. A tensor-only length mask replaces
  SpeechBrain's trace-time Python batch-length construction with an equivalent
  mask. This does not change the model weights or scoring formula.
- TorchScript debug metadata is removed before publication. Before/after
  outputs are bitwise equal for every checked batch size. Public runtime hashes
  therefore differ from the private deployed runtime's hash; runtime metadata
  records both. No example waveforms are stored in the frozen archives.
- Runtime export used PyTorch 2.11 CPU. The expert runtime was additionally
  checked on production PyTorch 2.9 CPU on 48 real TRAIN windows; maximum
  cross-version deviation was **0.000029 points**. Public sanitized artifacts
  both pass a separate load check on production PyTorch 2.9, using 48 real
  TRAIN windows; expert sanitized versus deployed outputs are bitwise equal.
- Original UINT8 from v0.1.0 and the graph used in the six-model comparison
  were compared on 12 real TRAIN windows: maximum output delta **0 points**.
- Existing audio-window tests pass, including omission of tails below three
  seconds before padding and the legacy silence fallback.
- Installed CLI smoke tests pass for both minis with TorchScript and Safetensors
  backends, including a padded three-second tail. Released TorchScript archives
  target CPU; GPU inference/fine-tuning uses the ECAPA Safetensors backend.

## Dataset comparison and figures

- Fixed comparative benchmark: 9,004 recordings, 32,312 unique inputs, 193,872
  predictions for six models; 18,539 real judgments from 43 raters, including
  1,406 expert judgments. Weak propagated labels are not extra human votes.
- Thirty-six sampled listener-input reconstructions are bitwise identical.
- Expert-only figure statistics were asserted against the saved benchmark to
  tolerance 1e-9. PNG/SVG histograms, CDFs, aggregate JSON/CSV and local HTML
  reports are included; raw per-record predictions are excluded.
- Ranking excludes tied expert targets and identical input pairs. Model ties
  are explicit; half-tie counting applies only to the stated order-error metric.
- TEST was used in earlier rounds. Group bootstrap does not correct repeated
  hypothesis selection or establish speaker-disjointness. Full-data statistics
  include TRAIN. The old original model's historical audio overlap is unknown.

## Publication boundaries

Safetensors has no optimizer, training arguments or private file paths. ONNX
docstrings/metadata and TorchScript debug data are removed. Only allowlisted
weights/configs, aggregate reports, figures, attribution and checksums are
uploaded. No source audio, database, private rater mapping, per-record prediction
table, SSH material or original training checkpoint container is released.
The original public feature dataset remains the separate v0.1.0 snapshot.

`assets-v0.2.0.json` verifies downloaded files by SHA256. `MODEL_CATALOG.json`,
`CONVERSION_VERIFICATION.json` and `SHA256SUMS` accompany the release assets.
The full private training queues are not a fully reproduced public training
pipeline; source checkpoints, selections and aggregate outcomes are documented
in the bilingual [experiment history](EXPERIMENTS_2026-10-30.md).

## Русский

Девять экспортов проверены побитово и строгой загрузкой весов. Контрольные
прогнозы открытого кода совпали с исходными архитектурами; удаление служебной
информации ONNX/TorchScript не изменило оценки. Обе мини проверены на 1 405
уникальных экспертных окнах; размеры batch 1/2/4/8/48 работают. Эти проверки
доказывают корректность экспорта, а не независимое качество модели.

Графики основаны на реальных человеческих прослушиваниях. Перенос на запись
или кластер не выдаётся за новые оценки. TEST уже использовался; все данные
включают TRAIN. Аудио, персональные ID, имена записей и таблицы индивидуальных
прогнозов не публикуются. Поздние экспертные метки не добавлены в старый
открытый датасет. Приватные обучающие очереди воспроизводятся не полностью;
публичны веса, запуск, агрегированные результаты и история.
