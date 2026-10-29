# Voice passability model / Модель оценки пасса голоса

[English](#english) · [Русский](#russian)

<a id="english"></a>

## English

Research tools for a HuBERT model trained on human ratings of perceived voice
femininity / passability. Contributions to label quality, evaluation, compact
models and reproducible training are welcome.

[Model card](MODEL_CARD.md) ·
[Dataset card and schema](DATASET_CARD.md) · [Contributing](CONTRIBUTING.md)

The first release contains the **original model used in the bot**, its UINT8
ONNX version, and an ECAPA feature dataset with ratings. Audio is not distributed.
Model files and data are [release assets](https://github.com/liya243/voice-passability-model/releases/tag/v0.1.0),
so cloning the repository stays small.

### Run on your own audio

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

### Work with the public feature dataset

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

### Experiments so far

We have already explored:

- HuBERT fine-tuning, alternative WavLM encoders, deeper heads, layer mixing,
  attention pooling and pairwise ranking losses.
- Microphone/room augmentation and transferring a rated window's label to
  other windows from the same TRAIN recording.
- Cluster/neighbor/graph label smoothing, rater shifts and range calibration,
  smooth strictly increasing rater functions, and common-order aggregation.
- Multiple seeds and weight averaging; INT8/INT4 quantization, smaller distilled
  students and progressive pruning.

Some models reduced overall error while making the highest human ratings worse.
Later scale alignment gave a better trade-off on reused validation, but practical
use led to restoring the original model. UINT8 ONNX was the useful compression
result; more aggressive compression changed predictions too much.

Cluster averaging plus smooth rater-scale alignment was promising in held-out
**label-consensus diagnostics**. Its benefit for a retrained HuBERT still needs
to be tested. See the bilingual [experiment history](EXPERIMENTS.md#english)
for recipes, numerical results, unsuccessful attempts and evaluation limits,
and [aggregate metrics](experiments/metrics.json) for machine-readable results.

### Reproducibility and contributions

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

---

<a id="russian"></a>

## Русский

Открытый исследовательский проект на основе HuBERT-модели, обученной на
человеческих оценках воспринимаемой феминности / пасса голоса. Приглашаем
участвовать в улучшении разметки, оценке качества, уменьшении моделей
и воспроизводимом обучении.

[Карточка модели](MODEL_CARD.md) · [Карточка и схема датасета](DATASET_CARD.md) ·
[Как участвовать](CONTRIBUTING.md)

Первый релиз содержит **исходную модель, используемую в боте**, её квантованную
UINT8 ONNX-версию и набор ECAPA-признаков с оценками. Аудио не распространяется.
Веса и данные находятся в [релизе v0.1.0](https://github.com/liya243/voice-passability-model/releases/tag/v0.1.0),
поэтому сам репозиторий остаётся небольшим.

### Запуск на своих аудиозаписях

Требуется Python 3.10+:

```bash
git clone https://github.com/liya243/voice-passability-model.git
cd voice-passability-model
python -m venv .venv
# Активируйте окружение, затем:
pip install -e ".[onnx]"
python scripts/download_assets.py --backend onnx
voice-passability ваша_запись.wav --backend onnx
```

Для полной модели:

```bash
pip install -e ".[torch]"
python scripts/download_assets.py --backend torch
voice-passability ваша_запись.wav --backend torch
```

Аудио обрабатывается на вашем компьютере: преобразуется в моно с частотой
16 кГц и нормализуется по пиковой амплитуде. Оценки вычисляются для
неперекрывающихся фрагментов по 5 секунд. Последний хвост короче 3 секунд
отбрасывается; хвост длиной 3–5 секунд дополняется тишиной. Запись короче
3 секунд не принимается. Итог — среднее оценённых окон.

Шкала 0–100 отражает **усвоенные моделью субъективные человеческие оценки**,
а не калиброванную вероятность гендера, идентичности или того, как голос
воспримет любой слушатель. Язык, микрофон и состав аудитории могут влиять
на результат.

### Работа с открытым датасетом признаков

```bash
pip install -e ".[research]"
python scripts/download_assets.py --dataset
voice-passability-baseline --dataset data
```

Датасет содержит **9 004 вектора ECAPA размерности 192 — по одному на запись**,
8 582 индивидуальные оценки, 646 настоящих оценок кластеров и отдельно
обозначенные исторические и производные метки, включая 159 производных
кластерных оценок. Сорока трём оценщикам присвоены случайные шестизначные
номера, общие для всех таблиц. Идентификаторы записей и кластеров тоже
случайные. Исходные имена, Telegram ID, временные метки и аудио исключены.

**Псевдонимизация не означает полную анонимность.** ECAPA предназначена для
распознавания говорящего: признаки позволяют сопоставлять голоса. Приватная
таблица соответствий идентификаторов не публикуется.
Подробнее: [раздел о приватности](DATASET_CARD.md#privacy).

Открытые признаки позволяют обучать регрессоры, выравнивать шкалы оценщиков,
искать общий порядок и анализировать разметку. Для сквозного дообучения HuBERT
нужны аудиозаписи; ECAPA-векторы их не заменяют. В `scripts/finetune.py` есть
небольшой пример дообучения на собственных допустимых к использованию записях.

Начальное разбиение данных учитывает акустическое сходство записей.
Независимость выборок по говорящим не подтверждена. Этот релиз не заявляет
независимо измеренного улучшения качества по сравнению с исходной моделью.

### Что уже пробовали

- Дообучение HuBERT, энкодеры WavLM, более глубокую голову, смешивание слоёв,
  attention pooling и обучение попарному порядку.
- Аугментации микрофона и помещения, перенос оценки отрезка на другие окна
  той же TRAIN-записи.
- Сглаживание меток по кластерам, соседям и графу; сдвиг и растяжение шкал,
  гладкие строго возрастающие функции оценщиков и поиск общего порядка.
- Несколько seed и усреднение весов; квантование INT8/INT4, компактных
  дистиллированных учеников и постепенное прореживание.

Часть моделей снижала общую ошибку, но хуже оценивала верхний диапазон.
Позднее выравнивание шкал дало более удачный баланс на повторно использованной
валидации, однако по результатам работы в боте вернули исходную модель.
Практическим результатом сжатия стала UINT8 ONNX; более сильное уменьшение
слишком меняло прогнозы.

Кластерные средние с гладким выравниванием перспективны по отложенной
**диагностике человеческой разметки**. Их пользу для дообученной HuBERT
ещё нужно проверить. В двуязычной [истории экспериментов](EXPERIMENTS.md#russian)
описаны рецепты, численные результаты, неудачные попытки и ограничения проверок.
Есть также [машиночитаемые сводные метрики](experiments/metrics.json).

### Воспроизводимость и участие

В `assets.json` указаны размеры файлов и их SHA256. Полная модель
экспортирована в Safetensors без изменения тензоров весов; состояние
оптимизатора и сохранённые аргументы обучения исключены. Из ONNX удалены
текстовые описания экспортёра. Результаты проверок — в `VERIFICATION.md`.

```bash
python -m unittest discover -s tests -v
```

Полезные направления: выравнивание шкал оценщиков, надёжная оценка качества
с учётом групп записей, компактные регрессоры, дистилляция и подробно
описанные эксперименты с обучением. Делитесь результатами через issue
и pull request. Не прикладывайте к публичным обсуждениям частные аудио
или персональные идентификаторы.

Код и веса: [Apache-2.0](LICENSE). Признаки и оценки:
[CC BY 4.0](LICENSE_DATA.md). Исходные аудиозаписи не включены и не
распространяются под этими лицензиями. Указание авторов предобученных
моделей — в [NOTICE](NOTICE).
