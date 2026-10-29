# Experiment history / История экспериментов

[English](#english) · [Русский](#russian) · [README](README.md)

<a id="english"></a>

## English

This is a retrospective summary of the October 2026 experiments, checked against
saved reports. **The public v0.1.0 weights are the original HuBERT model and its
UINT8 ONNX export.** Later candidates were tried in the bot, but the owner
preferred the original in practical use and requested its restoration.
Experimental weights are not included in this release.

### How to read the results

MAE means mean absolute error in points on the 0–100 human-rating scale; lower
is better. Score ranges refer to **human ratings**, not model predictions.
`new_voices` denotes a deliberately collected challenge set where the original
model failed or might fail. Its teacher predictions must not be treated as
ground truth.

Three distinct evaluations appear below:

- The historical test: 847 individual ratings, 45 groups, including only 40
  challenge ratings. It was initially held out, then reused across rounds.
- Later audio-model validation: 1,234 ratings, 68 groups, including 61 challenge
  ratings. Many choices were made using this same validation set.
- Label-consensus diagnostics: grouped or whole-rater cross-validation on saved
  human judgments. These measure predictability/agreement of ratings, not the
  quality of a newly trained audio model.

The original model may have seen some old recordings with weak administrator
cluster labels. Speaker identities are incomplete, so acoustic grouping does not
prove speaker-disjoint evaluation. Repeated judgments are observations, not
independent recordings. None of these results establishes performance on a new,
independent population. They also use different splits and targets from the
[public ECAPA baseline](baseline_result.json).

### What we tried

| Direction | Experiments | Observation |
|---|---|---|
| Direct fine-tuning | Head only; last four HuBERT blocks; challenge oversampling to 20% | Overall historical-test MAE improved, but high ratings became worse. |
| Losses and representations | Huber/MSE, linear and ordinal heads, score/source balancing, intermediate layers, layer mixtures, attention pooling | No recipe removed the trade-off across rating ranges. |
| Larger or alternative encoders | Unfreeze 8/12 HuBERT blocks; WavLM Base+ and Large, frozen and fine-tuned | Base+ helped some middle-range/challenge cases; Large did not win. |
| Microphone/room augmentation | Mild frequency-response changes, noise at SNR 25–40 dB, short reflections; no pitch, formant or speed changes | Some lower average errors, without preserving the original upper-range accuracy. Perceptual invariance was a hypothesis, not guaranteed. |
| More windows per recording | Transfer an assessed window's score to random TRAIN windows with weight 0.25 or 1; average two windows | Extra windows are weak labels, not new human judgments. The selected model was nicknamed “uniform.” |
| Neighborhood label smoothing | TRAIN-only K-means with 100/200 clusters, raw/cluster blends, ECAPA/HuBERT neighbors, graph smoothing, weak labels on unrated TRAIN recordings | Better overall errors than the original in some runs, but no candidate passed the upper-range selection constraints. |
| Rater-scale alignment | Shift or shift/range; administrator, shared-human, original-model and hybrid reference scales; exact and weaker cross-window links | The most useful later audio-model result was a rater shift anchored to the original model on ordinary TRAIN data, leaving challenge labels raw. |
| Hyperparameters and averaging | Reference strength .15–.50, shift bounds, regularization, exact links only, learning rate, 4/6 unfrozen blocks, five seeds and weight averaging | A combined candidate was numerically better on some metrics; the extra selection-objective gain was small. |
| More model capacity | Residual deeper head, attentive statistics pooling, mixture of the last four layers; matched continuation controls, three seeds | No convincing gain over the prior selected model. We did not add an extra HuBERT transformer block in this series. |
| Pairwise audio-model training | Regression plus RankNet-style loss; same-rater TRAIN pairs with rating gaps ≥15/25; five recipes × three seeds | No convincing improvement over the prior model; matched controls were important. |
| Human consensus | Strictly increasing smooth rater curves; common recording order; cluster means followed by calibration/ranking | Cluster averaging plus smooth scale alignment was promising for labels; global order optimization did not beat simpler aggregation. |
| Compression | FP16 storage, several INT8 formats, ONNX UINT8, smaller distilled students, pruning and mixed INT4 | UINT8 ONNX was the practical outcome. More aggressive reductions changed scores too much. |

### Early audio-model rounds: the upper-range trade-off

These are **historical-test** results, not the later validation table.

| Model / recipe | All MAE | Challenge MAE | Human scores >80 MAE |
|---|---:|---:|---:|
| Original HuBERT | 19.665 | 20.852 | 9.915 |
| Round 1: last four blocks, challenge oversampling | 15.269 | 18.637 | 12.63 |
| Round 2: selected WavLM Base+ | 15.594 | 17.110 | 20.96 |
| Round 3: channel augmentation + full-weight label transfer (“uniform”) | 14.755 | 19.641 | 14.140 |

Round 2 completed 18 main variants and four repeat runs; round 3 completed 19
runs including controls/repeats. In round 3, all 6,445 original TRAIN rating rows
remained anchors of weight 1; 5,399 were eligible for extra windows from longer
recordings. Validation/test audio was not augmented. The “uniform” name was an
informal nickname, not a proof of uniform error. It was later rejected in use.

### Later audio-model rounds: calibrating rater scales

All 6,447 TRAIN judgments were retained with equal weight. Calibration was fitted
on TRAIN only; validation targets stayed as raw human ratings. Safe variants
excluded original-model predictions on challenge recordings from calibration
anchors. Earlier teacher/hybrid attempts that used those anchors were excluded
from model selection.

| Model | All MAE | Human >40 MAE | Human >80 MAE | Challenge MAE |
|---|---:|---:|---:|---:|
| Original | 19.007 | 15.215 | 9.470 | 26.525 |
| Selected: `scale_teacher_shift_raw_new` | 17.586 | 13.952 | 8.916 | 20.362 |
| Combined: `averaged_combined` | 17.249 | 14.373 | 8.813 | 19.261 |

The combined candidate used exact shared-clip calibration links, six unfrozen
blocks and **weight averaging of five runs**, yielding one model with unchanged
inference cost. It was not a five-model inference ensemble. The selection
objective was `.45 × MAE>40 + .35 × MAE>80 + .20 × challenge MAE`.
Its additional gain over the selected candidate was 0.0666, below the predefined
0.1 threshold. Neither candidate was a universal winner across ranges.

These are exploratory improvements on repeatedly used validation. Even the
conditional group-bootstrap intervals for >80 improvement included zero; they
also do not account for selection bias. Subsequent capacity and pairwise-loss
series did not establish further improvement. Both selected and combined
candidates were archived and quantized. The combined model was tried in the
bot before the owner requested a return to the original.

### Correcting features and testing cluster consensus

A legacy ECAPA normalization bug had normalized a `[B, 1, 192]` tensor along
the singleton axis, turning coordinates into ±1. Features were recomputed with
correct normalization over 192 coordinates, normalized chunk averaging and a
final unit normalization. **The public features are corrected features.**
Old administrator labels came from K-means with K=200; 176 groups covering
8,757 recordings survived in the later snapshot. There was no fixed cosine
diameter threshold for those groups.

We explored K-means and HDBSCAN, then used complete-linkage clustering on
`d = 1 − cosine_similarity`, with no preassigned K or size limit. Each cluster's
maximum pairwise distance was checked against the threshold. On the frozen
9,003-recording snapshot, including all 246 challenge recordings:

| Maximum distance d | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clusters | 8,199 | 5,817 | 3,703 | 2,240 | 1,352 | 828 | 492 | 294 | 168 |

The annotation interface first used d=0.30, then d=0.50. The public dataset
snapshot has 9,004 recordings and 1,352 clusters after one incremental insertion;
it is not identical to the frozen research snapshot. Acoustic similarity is not
equivalent to perceived passability.

First, recording-level scale alignment at d=0.50 reduced rating-prediction MAE
from 14.32 to 12.81 on a fixed eligible cohort of 3,886 recordings. This number
was coverage of a strict diagnostic, not the size of the dataset.

Next, we averaged within each **recording/rater**, then within each
**cluster/rater** at d=0.30. This later snapshot had 8,582 individual judgments,
43 raters and 6,808 cluster/rater means. There were no newly collected whole-cluster
judgments in that experiment; copied legacy labels were excluded.

For numerical evaluation, five folds held out coarse d=0.80 acoustic groups.
Rater functions were fitted only on other groups. A target recording and all
judgments by its rater were excluded from the neighboring mean; predictions
were returned to the target rater's original scale. The matched cohort contained
1,968 recordings with at least three other recordings and two other raters.

| Label aggregation method | MAE | Human >80 MAE |
|---|---:|---:|
| Cluster means, no scale alignment | 15.110 | 18.982 |
| Rater shift | 14.002 | 17.352 |
| Shift and range | 13.371 | 16.696 |
| Smooth monotone, weak regularization | 13.220 | 15.768 |
| Smooth monotone, medium regularization | 13.149 | 16.119 |
| Smooth monotone, strong regularization | 13.265 | 16.672 |

The smooth functions used six-knot PCHIP, constrained positive derivatives,
regularization toward identity and simpler affine/shift fallbacks for sparse
raters. They preserve each person's ordering. The shared absolute scale is a
chosen convention, not an established objective “true pass percentage.”
For the medium recipe, averaging before/after transforming gave almost identical
MAE: 13.149 versus 13.147. The finding is empirical, not a general equality for
nonlinear functions. Challenge evidence was limited to 36 eligible recordings.

We also searched for a common order of individual recordings, **without features
or clusters**: 5,185 rated recordings, 43 partial rater lists, about 2.18 million
derived comparisons. Approximate Kemeny optimization did not beat numeric means
on held-out raters (roughly 77% pairwise agreement). No global optimum was proved.
An order alone does not supply percentages.

With d=0.30 cluster aggregation, a separate five-fold whole-rater evaluation gave
the following results on the **same 605,840 eligible recording pairs**:

| Consensus method | Held-out pairwise agreement |
|---|---:|
| Mean of individual-recording scores | 77.13% |
| Cluster means | 78.94% |
| Cluster means + smooth rater scales | 79.64% |
| Mean relative ranks | 79.52% |
| Approximate Kemeny order, equal pair weights | 78.83% |

This is agreement with a held-out rater's direction, not a pass score or the
fraction of people who agree. Ties count as half. Only 27.6% of all strict
comparisons were covered by this matched test; comparisons share items and are
dependent. **We have not yet established that retraining HuBERT on this final
cluster-plus-smooth-calibration recipe improves the audio model.**

### Compression and distillation

On the original teacher's historical test, UINT8 ONNX changed predictions by
0.627 points on average, p95 1.854, maximum 5.246. On validation the maximum was
11.596, so rare large deviations remain. Human >80 MAE was 9.915 for the teacher
and 9.966 for ONNX.

| Historical runtime export | Weights, MiB | Median CPU time / 5 s | RSS after inference, MiB |
|---|---:|---:|---:|
| Original FP32 export | 360.8 | 287 ms | 1,172 |
| Original UINT8 ONNX | 117.0 | 193 ms | 385 |

These timings used an Intel Xeon Icelake CPU, four pinned CPU threads, five
warm-ups and 30 measurements, not the production AMD server. Export formats
differ from the released Safetensors/metadata-cleaned ONNX; current file sizes
and hashes are in [assets.json](assets.json). Disk size, process RSS and peak
RAM are different measurements.

Eight student recipes were trained against the original teacher, including
6,000 additional unlabeled TRAIN windows; 25 original/quantized/student formats
were evaluated. The smallest export was about 24.4 MiB but differed by about
7.10 points on average (p95 22.52). The most faithful student was 198.6 MiB with
4.36-point average deviation. Neither replaced the teacher. An attempted QAT
setup failed its simulator-versus-FBGEMM parity check **before training**; no
trained QAT model resulted.

For the “uniform” teacher, narrower/depth-reduced students, warm starts, extra
windows and longer training were also tried. The selected full-precision student still
differed by about 5.35 points versus 0.869 for that teacher's UINT8 ONNX export.
A further ten training runs explored score losses, progressive pruning and
additional channel variants; mixed INT4/HQQ exports were evaluated too.
No smaller candidate met the validation fidelity limits (mean ≤1.5, p95 ≤5,
max ≤12 points). On a 64-window production-AMD check, mixed INT4 candidates were
both less faithful and slower: UINT8 506 ms versus 615/544 ms. Those measurements
used the “uniform” teacher and two threads; they must not be mixed with the
original-teacher Intel benchmark above.

### Available evidence and next steps

[Aggregate metrics](experiments/metrics.json) include the later validation
comparison, cluster-label diagnostics and all 25 original-teacher compression
formats, with source-document checksums. They contain no recording-level
predictions, rater identities or audio. Historical private reports, experimental
checkpoints, pipelines and splits are not all distributed in v0.1.0; the minimal
public fine-tuning example is not an exact reproduction of those runs.

The next useful test is training with the final cluster-plus-monotone labels,
fitting all preprocessing/calibration on TRAIN, and checking against newly
collected independent judgments. More model complexity alone has not solved the
problem. Contributions should report upper-range/challenge performance alongside
overall error, preserve raw judgments and separate derived labels from new votes.

---

<a id="russian"></a>

## Русский

Это сводка экспериментов октября 2026 года, сверенная с сохранёнными отчётами.
**В открытом релизе v0.1.0 находятся исходная HuBERT-модель и её UINT8 ONNX-версия.**
Поздние модели пробовали в боте, но по результатам использования владелица
предпочла исходную и попросила её вернуть. Экспериментальные веса в этот релиз
не включены.

### Как читать результаты

MAE — средняя абсолютная ошибка в пунктах шкалы человеческих оценок 0–100;
меньше — лучше. Диапазоны определяются **оценками людей**, а не прогнозами.
`new_voices` — намеренная подборка записей, на которых исходная модель ошибалась
или могла ошибиться. Её ответы для этих записей нельзя считать правильной разметкой.

Ниже разделены три вида проверок:

- Исторический тест: 847 оценок, 45 групп, всего 40 оценок трудных записей.
  Изначально он был отложен, затем использовался повторно между раундами.
- Поздняя валидация аудиомоделей: 1 234 оценки, 68 групп, включая 61 оценку
  трудных записей. По этой же выборке многократно выбирались настройки.
- Диагностика человеческого консенсуса: перекрёстная проверка по акустическим
  группам или целым оценщикам. Она измеряет предсказуемость и согласованность
  разметки, а не качество новой обученной аудиомодели.

Исходная модель могла видеть часть старых записей со слабыми админскими метками
кластеров. Личности говорящих известны не полностью: акустические группы не
гарантируют независимость выборок по людям. Повторные оценки — наблюдения,
а не независимые записи. Ни одна из этих проверок не устанавливает качество
на новой независимой популяции. Их разбиения и цели также отличаются от
[открытого ECAPA-бейзлайна](baseline_result.json).

### Что пробовали

| Направление | Эксперименты | Наблюдение |
|---|---|---|
| Прямое дообучение | Только голова; последние четыре блока HuBERT; увеличение доли трудных записей до 20% | Общая ошибка исторического теста снизилась, но верхние оценки стали хуже. |
| Потери и представления | Huber/MSE, линейная и порядковая головы, балансировка диапазонов/источников, промежуточные слои, смешивание слоёв, attention pooling | Компромисс между диапазонами не удалось устранить. |
| Другие и более крупные энкодеры | Разморозка 8/12 блоков HuBERT; WavLM Base+ и Large, замороженные и дообученные | Base+ помог части средних и трудных случаев; Large не победил. |
| Аугментации микрофона и помещения | Мягкая АЧХ, шум SNR 25–40 дБ, короткие отражения; без изменения высоты, формант и скорости | Иногда снижали общую ошибку, но не сохраняли исходное качество наверху. Неизменность восприятия была гипотезой. |
| Дополнительные окна записи | Перенос оценки на случайные TRAIN-окна с весом 0,25 или 1; среднее двух окон | Это слабые метки, а не новые человеческие оценки. Выбранный вариант назвали «равномерным». |
| Сглаживание меток по соседям | K-means на TRAIN со 100/200 группами, смеси исходных и кластерных меток, соседи ECAPA/HuBERT, графовое сглаживание, слабые метки неоценённых TRAIN-записей | Бывали лучшие общие MAE, но ограничения качества верхнего диапазона не проходили. |
| Выравнивание шкал оценщиков | Сдвиг или сдвиг/растяжение; админская, общая человеческая, исходной модели и гибридная опоры; точные и приблизительные связи | Наиболее полезный поздний вариант — сдвиг с опорой на исходную модель для обычного TRAIN и без изменения трудных меток. |
| Параметры и усреднение | Сила опоры 0,15–0,50, пределы сдвига, регуляризация, только точные связи, learning rate, 4/6 размороженных блоков, пять seed и среднее весов | Объединённый вариант численно улучшал часть метрик; дополнительный выигрыш критерия был мал. |
| Увеличение сложности | Более глубокая остаточная голова, attentive statistics pooling, смесь последних четырёх слоёв; контроль продолжения обучения, три seed | Убедительной прибавки к прежнему кандидату нет. Дополнительный трансформерный блок HuBERT в этой серии не добавляли. |
| Обучение аудиомодели порядку | Регрессия + потеря типа RankNet; TRAIN-пары одного оценщика с разницей ≥15/25; пять рецептов × три seed | Убедительной прибавки нет; важны сопоставимые контрольные обучения. |
| Человеческий консенсус | Гладкие строго возрастающие шкалы, общий порядок записей, кластерные средние с выравниванием/ранжированием | Кластеры с гладким выравниванием перспективны для разметки; глобальная оптимизация порядка не обошла простое объединение. |
| Сжатие | Хранение FP16, варианты INT8, ONNX UINT8, компактные ученики, прореживание и смешанный INT4 | Практический результат — UINT8 ONNX. Более сильное уменьшение слишком меняло оценки. |

### Ранние раунды аудиомоделей: компромисс верхнего диапазона

Это **исторический тест**, а не поздняя валидация из следующей таблицы.

| Модель / рецепт | Общая MAE | Трудные MAE | Человеческие оценки >80 MAE |
|---|---:|---:|---:|
| Исходная HuBERT | 19,665 | 20,852 | 9,915 |
| Раунд 1: четыре последних блока, больше трудных записей | 15,269 | 18,637 | 12,63 |
| Раунд 2: выбранная WavLM Base+ | 15,594 | 17,110 | 20,96 |
| Раунд 3: канальные аугментации + перенос меток с полным весом («равномерная») | 14,755 | 19,641 | 14,140 |

Во втором раунде завершились 18 основных вариантов и четыре повтора; в третьем —
19 запусков с контролями и повторами. В третьем раунде все 6 445 исходных
TRAIN-оценок оставались якорями с весом 1; для 5 399 можно было брать
дополнительные окна длинных записей. Валидация и тест не аугментировались.
«Равномерная» — условное название, не доказательство равномерности ошибок.
Позднее эта модель была отвергнута по результатам использования.

### Поздние раунды: выравнивание шкал оценщиков

Все 6 447 TRAIN-оценок сохранены с одинаковым весом. Калибровка подбиралась
только на TRAIN, а проверочные метки оставались исходными человеческими оценками.
В корректных вариантах ответы исходной модели на трудных записях исключены
из опор. Предварительные teacher/hybrid-запуски с такими опорами исключены
из выбора модели.

| Модель | Общая MAE | Люди >40 MAE | Люди >80 MAE | Трудные MAE |
|---|---:|---:|---:|---:|
| Исходная | 19,007 | 15,215 | 9,470 | 26,525 |
| Отобранная: `scale_teacher_shift_raw_new` | 17,586 | 13,952 | 8,916 | 20,362 |
| Объединённая: `averaged_combined` | 17,249 | 14,373 | 8,813 | 19,261 |

Объединённая использует только точные общие клипы для связей калибровки,
шесть размороженных блоков и **среднее весов пяти запусков**. Это одна модель
с прежней стоимостью инференса, а не пять одновременно работающих моделей.
Критерий выбора: `0,45 × MAE>40 + 0,35 × MAE>80 + 0,20 × MAE трудных`.
Дополнительный выигрыш составил 0,0666 при заранее заданном пороге 0,1.
Универсального победителя во всех диапазонах не было.

Это исследовательские улучшения на многократно использованной валидации.
Даже условные групповые bootstrap-интервалы улучшения >80 включали ноль;
они также не учитывают оптимизм выбора по этой выборке. Последующие серии
с увеличением сложности и попарной потерей не установили дальнейший выигрыш.
Отобранная и объединённая сохранены и квантованы. Объединённую пробовали
в боте, затем владелица попросила вернуть исходную модель.

### Исправление признаков и проверка кластерного консенсуса

В старом извлечении ECAPA обнаружилась ошибка: тензор `[B, 1, 192]`
нормализовался по оси длиной 1, превращая координаты в ±1. Признаки пересчитали
с нормализацией по 192 координатам, усреднением нормированных векторов отрезков
и итоговой единичной нормализацией. **В открытом датасете признаки исправлены.**
Старые админские метки происходят из K-means с K=200; в позднем снимке
сохранились 176 групп / 8 757 записей. Фиксированного порога косинусного
диаметра у тех групп не было.

Пробовали K-means и HDBSCAN, затем полное связывание по
`d = 1 − cosine_similarity`, без заданного числа или предела размера групп.
Для каждого кластера проверили максимальное расстояние всех пар. На снимке
9 003 записей, включая все 246 трудных:

| Максимальное расстояние d | 0,10 | 0,20 | 0,30 | 0,40 | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Кластеров | 8 199 | 5 817 | 3 703 | 2 240 | 1 352 | 828 | 492 | 294 | 168 |

В интерфейсе оценивания сначала использовался d=0,30, затем d=0,50.
Открытый снимок содержит 9 004 записи и 1 352 кластера после одного
инкрементального добавления; это другой снимок, чем в исследовании.
Акустическая близость не равна близости воспринимаемого пасса.

Сначала выравнивание шкал на уровне записей при d=0,50 снизило ошибку
предсказания оценки с 14,32 до 12,81 на одной выборке из 3 886 подходящих
записей. Это охват строгой диагностики, а не размер датасета.

Затем при d=0,30 усредняли сначала внутри пары **запись/оценщик**, потом
**кластер/оценщик**. Этот снимок содержал 8 582 индивидуальные оценки,
43 оценщика и 6 808 кластерных средних отдельных людей. Новых оценок целых
кластеров в том эксперименте ещё не было; скопированные старые метки исключены.

Числовая проверка использовала пять частей по крупным акустическим группам
d=0,80. Шкалы подбирались только по другим группам. Из соседского среднего
полностью исключались целевой файл и все оценки его оценщика, а прогноз
возвращался в исходную шкалу проверяемого человека. Сопоставимая выборка:
1 968 записей, минимум три других файла и два других оценщика для каждой.

| Способ объединения меток | MAE | Люди >80 MAE |
|---|---:|---:|
| Кластерное среднее без выравнивания | 15,110 | 18,982 |
| Сдвиг оценщика | 14,002 | 17,352 |
| Сдвиг и растяжение | 13,371 | 16,696 |
| Гладкая монотонная, слабая регуляризация | 13,220 | 15,768 |
| Гладкая монотонная, средняя регуляризация | 13,149 | 16,119 |
| Гладкая монотонная, сильная регуляризация | 13,265 | 16,672 |

Гладкие функции — PCHIP с шестью узлами, ограничениями положительной
производной и регуляризацией к исходной шкале. При малом числе опор
используются аффинная функция или сдвиг. Порядок оценок каждого человека
сохраняется. Общая абсолютная шкала условная, а не установленный объективный
«истинный процент пасса». Для средней регуляризации порядок усреднения
и преобразования почти не влиял на MAE: 13,149 против 13,147. Это наблюдение
для конкретных данных, а не общее равенство нелинейных преобразований.
На трудных записях проверка доступна только для 36 файлов.

Отдельно искали общий порядок обычных записей **без признаков и кластеров**:
5 185 оценённых записей, 43 частичных списка, около 2,18 млн производных пар.
Приближённый поиск Кемени не обошёл числовые средние на отложенных оценщиках
(примерно 77% совпадения порядка). Глобальный оптимум не доказан.
Сам по себе порядок не задаёт проценты.

После кластерного усреднения d=0,30 отдельная проверка по пяти частям целых
оценщиков дала результаты на **одних и тех же 605 840 парах записей**:

| Метод консенсуса | Совпадение порядка с отложенным человеком |
|---|---:|
| Среднее оценок отдельных записей | 77,13% |
| Кластерные средние | 78,94% |
| Кластерные средние + гладкие шкалы | 79,64% |
| Средние относительных рангов | 79,52% |
| Приближённый порядок Кемени, одинаковый вес пар | 78,83% |

Это доля совпадения направления пары с оценщиком, а не процент пасса
и не доля согласных людей. Ничья получает половину балла. Сопоставимая проверка
покрывает только 27,6% строгих пар; пары зависимы, поскольку содержат общие
записи. **Улучшение HuBERT после дообучения именно на итоговом рецепте
«кластеры + гладкое выравнивание» ещё не установлено.**

### Квантование и дистилляция

На историческом тесте исходного учителя UINT8 ONNX менял ответы в среднем
на 0,627 пункта, p95 — 1,854, максимум — 5,246. На валидации максимум достигал
11,596: редкие заметные отклонения остаются. MAE к людям в диапазоне >80:
9,915 у учителя и 9,966 у ONNX.

| Исторический экспорт для запуска | Веса, МиБ | Медиана CPU на 5 с | RSS после инференса, МиБ |
|---|---:|---:|---:|
| Исходная FP32 | 360,8 | 287 мс | 1 172 |
| Исходная UINT8 ONNX | 117,0 | 193 мс | 385 |

Это Intel Xeon Icelake, четыре закреплённых CPU-потока, пять прогревов
и 30 измерений, а не основной AMD-сервер. Исторические форматы отличаются
от опубликованных Safetensors и ONNX с очищенными метаданными; актуальные
размеры и хеши — в [assets.json](assets.json). Размер файла, RSS процесса
и пиковая RAM — разные показатели.

Обучили восемь рецептов учеников исходного учителя, включая 6 000
дополнительных неразмеченных TRAIN-окон; оценили 25 исходных, квантованных
и студенческих форматов. Минимальный экспорт — около 24,4 МиБ, но среднее
отклонение около 7,10 пункта, p95 — 22,52. Наиболее близкий ученик —
198,6 МиБ и 4,36 пункта среднего отклонения. Замены учителя не получилось.
Попытка QAT остановлена **до обучения** из-за несовпадения симулятора
с FBGEMM; готовой обученной QAT-модели нет.

Для «равномерного» учителя также пробовали более узких и коротких учеников,
инициализацию прежними весами, дополнительные окна и более долгое обучение.
Выбранный полноточный ученик всё ещё отличался примерно на 5,35 пункта против 0,869
у UINT8 ONNX этого учителя. Ещё десять обучений проверяли функции потерь,
постепенное прореживание и дополнительные канальные варианты; оценили также
смешанные INT4/HQQ-экспорты. Ни один меньший кандидат не прошёл ограничения
валидационного отклонения: среднее ≤1,5, p95 ≤5, максимум ≤12 пунктов.
На 64 окнах основного AMD-сервера смешанные INT4 были и менее точными,
и более медленными: 506 мс у UINT8 против 615/544 мс. Это «равномерный» учитель
и два потока; эти измерения нельзя смешивать с Intel-тестом исходной модели.

### Доступные результаты и дальнейшая проверка

В [машиночитаемых метриках](experiments/metrics.json) — поздняя валидация,
диагностика кластерных меток и все 25 вариантов сжатия исходного учителя,
с контрольными суммами исходных документов. Там нет прогнозов отдельных
записей, идентификаторов оценщиков или аудио. Исторические приватные отчёты,
экспериментальные веса, процессы обучения и разбиения не полностью входят
в v0.1.0; минимальный открытый пример дообучения не воспроизводит те запуски
в точности.

Полезная следующая проверка — обучение на итоговых кластерных метках
с монотонным выравниванием, подбором всех преобразований только на TRAIN
и оценкой по новым независимым человеческим ответам. Одно увеличение сложности
пока проблему не решило. Для новых экспериментов нужны отдельные метрики
верхнего диапазона и трудных записей, сохранение исходных оценок и явное
отделение производных меток от новых голосов людей.
