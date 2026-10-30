# Smooth expert targets / Гладкие экспертные цели

[English](#english) · [Русский](#russian) · [Earlier experiments](EXPERIMENTS_2026-10-30.md) · [Aggregate JSON](experiments/smoothing-2026-10-31.json) · [Figures](docs/figures/2026-10-31/README.md)

<a id="english"></a>

## English

Experiments completed October 30–31, 2026. We tested whether individual weak
scores interpolated in voice-embedding space improve on assigning one expert
score to an entire acoustic cluster. **No robust improvement over the retained
expert models was established. No new model weights are released for this round.**
The selected models and artifacts in v0.2.0 remain available.

### Local hypothesis check

The frozen expert snapshot contains 1,406 genuine judgments: TRAIN 989 / VAL 219 /
TEST 198. A cluster judgment counts once for its actually heard representative.
Propagating that judgment does not create more human observations.

Features were corrected, frozen pretrained SpeechBrain ECAPA192 vectors, with
no encoder fitting to the expert targets. We compared recording-level vectors
and a secondary approximation of the heard interval obtained by overlap-weighted
averaging of existing 3.5-second chunk vectors. The latter is not an exact
embedding of a five-second waveform.

Candidates included nearest example/cluster, distance-weighted neighbors,
globally smooth Gaussian-weighted averaging, RBF kernel ridge, linear ridge and
a soft-label graph regressor. Hyperparameters used four acoustic-group folds
inside TRAIN only, minimizing half MAE plus half five-score-bin macro MAE.
Graph fitting used only TRAIN groups, including their unlabeled vectors; held
inputs were queried at prediction time. The local overall choice was ten
neighbors with exponential cosine-distance weights, bandwidth 0.03.

Two different questions were evaluated. For the interpolation diagnostic,
complete recordings and exact file/PCM copies were hidden within TRAIN while
other acoustically related recordings could remain. For generalization, entire
acoustic groups were held out. Parameters for the interpolation diagnostic were
fixed from the acoustic-group CV; no new search was performed there.

| Method, recording-level vectors | Recording-holdout MAE, n=989 | Reused group TEST MAE, n=198 |
|---|---:|---:|
| Cluster transfer, nearest rated cluster fallback | 15.34 | 20.43* |
| Nearest rated example | 14.19 | 19.92 |
| Globally smooth weighted mean | 12.95 | 18.52 |
| Weighted nearest neighbors | 12.66 | 17.98 |
| RBF kernel ridge | 14.47 | 17.42 |

*All fine clusters are contained in one group split. A held TEST cluster has no
TRAIN expert label, so this TEST baseline uses the nearest rated TRAIN cluster.
The recording-holdout baseline uses its own cluster's remaining expert label
when one exists, otherwise the nearest rated cluster. Only 46 hidden examples
had another expert-rated recording in their own fine cluster. This limits direct
evidence about interpolation inside already rated clusters.

The recording-holdout neighbor improvement was −2.68 points, with approximate
group-bootstrap 95% interval [−3.45, −1.95]. On group TEST, its difference from
the nearest-cluster baseline was −2.45 [−5.55, +0.63]. These bootstrap intervals
condition on the fitted predictions and do not fully account for model search
or overlapping CV training sets. Local success motivated a controlled GPU test;
it did not establish that an audio model trained on synthetic targets would win.

### Exact features and common weak-label pool

The GPU round used a single RTX 3090. Exact ECAPA192 vectors were recomputed in
FP32 for 1,405 unique actually heard five-second waveforms and 11,756 candidate
TRAIN windows. Both took about 30 seconds. The pretrained encoder was frozen;
TF32 was disabled. On 144 shared CPU/GPU inputs, maximum vector-coordinate
difference was 2.84e-6 and maximum cosine discrepancy 1.79e-7.

Functions were retuned using only the genuine TRAIN labels and the same grouped
CV rule. The chosen smooth mean had bandwidth 0.02; the chosen neighbor model
used three neighbors and bandwidth 0.1, with no inverse-bin weighting in either
function. Thus the exact-window neighbor count differed from the earlier
recording-vector diagnostic's ten neighbors.

A **shared pool of 4,835 weak windows from 2,971 recordings and 690 clusters**
was used for all three arms. Each window had cosine distance at most 0.3 from a
genuinely rated TRAIN window. Exact repeated/heard inputs and windows with less
than three seconds of real audio were excluded. This trust radius in exact
window space is separate from the miniapp's 0.5 clustering threshold on
recording-level vectors.

The 989 real TRAIN scores were unchanged. Their five-bin inverse-frequency
weights were fitted on genuine TRAIN judgments only and capped at three.
Weak weights used a shared confidence factor exp(−(distance/0.3)²), fine-cluster
size normalization and the shared cluster-score bin weights. Their total loss
mass was 25% of the human mass. The three arms had identical waveforms, weights,
batch order, optimizer settings and initialization within a seed; only weak
targets differed. Original-model predictions were not used as targets, including
for the deliberately difficult `new_voices` collection.

### Twelve completed runs

- Three refinements of the existing expert HuBERT, one per target method.
  Twelve epochs each; last eight blocks, backbone LR 5e-7, head LR 1e-5,
  microbatch 12, accumulation four, AdamW weight decay 0.01.
- Six trainings initialized from the original HuBERT: three target methods ×
  two seeds. Twelve epochs each; last eight blocks, backbone LR 1e-6, head LR
  5e-5; other shared settings as above.
- Three refinements of expert ECAPA256 mini: twelve epochs each, encoder LR
  2e-6, head LR 1e-5, batch 32, accumulation two.

All used Huber loss with beta 0.05 on the 0–1 scale. Epoch zero was eligible to
prevent selecting a degraded refinement. Epoch/model selection used VAL only:
0.2 score-bin macro MAE + 0.2 cluster macro MAE + 0.3 MAE above 40 +
0.3 MAE above 80. After all first-seed full-model refinements retained epoch zero,
their repeat seed was skipped by a VAL-only futility rule. One partial repeat
was interrupted and is not counted among the twelve completed runs. Original-
initialization trials and mini trials were then prioritized within the paid
deadline. TEST was not used for queue adaptation or model selection.

| Initialization / targets | Seeds | TEST MAE | TEST >80 MAE | Expert order error |
|---|---:|---:|---:|---:|
| Original HuBERT / cluster | 2 | 16.38 | 11.88 | 19.97% |
| Original HuBERT / smooth | 2 | 15.63 | 13.09 | 19.83% |
| Original HuBERT / neighbors | 2 | 16.42 | 11.64 | 19.92% |
| Previous expert full, reference | — | 16.19 | 13.01 | 20.47% |
| Previous expert mini, reference | — | 15.08 | 12.70 | 18.99% |

New-training rows average their two seeds. TEST has 198 real judgments, including
32 above 80. Order error uses unequal-target pairs of distinct heard inputs,
excluding target ties and counting model ties as half an error.

### Why the smooth mean is not a confirmed winner

In seed one, smooth targets gave 15.79 versus 15.74 for the cluster control.
In seed two, they gave 15.46 versus 17.01, with a worse upper-range error. The
second comparison also selected different epochs: six for smooth and five for
cluster. Their best VAL objectives differed by only about 0.003 points. At the
same epoch six, their VAL MAEs were almost identical: 13.417 and 13.423.
Checkpoint-selection instability is therefore a plausible contributor to the
TEST difference; a matched-epoch TEST counterfactual was not established.

For seed one, all three methods selected epoch six. Smooth and neighbor models
then changed TEST predictions by only 0.27 and 0.41 points on average. With the
actual weak-example weights, their targets differed from cluster targets by
1.56 and 2.43 points; that weak loss itself had only one quarter of human mass.

The fixed VAL selection retained the previous full expert and mini expert
weights. All six selected epoch-zero checkpoints were checked tensor-by-tensor
against those initial weights. No stable replacement for the deployed expert
mini was found. The twelve selected checkpoints, exact features, weak targets,
histories and reports were archived privately; SHA256 checks passed. These
unsuccessful/ambiguous checkpoints are **not uploaded to GitHub or releases**.

The [aggregate JSON](experiments/smoothing-2026-10-31.json),
[metric table](experiments/smoothing-2026-10-31.csv) and
[figures](docs/figures/2026-10-31/README.md) contain no audio, per-record
predictions, original filenames or rater IDs. The public v0.1.0 feature dataset
is unchanged. This is a reused-test, single-expert study with unverified
speaker-disjointness, not a new independent population benchmark.

<a id="russian"></a>

## Русский

Эксперименты 30–31 октября 2026 года проверяли индивидуальные слабые оценки
записей как гладкую функцию в пространстве голосовых векторов вместо одной
оценки на кластер. **Устойчивого улучшения сохранённых экспертных моделей
не получено. Новые веса этого этапа не публикуются.** Успешные отобранные модели
прежнего этапа остаются в v0.2.0.

### Локальная проверка

Зафиксированы 1 406 настоящих прослушиваний: TRAIN989 / VAL219 / TEST198.
Оценка кластера учитывается один раз для действительно услышанного примера.
Производные метки не являются новыми человеческими ответами.

Использовали исправленные ECAPA192-векторы записей от замороженного pretrained
энкодера. Второе представление приближало услышанный интервал средним готовых
векторов 3,5-секундных частей с весами по пересечению. Это ещё не точный вектор
пятисекундного аудио.

Проверили ближайший пример/кластер, взвешенных соседей, гладкое усреднение с
гауссовыми весами, RBF kernel ridge, линейную ridge и мягкую графовую регрессию.
Параметры выбирались только на TRAIN в четырёх фолдах акустических групп:
цель — полусумма MAE и средней MAE пяти диапазонов оценок. Граф строился только
по TRAIN-группам, включая их неразмеченные векторы. Основной локальный выбор —
десять соседей, exp(−косинусное расстояние/0,03).

Разделяли две проверки: интерполяцию со скрытием целой записи и её точных
файловых/PCM-копий, когда похожие записи могли оставаться, и перенос на целиком
скрытые акустические группы. В первой проверке использовали уже выбранные
параметры, без нового поиска.

| Метод на векторах записей | Скрытые записи, n=989: MAE | Прежний групповой TEST, n=198: MAE |
|---|---:|---:|
| Кластерная оценка / ближайший оценённый кластер | 15,34 | 20,43* |
| Ближайший экспертный пример | 14,19 | 19,92 |
| Гладкое взвешенное среднее | 12,95 | 18,52 |
| Взвешенные соседи | 12,66 | 17,98 |
| Гладкая RBF-регрессия | 14,47 | 17,42 |

*Все мелкие кластеры целиком находятся в одной выборке. Поэтому для TEST нет
оценки собственного кластера из TRAIN: baseline выбирает ближайший оценённый
TRAIN-кластер. В проверке скрытых записей настоящая оценка того же кластера
оставалась только у 46 примеров. Это ограничивает доказательства именно
интерполяции внутри уже оценённых кластеров.

Разница соседского варианта и кластерного baseline при скрытии записей:
−2,68 пункта, приблизительный групповой bootstrap 95% [−3,45; −1,95]. На
групповом TEST: −2,45 [−5,55; +0,63]. Интервалы условны на выбранных прогнозах
и не учитывают полностью подбор моделей и пересечение обучающих фолдов.
Локальный результат дал основание проверить обучение аудиомодели на GPU.

### Точные признаки и общая слабая разметка

На RTX 3090 за примерно 30 секунд пересчитали точные ECAPA192-векторы 1 405
уникальных услышанных пятисекундных окон и 11 756 кандидатных TRAIN-окон.
Encoder заморожен, FP32, TF32 отключён. На 144 общих с CPU окнах максимальная
разница координаты 2,84e-6, косинусная погрешность 1,79e-7.

По групповому TRAIN-CV для точных окон выбраны гладкое среднее с шириной 0,02
и три соседа с шириной 0,1, без перевзвешивания диапазонов внутри функций.
Это отличается от прежних десяти соседей на векторах целых записей.

Общий для всех способов набор: **4 835 слабых окон из 2 971 записи и 690
кластеров**. Расстояние до реального экспертного TRAIN-окна ≤0,3; исключены
точные повторы, услышанные входы и окна с реальным аудио короче трёх секунд.
Этот радиус доверия по точным окнам отличается от порога кластеров миниаппа
0,5 по векторам целых записей.

Настоящие 989 TRAIN-оценок не изменены. Балансировка пяти диапазонов обучалась
только по ним, веса ограничены тремя. Слабым примерам даны общие для всех
вариантов веса: exp(−(расстояние/0,3)²), нормализация по размеру мелкого кластера
и вес диапазона исходной кластерной оценки. Их общая масса — 25% человеческой.
Аудио, веса, порядок minibatch и начальные параметры внутри seed одинаковы;
меняются только слабые цели. Оценки старой модели, включая `new_voices`, в
качестве обучающих целей не использовались.

### Двенадцать завершённых обучений

- Три дообучения готовой экспертной HuBERT, по одному на способ разметки:
  12 эпох, последние восемь блоков, LR backbone 5e-7 / головы 1e-5,
  microbatch12, накопление4, AdamW weight decay0,01.
- Шесть обучений от исходной HuBERT: три способа × два seed, по 12 эпох,
  LR backbone1e-6 / головы5e-5; остальные настройки общие.
- Три дообучения экспертной ECAPA256 mini: по 12 эпох, LR encoder2e-6 /
  головы1e-5, batch32, накопление2.

Потеря Huber, beta0,05 на шкале 0–1. Начальная epoch0 также участвовала в отборе.
Критерий VAL: 0,2 средней MAE диапазонов + 0,2 средней MAE кластеров +
0,3 MAE выше40 + 0,3 MAE выше80. После трёх дообучений полной экспертной,
сохранивших epoch0, повтор этого этапа остановлен по VAL-критерию
бесперспективности. Один частичный повтор прерван и не включён в двенадцать.
Время направили на обучение от исходной модели и проверку мини. Решения об
очереди и выборе моделей принимались до TEST, без его оценок.

| Старт / слабые метки | Seed | TEST MAE | TEST >80 MAE | Ошибка порядка |
|---|---:|---:|---:|---:|
| Исходная HuBERT / кластеры | 2 | 16,38 | 11,88 | 19,97% |
| Исходная HuBERT / гладкие | 2 | 15,63 | 13,09 | 19,83% |
| Исходная HuBERT / соседи | 2 | 16,42 | 11,64 | 19,92% |
| Прежняя экспертная полная | — | 16,19 | 13,01 | 20,47% |
| Прежняя экспертная мини | — | 15,08 | 12,70 | 18,99% |

Новые варианты усреднены по двум seed. В TEST198 оценок, из них32 выше80.
Ошибка порядка — доля обратных пар с разными экспертными оценками; одинаковые
входы и равные цели исключаются, равные прогнозы дают половину ошибки.

### Почему средний выигрыш не подтверждает новую лучшую модель

У первого seed гладкий вариант почти совпал с контролем: 15,79 против15,74.
У второго он получил15,46 против17,01, но хуже оценивал верхний диапазон.
При этом выбраны разные эпохи: гладкая6 и кластерная5. Лучшие VAL-критерии
различались примерно на0,003. На общей эпохе6 VAL MAE практически одинаковы:
13,417 и13,423. Нестабильность отбора эпохи могла внести вклад в разницу TEST;
контрфактический TEST на одинаковой эпохе не установлен.

В первом seed все способы выбрали эпоху6. Изменение TEST-прогнозов относительно
кластерного контроля — всего0,27 пункта для гладкой функции и0,41 для соседей.
С учётом реальных весов слабые цели отличаются от кластерных на1,56 и2,43
пункта, а их суммарная масса — четверть человеческой.

Итоговый отбор по VAL сохранил прежние экспертные полную и мини-модели.
Тензоры всех шести выбранных epoch0-чекпойнтов сравнены с начальными весами.
Устойчивой замены установленной экспертной мини не найдено. Все двенадцать
выбранных чекпойнтов, признаки, цели, истории и отчёты сохранены приватно и
проверены по SHA256. **Неудачные и неоднозначные веса этого этапа в GitHub и
релизы не загружаются.**

Публичны [агрегированные данные](experiments/smoothing-2026-10-31.json),
[таблица метрик](experiments/smoothing-2026-10-31.csv) и
[графики](docs/figures/2026-10-31/README.md). Аудио, прогнозы отдельных записей,
исходные имена и ID оценщиков исключены. Датасет v0.1.0 не изменялся. Это
исследование шкалы одного эксперта с ранее использованным TEST и без доказанной
независимости по говорящим, а не новый независимый бенчмарк.
