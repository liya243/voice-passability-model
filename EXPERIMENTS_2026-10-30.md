# Expert training and independent ECAPA students / Экспертное обучение и ученики ECAPA

[English](#english) · [Русский](#russian) · [Earlier history](EXPERIMENTS.md) · [Release v0.2.0](https://github.com/liya243/voice-passability-model/releases/tag/v0.2.0)

<a id="english"></a>

## English

This continuation records the expert-data experiments completed on October 30,
2026. It supplements the earlier history rather than replacing unsuccessful
results. Nine selected checkpoints are released; intermediate trials and failed
students are described below. "Selected" means selected for a stated validation
objective, not proven superior on a new population.

### Supervision, imbalance and audio audit

The frozen expert cohort contains **1,406 real listening judgments from one vocal
expert**: 206 individual judgments and 1,200 genuine cluster-representative
judgments. The split is TRAIN 989 / VAL 219 / TEST 198. Another 159 adopted
cluster labels are derived labels, not additional listening judgments. A cluster
judgment counts once for the recording actually heard; assigning it to other
members is explicitly weak supervision.

The corpus contains 9,004 recordings, 1,352 fine acoustic clusters at cosine
distance threshold 0.5 and 294 coarse groups. Recordings and related acoustic
groups are kept together by the frozen split. This does **not** establish
speaker-disjointness. Score-bin balancing, group-aware selection and separate
upper-range metrics address the highly uneven score distribution. Different
loss/sampling settings were compared rather than treating all windows as
independent equally informative labels.

Silero speech activity and YAMNet sound classification audited all 9,004 files
(33,881 analysis tasks). Only eight whole recordings were strongly suspicious:
five music/no detected speech, two other non-speech and one silence. Another
285 were uncertain, not proven rubbish. Nothing was deleted. Hard exclusion
and soft quality weighting were compared with controls over three seeds;
effects were small. CPU parallelism was faster than the tested GPU/CPU hybrid
for this particular pipeline; these timings are not model-inference timings.

### Full models and context length

Expert HuBERT fine-tuning and refinement varied learning rate, loss weights,
balancing, regularization and upper-range safeguards. The retained `refine_02`
checkpoint is `expert-hubert-v1`. A later upper-guard selection resolved to the
same weights, so it is not published twice. Overall expert error improved over
the original while the original retained an advantage for ratings above 80.

The context/architecture stage completed **26 trials**: 22 initial runs plus
four ECAPA duration extensions. It compared full HuBERT (last eight of twelve
blocks trainable), WavLM and pretrained ECAPA, with 2/3/5/8/10-second contexts
where applicable. Supervision scopes were the actually heard segment, other
windows in its TRAIN recording, or other TRAIN cluster members; transferred
labels had a separate weak budget (0.25). A fresh HuBERT head was also tested.
Longer audio does not create a genuinely rated longer segment: those targets
remain transferred five-second judgments. Additional planned runs were stopped
by the server budget and are not reported as completed.

WavLM did not win. The selected supervised ECAPA used two-second TRAIN windows
and cluster propagation, but **five-second inference** was better than using
two-second inference. `expert-ecapa-supervised` scored TEST MAE 13.66 and
upper-range MAE 12.82. A full HuBERT selected on the common long-recording VAL
subset, `expert-hubert-recording`, scored 16.00 / 12.52. Its score is from a
different selection objective, not an interchangeable new independent test.

### Distillation that finally helped

Reduced HuBERT students from earlier rounds had not preserved predictions well.
We switched to **independent ECAPA-TDNN waveform students**, not fewer HuBERT
layers. An initial eight-run analytic-feature study used 88 eGeMAPS features
and 25 temporal summaries including pitch, harmonic and spectral information;
it was not pitch-only. Its tiny MLP/CNN had about 20–23 thousand parameters.
Although one MLP reached expert TEST MAE 15.97, its mean teacher deviation was
12.98 points. This was insufficient for faithful teacher replacement. These
analytic students are documented but not promoted as successful release assets.

The successful expert ECAPA queue completed eight runs with no failed/deferred
jobs. It compared a pretrained 1,024-channel encoder (20.79M inference
parameters) and a randomly initialized 256-channel mini (1.78M parameters).
Losses combined teacher scores, human labels and optional feature matching.
Hybrid runs balanced score distillation and human supervision; feature weight
0.05 was tested. A learned 192→768 alignment matched centered teacher features
using TRAIN only and was discarded for inference. The teacher supplied 12,745
TRAIN units; genuine expert labels retained their separate role.

Three roles are retained: `expert-ecapa-distilled-human` (human-validation
selection), `expert-ecapa-distilled-fidelity` (teacher-fidelity selection), and
`expert-mini` (compact human-validation selection). Their expert TEST MAEs were
14.26, 13.84 and 15.08 respectively. The mini is not an exact copy of HuBERT;
its improved human score is a different property from fidelity to the teacher.

The same architecture was then distilled from the original HuBERT in four
completed trials: ECAPA256 score-only or score+features (weights 0.05/0.01),
and ECAPA384 with features. No human loss was used in these original-teacher
runs. **418 `new_voices` TRAIN windows were excluded from teacher supervision**,
leaving 12,327 trusted units: that challenge set was deliberately collected
where the original might be wrong. The six-model comparison uses `original-mini`
selected by frozen VAL fidelity among ECAPA256 candidates. The human-selected
epoch and the larger fidelity-selected ECAPA384 are also released under
distinct names; they must not be confused with that benchmark checkpoint.

### Same-input comparison of six models

All six were evaluated on 32,312 unique five-second inputs across the 9,004
recordings, including 18,539 genuine judgments by 43 raters. Expert TEST contains
198 judgments (32 above 80); it was previously inspected. Preprocessing
reconstructed what raters heard: exact segment start, listener PCM16 roundtrip,
mono/resampling/padding and per-window peak normalization. Thirty-six sampled
input reconstructions were bitwise identical. CPU/GPU differences stayed below
0.00012 points in the check.

| Model | Expert TEST MAE ↓ | Human >80 MAE ↓ | Expert order inversions ↓ | CPU ms / 5 s ↓ |
|---|---:|---:|---:|---:|
| Original HuBERT | 21.99 | 10.64 | 23.86% | 181.3 |
| Original UINT8 | 22.14 | 10.47 | 24.02% | 115.2 |
| Original mini | 19.29 | 9.34 | 19.70% | 15.6 |
| Expert HuBERT | 16.19 | 13.01 | 20.47% | 180.7 |
| Expert UINT8 | 16.34 | 12.56 | 20.45% | 115.1 |
| Expert mini | 15.08 | 12.70 | 18.99% | 15.4 |

Order error compares the order of distinct actually heard inputs with unequal
expert ratings: 18,849 TEST pairs, excluding 654 tied-target pairs. A reversed
order is an inversion; prediction ties count as half an error in the separately
reported order-error statistic (there were none in TEST). It is **not** the
fraction of examples with inaccurate scores. Gap-restricted results,
Spearman/Kendall and grouped paired bootstrap are in the aggregate report.

Paired group-bootstrap 95% intervals: original-mini minus original MAE
−2.70 [−4.78, −0.88]; expert-mini minus expert MAE −1.11 [−3.25, +0.95].
Thus the latter difference is not conclusive. Timings use the GPU host's CPU,
four pinned cores, three rounds/60 measurements, batch one; frontend included,
decoding/network excluded. They are not end-to-end bot latency. Published
Safetensors sizes differ slightly from the original checkpoint containers.

Across all human judgments, MAE is 14.09 original / 13.96 original-mini /
14.47 expert / 14.38 expert-mini. On the reused all-rater TEST (2,773 judgments),
the corresponding values are 17.54 / 17.22 / 16.67 / 16.07. One person supplied
48% of all ratings. Equal-rater weighting changes both rater influence and the
voices being evaluated; it is a sensitivity check, not a universal ground truth.
Expert tuning optimizes one expert's scale, not unanimous listener perception.

### Distributions, release and practical trial

[Expert-only figures](docs/figures/2026-10-30/expert-only/README.md) and
[all-human figures](docs/figures/2026-10-30/all-human/README.md) show signed and
absolute error histograms for all judgments and TEST, plus empirical CDFs.
All-data plots include TRAIN. In expert TEST, expert-mini has median absolute
error 9.81, P90 33.74 and errors above 40 in 7.07% of judgments. These describe
a distribution, not just an average.

Nine selected checkpoints are exported as tensor-only Safetensors, plus expert
UINT8 ONNX and self-contained TorchScript runtimes for both minis. All exported
tensors are bitwise checked against their source states. The supervised ECAPA
export removes unused speaker-classification components. TorchScript includes
peak normalization, Fbank and ECAPA: only PyTorch is needed at inference.

On October 30 the expert-mini runtime was enabled for both bot recording
scoring and miniapp live scoring. One shared worker avoids loading a separate
model for each route. The deployed 7.30 MB runtime retains five-second context,
750 ms live interval and admission-time latest-window selection. Public live
WebSocket smoke testing returned an inference time of 33 ms on the production
CPU; this is one smoke measurement, not a load benchmark. Previous weights and
rollback files were preserved. Recording tails shorter than three seconds are
still omitted before padding.

Audio, private file identifiers, original rater IDs, per-record predictions and
training containers are **not** published. The v0.1.0 pseudonymized feature
dataset remains a separate older snapshot; this release does not silently add
the later expert judgments to it. Training scripts/configurations from all
private queues are not fully reproduced here; released inference and aggregate
history are reproducible to the extent described in [verification](VERIFICATION_V0.2.md).
TEST reuse, incomplete speaker identities and historical training overlap of
the original prevent claiming a new independent accuracy guarantee.

<a id="russian"></a>

## Русский

Продолжение фиксирует завершённые к 30 октября 2026 года эксперименты.
[Прежняя история](EXPERIMENTS.md) сохранена вместе с неудачными результатами.
В релизе девять выбранных чекпойнтов. «Выбранный» означает выбор по конкретной
валидационной цели, а не доказанное превосходство на новой аудитории.

### Разметка, дисбаланс и проверка аудио

Зафиксированы **1 406 настоящих прослушиваний одного эксперта вокала**:
206 индивидуальных оценок и 1 200 оценок прослушанных представителей кластеров.
TRAIN / VAL / TEST: 989 / 219 / 198. Ещё 159 унаследованных кластерных меток
не считаются новыми прослушиваниями. Оценка кластера учитывается один раз для
действительно услышанной записи; перенос на другие записи — слабая разметка.

В корпусе 9 004 записи, 1 352 мелких акустических кластера при косинусном
пороге 0,5 и 294 крупные группы. Связанные записи и группы не пересекают
зафиксированные выборки. Независимость по говорящим этим не доказана.
Неравномерность оценок учитывали балансировкой диапазонов, выбором по группам,
отдельной проверкой верхнего диапазона и сравнением вариантов весов и потерь.

Silero VAD и YAMNet проверили все 9 004 записи, выполнив 33 881 задачу анализа.
Сильно подозрительных целых записей оказалось восемь: пять с музыкой без
обнаруженной речи, две с другими звуками и одна с тишиной. Ещё 285 — сомнительные,
не доказанный мусор. Ничего не удаляли. Жёсткое исключение и мягкое понижение
веса сравнивали с контролем по трём seed: влияние оказалось небольшим.
Параллельный CPU-анализ этого конвейера был быстрее проверенного гибридного
GPU/CPU-варианта; это не измерение скорости самих моделей.

### Полные модели и длина контекста

При дообучении экспертной HuBERT варьировали скорость обучения, веса потерь,
балансировку, регуляризацию и ограничения верхнего диапазона. Сохранённая
`refine_02` называется `expert-hubert-v1`; поздний выбор с upper-guard привёл
к тем же весам, поэтому дубликат не публикуется. Общая ошибка относительно
эксперта снизилась, но исходная сохранила преимущество при оценках выше 80.

В этапе контекстов и архитектур завершено **26 обучений**: 22 исходных и четыре
дополнительных с длительностями ECAPA. Проверяли полную HuBERT с обучением
последних восьми из двенадцати блоков, WavLM, предобученную ECAPA и контексты
2/3/5/8/10 секунд там, где это было применимо. Оценку использовали для
услышанного отрезка, других TRAIN-окон записи или TRAIN-членов кластера.
Перенос имел отдельный слабый бюджет 0,25. Проверили свежую голову HuBERT.
Длинный контекст всё равно имеет перенесённую оценку пяти секунд, а не новое
реальное прослушивание. Часть дополнительных запланированных запусков не
выполнялась из-за окончания серверного времени.

WavLM не победила. Лучшая supervised ECAPA обучалась на двухсекундных TRAIN-окнах
с переносом на кластер, но **оценивала пять секунд**: оценивание по две секунды
оказалось хуже. Её релизное имя `expert-ecapa-supervised`, TEST MAE 13,66,
верхний диапазон 12,82. HuBERT `expert-hubert-recording`, выбранная на общей
валидационной подвыборке длинных записей, получила 16,00 / 12,52. Цели выбора
различались; это не новый независимый тест.

### Удачная дистилляция

После неудачных уменьшенных HuBERT перешли к **самостоятельным ECAPA-TDNN**, а
не сокращению числа слоёв HuBERT. Сначала восемь опытов с аналитическими
признаками использовали 88 eGeMAPS и 25 временных сводок: питч, гармоники,
спектральные признаки, а не один питч. MLP/CNN имели около 20–23 тысяч параметров.
MLP достигла экспертного TEST MAE 15,97, но среднее отклонение от учителя было
12,98 пункта. Для сохранения поведения учителя этого мало; эти веса не включены
в релиз как успешная замена.

В очереди ECAPA экспертного учителя завершены восемь запусков без ошибок и
отложенных задач. Проверяли предобученную ширину 1 024 (20,79 млн параметров
для inference) и мини шириной 256 с нуля (1,78 млн). Потери включали оценки
учителя, реальные человеческие метки и необязательное сопоставление признаков.
В гибридных вариантах сравнивали баланс человеческой разметки и дистилляции,
вес признаков 0,05. Проекция 192→768 училась только на TRAIN и удалялась при
экспорте. Учитель предоставил 12 745 TRAIN-единиц; реальные оценки учитывались
отдельно от слабых целей.

Сохранены три роли: `expert-ecapa-distilled-human` по человеческой валидации,
`expert-ecapa-distilled-fidelity` по близости учителю и компактная `expert-mini`.
Их экспертные TEST MAE: 14,26 / 13,84 / 15,08. Мини не является точной копией
HuBERT; хорошее совпадение с человеком и точное воспроизведение учителя — разные
цели.

Затем выполнены четыре дистилляции исходного учителя: ECAPA256 только по оценке
или с признаками весом 0,05/0,01 и ECAPA384 с признаками. Человеческой потери
в этом обучении не было. **418 TRAIN-окон `new_voices` исключили из целей
исходного учителя**, оставив 12 327: эти записи специально собирали на его
предполагаемых ошибках. В сравнении шести моделей `original-mini` выбиралась по
зафиксированной VAL-близости учителю среди ECAPA256. Отдельно опубликованы
человеческий выбор эпохи и более широкая ECAPA384; это другие чекпойнты.

### Сравнение шести моделей на одинаковых входах

Обработаны 32 312 уникальных пятисекундных входов из 9 004 записей, включая
18 539 настоящих оценок 43 людей. Экспертный TEST — 198 оценок, из них 32 выше
80; этот TEST уже использовали. Восстанавливали услышанный человеком отрезок:
точный старт, PCM16, моно, padding/resampling, пиковая нормализация каждого окна.
Проверка 36 входов дала побитовое совпадение, разница CPU/GPU была меньше
0,00012 пункта.

| Модель | TEST MAE ↓ | Ошибка при оценке >80 ↓ | Инверсии порядка ↓ | CPU мс / 5 с ↓ |
|---|---:|---:|---:|---:|
| Исходная полная | 21,99 | 10,64 | 23,86% | 181,3 |
| Исходная UINT8 | 22,14 | 10,47 | 24,02% | 115,2 |
| Исходная мини | 19,29 | 9,34 | 19,70% | 15,6 |
| Экспертная полная | 16,19 | 13,01 | 20,47% | 180,7 |
| Экспертная UINT8 | 16,34 | 12,56 | 20,45% | 115,1 |
| Экспертная мини | 15,08 | 12,70 | 18,99% | 15,4 |

Инверсия — обратный порядок двух разных услышанных входов с разными оценками
эксперта. В TEST 18 849 сравнимых пар; 654 пары с одинаковой экспертной оценкой
исключены. Равенство прогнозов учитывается отдельно и даёт половину ошибки
порядка; в TEST таких равенств не было. Это **не** доля записей с неточной
оценкой. В машиночитаемом отчёте есть ограничения по разнице экспертных оценок,
Spearman/Kendall и bootstrap по группам.

Парный групповой bootstrap дал 95% интервалы разницы MAE: исходная мини минус
исходная −2,70 [−4,78; −0,88]; экспертная мини минус экспертная −1,11
[−3,25; +0,95]. Вторая разница пока неубедительна. Скорость измерена на CPU
GPU-хоста: четыре закреплённых ядра, три раунда/60 измерений, batch=1,
frontend включён, сеть и декодирование исключены. Это не задержка бота целиком.

По всем людям MAE: исходная 14,09 / исходная мини 13,96 / экспертная 14,47 /
экспертная мини 14,38. По прежнему all-rater TEST из 2 773 оценок: 17,54 /
17,22 / 16,67 / 16,07. Один человек дал 48% оценок. Равный вес оценщиков
одновременно меняет состав голосов; это проверка чувствительности, а не
объективная общая шкала. Экспертная модель учится шкале одного эксперта.

### Графики, релиз и проверка в боте

[Экспертные графики](docs/figures/2026-10-30/expert-only/README.md) и
[графики всех людей](docs/figures/2026-10-30/all-human/README.md) показывают
знаковые/абсолютные ошибки для всех данных и TEST, а также эмпирические CDF.
Все данные включают TRAIN. В экспертном TEST мини имеет медиану ошибки 9,81,
P90 33,74 и ошибку больше 40 у 7,07% оценок.

В релизе девять выбранных моделей в Safetensors, экспертная UINT8 ONNX и
самостоятельные TorchScript обеих мини. Все тензоры проверены побитово;
у supervised ECAPA убраны неиспользуемые компоненты классификации говорящих.
TorchScript включает нормализацию, Fbank и ECAPA, поэтому для запуска нужен
только PyTorch, без SpeechBrain и загрузки предобученных моделей.

30 октября экспертную мини включили и в боте, и для живой речи миниаппа.
Обе точки используют один worker и одну копию модели. Файл запуска 7,30 МБ,
контекст пять секунд, интервал живых оценок 750 мс; актуальное окно выбирается
после получения места в очереди. Проверка публичного WebSocket вернула 33 мс
inference на основном сервере: это одиночная проверка, не нагрузочный тест.
Старые веса и откат сохранены. Хвосты записи короче трёх секунд по-прежнему
не дополняются для отдельной оценки.

Аудио, приватные имена записей, исходные ID оценщиков, прогнозы отдельных
записей и контейнеры обучения не публикуются. Открытый датасет v0.1.0 остаётся
прежним снимком и не пополняется поздними экспертными оценками молча.
Все частные обучающие очереди здесь полностью не воспроизведены; границы
проверок описаны в [VERIFICATION_V0.2.md](VERIFICATION_V0.2.md). Повторное
использование TEST, неизвестные личности говорящих и историческое пересечение
исходной модели не позволяют обещать независимое улучшение качества.
