# Gold set, expert-anchored fine-tuning and voice weight v2 / Эталон, дообучение на шкале экспертки и вес голоса v2

[English](#english) · [Русский](#russian)

<a id="english"></a>

## English

November 21, 2026. Assets: [v0.3.0](https://github.com/liya243/voice-passability-model/releases/tag/v0.3.0).
No audio is published; all people, recordings, clips and voice groups use random ids.

### Gold set v1

500 fixed five-second segments, one per voice cluster, rated in a dedicated miniapp mode:

- 4,482 gold-mode ratings plus 1,003 earlier ratings of the same five seconds (overlap of at least 2.5 s);
- median 5 listeners per segment, at least 5 on 440 of 500; the expert rated 498;
- most gold-mode ratings come from four listeners and the expert, so the gold set reflects a small, consistent group;
- split-half reliability of the listener mean 0.88 (0.94 for the full count).

### Expert-anchored consensus

The target used below, `consensus_percent`, puts every rater on the expert's scale:

- each rater has a monotone piecewise-linear curve (knots every 10 points), the expert's curve is the identity, other curves shrink toward a pooled listener curve;
- a rater's weight is set by disagreement with the expert on shared segments; the expert's own noise comes from her repeats (±8.2 points);
- the best listeners disagree with her by ±13–15 points, the rest by ±20–26; legacy weak labels count as one more rater;
- the latent value of each segment is found on a 0–100 grid; `consensus_sd` is its posterior spread.

The curves are in `rater_scales.csv`.

### Baseline on the gold set

The original model ranks voices well (r = 0.83) but scores them about **14 points higher** than listeners do. An isotonic recalibration fitted outside gold voices removes the shift but compresses the top of the scale.

### Fine-tuning (ft1–ft7)

- Start: the original HuBERT, not the expert version.
- Labels: the consensus of 8,090 segments outside gold voices, weighted by 1 / (consensus_sd² + 16).
- Excluded: every ECAPA voice cluster that contains a gold segment, and all English recordings.
- Retention: on unlabelled Russian windows, predictions stay close to the recalibrated original model.
- Early stopping on 10% of voice groups that are outside the gold set.

| Run | Unfrozen layers | Retention weight | Learning rate | Note |
|---|---|---|---|---|
| ft1 | 6 | 0.5 | 1e-5 | |
| ft2 | 6 | 0.1 | 2e-5 | |
| ft3 | 12 | 0.5 | 1e-5 | **released, in the bot** |
| ft4 | 6 | 1.0 | 5e-6 | |
| ft5 | 12 | 0.1 | 2e-5 | |
| ft6 | 12 | 0 | 1e-5 | |
| ft7 | 12 | 0.1 | 2e-5 | weight ×2 for consensus ≥ 70; released as a fallback |

**How ft3 was chosen.** Most of the bot's users sit between 40 and 90, so the model was chosen by MAE in that range on held-out voices outside the gold set: 7.9 against 14.8 for the original. The gold set was then checked once. Several runs had been compared on gold before this rule was set, so differences between runs (about 0.5 points) carry some selection optimism. The gap to the original model does not.

**Gold set, MAE against the consensus by 10-point band** (496 segments):

| Model | All | 40–90 | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | 90–100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 16.9 | 13.6 | 21.7 | 21.5 | 20.7 | 28.0 | 17.6 | 17.0 | 13.2 | 12.9 | **6.4** | **7.7** |
| ft3 | **10.7** | **8.5** | **12.7** | **9.2** | **10.6** | **18.0** | **11.7** | **8.7** | **6.2** | **7.7** | 10.4 | 16.4 |
| Segments | 496 | 283 | 119 | 15 | 17 | 41 | 34 | 74 | 78 | 51 | 46 | 21 |

- Errors above 25 points: 24% → 6%. The bootstrap interval over voices is clearly below zero.
- Correlation with the expert alone: 0.78 → 0.87. Against the plain listener mean, MAE is 15.3 → 8.4; for comparison, one listener differs from the mean of the others by 10.9.
- Largest gains: hidden technique errors 22.5 → 12.0, low pitch 24.6 → 14.4, user complaints 18.7 → 11.5, English probe 17.5 → 9.5.
- **Weakness at the top.** At 80–100 ft3 scores lower than the consensus, and held-out voices outside the gold set show the same. ft7 is closer to the original there (gold MAE at ≥ 70: 7.8 against 9.4).
- **U8U8 ONNX.** The compressed ft3 differs from the full model by 0.56 points on average (p95 1.5) and takes 0.67 s per window on the bot's ARM CPU, like the original.

The per-segment predictions of every model are in `gold_segments.csv.gz`.

### Voice weight v2

807 pairwise answers to "which voice is heavier", 804 of them by the expert; 401 were "about the same". Every one of the 1,389 clips heard was rebuilt byte for byte (its WAV SHA256 matched).

- Model: Davidson ranking (ties allowed) on 15 Praat features of a 0.5 s clip.
- Validation: 5 folds × 10 splits by voice (author for curated recordings, ECAPA cluster otherwise). In a test pair both voices are unseen; about 160 decisive pairs per split.

| Features | Agreement with the expert's choice | On pairs with dataset voices |
|---|---|---|
| **Acoustic, 15 features (released)** | **90%** | **90%** |
| Acoustic + WavLM layer 3 | 89% | 89% |
| Pitch only | 83% | 82% |
| WavLM layer 3 | 82% | 76% |
| Old bot model | 78% | 76% |
| Acoustic without pitch | 75% | 73% |

- **Expert self-consistency.** Repeats show a pair in either order, so they are compared by the chosen voice. She gives the same answer on 83% of 53 repeats; when both answers pick a side, the side is the same on 92% (12 repeats). An earlier 78% figure compared the letters A/B and was wrong.
- **Ties.** On "about the same" pairs the prediction gap is half that of decisive pairs: 0.63 against 1.31 SD.
- **Learning curve.** 25 decisive pairs give 83%, 100 give 88%, 200 give 92%, and the curve flattens after that.
- **Pitch.** It is included: without it the model falls below the old one.
- **Percent scale.** A recording's percent is the position of its median 0.5 s score among 1,998 random dataset recordings from 1,109 voices, each voice counted once. Curated "heavy" recordings get a median of 75%, "light" ones 28%.
- **Limitations.** There is one rater, no anchor scale yet, and only about 34 test pairs with dataset voices.

Code and bundle: `weight/`.

<a id="russian"></a>

## Русский

21 ноября 2026. Файлы: [v0.3.0](https://github.com/liya243/voice-passability-model/releases/tag/v0.3.0).
Аудио не публикуется; у людей, записей, кусков и групп голоса — случайные идентификаторы.

### Эталон v1

500 фиксированных пятисекундных отрезков, по одному на кластер голоса. Их оценивали в отдельном режиме миниаппа:

- 4 482 оценки в режиме эталона и 1 003 более ранние оценки тех же пяти секунд (перекрытие не меньше 2.5 с);
- медиана — 5 слушателей на отрезок, не меньше 5 — на 440 из 500; экспертка оценила 498;
- почти все оценки эталона поставили четыре слушателя и экспертка, так что эталон отражает небольшую согласованную группу;
- надёжность среднего слушателей по половинам 0.88, для полного числа слушателей 0.94.

### Консенсус на шкале экспертки

Цель ниже, `consensus_percent`, переводит каждого оценщика на шкалу экспертки:

- у каждого оценщика своя монотонная кусочно-линейная кривая (узлы через 10 пунктов); у экспертки это тождество, у остальных кривая сжимается к общей кривой слушателей;
- вес оценщика задаёт его расхождение с эксперткой на общих отрезках; собственный шум экспертки — по её повторам (±8.2 пункта);
- лучшие слушатели расходятся с ней на ±13–15 пунктов, остальные на ±20–26; старые слабые метки считаются ещё одним оценщиком;
- значение отрезка ищется на сетке 0–100; `consensus_sd` — его апостериорный разброс.

Кривые лежат в `rater_scales.csv`.

### Исходная модель на эталоне

Исходная модель хорошо упорядочивает голоса (r = 0.83), но ставит в среднем **на 14 пунктов выше** слушателей. Изотоническая перекалибровка, подобранная вне голосов эталона, убирает сдвиг, но сжимает верх шкалы.

### Дообучение (ft1–ft7)

- Старт: исходная HuBERT, не экспертная.
- Метки: консенсус 8 090 отрезков вне голосов эталона с весом 1 / (consensus_sd² + 16).
- Исключены: все кластеры голоса ECAPA, где есть отрезок эталона, и все английские записи.
- Удержание: на неразмеченных русских окнах прогноз держится близко к перекалиброванной исходной модели.
- Ранняя остановка по 10% групп голоса вне эталона.

| Вариант | Размороженных слоёв | Вес удержания | Шаг обучения | Примечание |
|---|---|---|---|---|
| ft1 | 6 | 0.5 | 1e-5 | |
| ft2 | 6 | 0.1 | 2e-5 | |
| ft3 | 12 | 0.5 | 1e-5 | **опубликован, работает в боте** |
| ft4 | 6 | 1.0 | 5e-6 | |
| ft5 | 12 | 0.1 | 2e-5 | |
| ft6 | 12 | 0 | 1e-5 | |
| ft7 | 12 | 0.1 | 2e-5 | вес ×2 у консенсуса ≥ 70; опубликован как запасной |

**Как выбран ft3.** Большинство пользователей бота находится в диапазоне 40–90, поэтому модель выбирали по MAE в этом диапазоне на отложенных голосах вне эталона: 7.9 против 14.8 у исходной. Эталон после этого проверен один раз. Несколько вариантов сравнивались на эталоне до того, как правило было принято, поэтому разница между вариантами (около 0.5 пункта) немного оптимистична. На разрыв с исходной моделью это не влияет.

**Эталон, MAE против консенсуса по полосам в 10 пунктов** (496 отрезков):

| Модель | Все | 40–90 | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | 90–100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Исходная | 16.9 | 13.6 | 21.7 | 21.5 | 20.7 | 28.0 | 17.6 | 17.0 | 13.2 | 12.9 | **6.4** | **7.7** |
| ft3 | **10.7** | **8.5** | **12.7** | **9.2** | **10.6** | **18.0** | **11.7** | **8.7** | **6.2** | **7.7** | 10.4 | 16.4 |
| Отрезков | 496 | 283 | 119 | 15 | 17 | 41 | 34 | 74 | 78 | 51 | 46 | 21 |

- Ошибок больше 25 пунктов: 24% → 6%. Бутстреп-интервал по голосам надёжно ниже нуля.
- Корреляция с одной эксперткой: 0.78 → 0.87. Против простого среднего слушателей MAE 15.3 → 8.4; для сравнения, один слушатель отличается от среднего остальных на 10.9.
- Больше всего выиграли срезы: скрытые ошибки техники 22.5 → 12.0, низкий тон 24.6 → 14.4, жалобы 18.7 → 11.5, английские 17.5 → 9.5.
- **Слабое место — верх шкалы.** На 80–100 ft3 занижает относительно консенсуса, и на отложенных голосах вне эталона видно то же. ft7 там ближе к исходной (MAE на эталоне при ≥ 70: 7.8 против 9.4).
- **U8U8 ONNX.** Сжатая ft3 отличается от полной в среднем на 0.56 пункта (p95 1.5) и считает окно за 0.67 с на ARM-процессоре бота, как исходная.

Прогнозы всех моделей по каждому отрезку — в `gold_segments.csv.gz`.

### Вес голоса v2

807 ответов на вопрос «какой голос тяжелее», 804 из них от экспертки; 401 — «примерно одинаково». Все 1 389 прослушанных кусков восстановлены байт в байт (SHA256 WAV совпал).

- Модель: ранжирование Дэвидсона (с ничьими) на 15 признаках Praat полусекундного куска.
- Проверка: 5 частей × 10 разбиений по голосам (по автору для отобранных записей, по кластеру ECAPA для остальных). В тестовой паре оба голоса невиданные; около 160 решающих пар на разбиение.

| Признаки | Совпадение с выбором экспертки | На парах с голосами датасета |
|---|---|---|
| **Акустика, 15 признаков (опубликована)** | **90%** | **90%** |
| Акустика + WavLM слой 3 | 89% | 89% |
| Только высота тона | 83% | 82% |
| WavLM слой 3 | 82% | 76% |
| Старая модель бота | 78% | 76% |
| Акустика без высоты тона | 75% | 73% |

- **Самосогласованность экспертки.** На повторах пара показывается в любом порядке, поэтому повторы сравниваются по выбранному голосу. Тот же ответ — в 83% из 53 повторов; если оба раза выбрана сторона, то та же в 92% (12 повторов). Прежние 78% сравнивали буквы A/B и были неверны.
- **Ничьи.** На парах «примерно одинаково» разница прогнозов вдвое меньше, чем на парах с выбором: 0.63 против 1.31 SD.
- **Кривая обучения.** 25 решающих пар дают 83%, 100 — 88%, 200 — 92%, дальше кривая почти не растёт.
- **Высота тона.** Включена: без неё модель хуже старой.
- **Шкала процентов.** Процент записи — место медианы её полусекундных оценок среди 1 998 случайных записей датасета от 1 109 голосов, каждый голос учитывается один раз. Отобранные «тяжёлые» записи получают медиану 75%, «лёгкие» — 28%.
- **Ограничения.** Оценщик один, шкалы по опорам пока нет, пар с голосами датасета в тесте всего около 34.

Код и модель: `weight/`.
