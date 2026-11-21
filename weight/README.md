# Voice weight v2 / Вес голоса v2

[English](#english) · [Русский](#russian)

<a id="english"></a>

## English

The model behind `/weight` in the bot and the live meter in the miniapp since November 21, 2026. It is a Davidson ranking model (ties allowed) trained on 807 pairwise "which voice is heavier" answers. It scores 15 Praat features of every voiced 0.5 s block. On voices unseen in training it agrees with the expert's choice in 90% of decisive pairs; the old bot model agrees in 78%. Details are in [EXPERIMENTS_2026-11-21.md](../EXPERIMENTS_2026-11-21.md#english).

```bash
pip install numpy scipy soundfile praat-parselmouth
python weight/predict_weight.py recording.wav
```

- **Output.** `percent` is the position of the recording's median block score among dataset voices: 50 is average weight, 90 is heavier than 90% of voices. `percent_p25`–`percent_p75` is the spread across blocks.
- **Blocks.** Each block is preprocessed like the miniapp clips: mean removed, RMS 0.12 capped at peak 0.95, 3 ms fades. A block counts if it has at least 5 pitched frames and a voiced fraction of at least 0.33.
- **Bundle.** `weight_v2_20261121.json` is plain JSON (no pickle): feature means and SDs, fill values, weights and the percent table.
- **What the percent means.** It is a learned perception rating relative to this dataset, not a physical measure. Pitch is included and is the strongest feature.

<a id="russian"></a>

## Русский

Модель, которая с 21 ноября 2026 считает `/weight` в боте и живую шкалу веса в миниаппе. Это модель ранжирования Дэвидсона (с ничьими), обученная на 807 ответах «какой голос тяжелее». Она оценивает 15 признаков Praat каждого полусекундного куска с голосом. На голосах, которых не было в обучении, модель совпадает с выбором экспертки в 90% пар, где выбрана сторона; старая модель бота — в 78%. Подробности — в [EXPERIMENTS_2026-11-21.md](../EXPERIMENTS_2026-11-21.md#russian).

```bash
pip install numpy scipy soundfile praat-parselmouth
python weight/predict_weight.py recording.wav
```

- **Результат.** `percent` — место медианы записи среди голосов датасета: 50 — средний вес, 90 — тяжелее 90% голосов. `percent_p25`–`percent_p75` — разброс по кускам.
- **Куски.** Каждый кусок обрабатывается так же, как в миниаппе: среднее убирается, RMS приводится к 0.12 с ограничением пика 0.95, по краям затухание 3 мс. Кусок засчитывается, если в нём не меньше 5 кадров с тоном и доля озвученного не меньше 0.33.
- **Модель.** `weight_v2_20261121.json` — обычный JSON (без pickle): средние и SD признаков, значения для пропусков, веса и таблица процентов.
- **Что значит процент.** Это выученная оценка восприятия относительно этого датасета, а не физическая величина. Высота тона включена и даёт самый сильный вклад.
