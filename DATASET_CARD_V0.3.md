# Gold ratings, consensus labels and voice-weight pairs — v0.3.0

Archive: `gold-and-weight-v0.3.0.zip` in the [v0.3.0 release](https://github.com/liya243/voice-passability-model/releases/tag/v0.3.0).
No audio, transcripts, file names, Telegram ids or timestamps are included. Rows are shuffled.

`rater_id` and `record_id` are the v0.1.0 pseudonyms wherever the person or recording was already in v0.1.0. Later raters and recordings get new random ids. `gold_id`, `clip_id`, `voice_group` and `vote_id` are random and exist only in this release. The rater with `is_expert = True` is the expert who also appears in v0.1.0.

## Files and fields

`gold_segments.csv.gz` has one row per gold v1 segment (500 rows):

- `gold_id`, `record_id`, `source_kind`: `dataset`, `new_voices`, `additional_data_ru` or `additional_data_en`.
- `segment_start_seconds`, `segment_duration_seconds` (5.0): the exact audio heard.
- `slice`: the selection stratum, for example `hidden technique errors (expert >= 25 below others)`.
- `consensus_percent`, `consensus_sd`: the expert-anchored consensus and its posterior spread (see below).
- `listener_mean_percent`, `listener_count`: the plain mean of per-person means, expert excluded.
- `expert_percent`: the expert's mean score, blank on 2 segments.
- `pred_original_fp32`, `pred_original_uint8`, `pred_expert_mini`, `pred_original_mini`, `pred_ft3`, `pred_ft7`: model scores of exactly this window, peak-normalized per window.

`gold_ratings.csv.gz` (5,485 rows):

- `gold_id`, `rater_id`, `is_expert`, `score_percent`.
- `presentation_no`: 1–4 for gold-mode ratings (repeats of one segment by one person), 0 otherwise.
- `origin`: `gold` (gold mode, 4,482 rows), `earlier_individual` or `earlier_cluster`, which are earlier ratings of the same five seconds with at least 2.5 s overlap. The earlier ones are also in v0.1.0; do not count them twice.

`consensus_labels.csv.gz` covers every listened five-second item plus legacy-only dataset files (10,415 rows):

- `record_id`, `source_kind`, `segment_start_seconds`, `segment_duration_seconds`.
- `consensus_percent`, `consensus_sd`, `listen_count` (genuine listens), `has_expert`.
- `gold_id`: set on gold segments; train on the others only.
- `no_voice_reports`: how many times the item was reported as "not a voice".
- Items with `listen_count = 0` rest on the legacy weak label alone, and their `consensus_sd` is large.

`rater_scales.csv` holds one row per rater, including the legacy weak labels:

- `votes`, `noise_sd`: disagreement with the expert, in points.
- `retest_sd`: spread on gold repeats.
- `h(0)`…`h(100)`: the rater's answer curve on the expert scale, every 20 points.

`weight_clips.csv.gz` has one row per 0.5 s clip heard in weight pairs (1,389 rows):

- `clip_id`, `record_id`, `voice_group`: the curated author or the ECAPA voice cluster, used for grouped validation.
- `source_kind`, `folder_label` (`heavy` / `light` for curated recordings), `start_seconds`, `duration_seconds`.
- The 15 Praat features used by weight v2: `f0_median` … `centroid_hz`.
- `old_bot_model_score`, `weight_v2_score`, `weight_v2_percent`.

`weight_clip_wavlm.npz` (load with `allow_pickle=False`): `clip_ids` aligned with `wavlm` float16 `(1389, 4, 768)`, the time-mean of `microsoft/wavlm-base-plus` layers 3, 6, 9 and 12.

`weight_pairs.csv.gz` (807 rows):

- `vote_id`, `rater_id`, `is_expert`, `clip_a`, `clip_b`.
- `choice`: `a` means the first clip is heavier, `b` the second, `tie` "about the same".
- `pair_kind`, `presentation_no`, `repeat_of`: the `vote_id` of the first presentation. A repeat may show the pair in the other order, so compare repeats by the chosen clip.

## The expert-anchored consensus

Each rater answers `s = h_r(z) + noise`:

- `h_r` is monotone with knots every 10 points; the expert's `h` is the identity, and other curves shrink toward a pooled listener curve;
- a rater's noise is the disagreement with the expert on shared items, minus the expert's own repeat noise (8.2 points);
- `z` is solved on a 0–100 grid with a weak N(50, 40²) prior.

The fit alternates between curves and `z` until it converges. Code is described in [EXPERIMENTS_2026-11-21.md](EXPERIMENTS_2026-11-21.md#english).

## Valid comparisons

- **Gold segments are a test set.** Leave out every voice that has a gold segment; the gold voices are ECAPA clusters that cannot be rebuilt from public data, so excluding by `record_id` is a minimum.
- **Do not tune on the gold set.**
- **Weight pairs.** Split by `voice_group` so that both voices of a test pair are unseen.

## Privacy and license

Pseudonymized does not mean anonymous: the voice features can still be compared with other recordings of the same speaker. Do not try to re-identify people. License as in [LICENSE_DATA.md](LICENSE_DATA.md).
