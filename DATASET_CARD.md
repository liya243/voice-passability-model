# ECAPA features and human ratings — v0.1.0

## Scope

9,004 recording-level 192-dimensional embeddings, 1,352 current clusters,
43 pseudonymized raters, 8,582 genuine individual listening ratings,
646 genuine cluster ratings, 159 separately marked adopted cluster means,
and 8,757 historical weak labels. The release is a fixed snapshot; online
collection continues separately. See `dataset_manifest.json` for counts.

Embeddings use [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
Audio was mono 16 kHz, split into 3.5 s chunks, keeping final tails of at
least 1 s; short files use one whole-file chunk. Each chunk is peak-normalized,
then its embedding is L2-normalized. The equal-chunk mean is L2-normalized
again. No audio or text transcripts are included.

## Files and fields

`ecapa_features.npz` (load with `allow_pickle=False`):

- `record_ids`: release-local random record identifiers, aligned with rows.
- `embeddings`: float32 matrix `(9004, 192)` of unit vectors.

`records.csv.gz`:

- `record_id`, `cluster_id`: random identifiers, not original filenames/IDs.
- `added_revision`: when the recording joined the current cluster family.
- `split`: `train`, `validation`, `test`; `acoustic_split_group` identifies
  the acoustic block used to avoid splitting closely related recordings.
- `challenge_set`: curated recordings originally collected where the old
  model could be wrong. Old-model predictions are not a trusted target here.
- `individual_mean_percent`: first average repeats/segments per recording and
  rater, then average raters equally. Blank means no genuine individual rating.
- `individual_rater_count`, `individual_vote_count`, `individual_rater_std_percent`.
- `weak_legacy_score_percent`: administrator-assigned old-cluster label copied
  to this recording. Blank where unavailable. It is not an independent
  individual listening judgment and is not the default baseline target.

`individual_ratings.csv.gz`:

- `record_id`, `rater_id`, `score_percent`.
- `segment_start_seconds`, `segment_duration_seconds`: the relative audio
  segment heard by the rater. A file embedding covers the whole recording,
  so segment judgments and whole-recording features need not match perfectly.
- Each row is a real recorded judgment. Repeated judgments are retained.
  Source database IDs and timestamps are omitted; rows are shuffled.

`cluster_ratings.csv.gz`:

- `cluster_id`, `rater_id`, `score_percent`, `cluster_revision`.
- `heard_record_id`, `segment_start_seconds`, `segment_duration_seconds`
  retain which example was actually heard, using only public random IDs.
- These are genuine cluster-mode judgments. They must not be counted once
  per member as independent listening events.

`adopted_cluster_ratings.csv.gz`:

- Same cluster/rater/score/revision fields, plus source file/vote counts.
- `origin=derived_individual_cluster_mean`: explicitly adopted historical
  individual means. They reuse source individual judgments. Do not add them
  as independent additional votes when those individual votes are already used.

The same random six-digit `rater_id` means the same person across all tables.
This preserves the information needed for rater-specific calibration and
order-agreement research. The real-to-random mapping remains private.

## Cluster revisions

The initial clustering used cosine complete-link agglomeration at distance
0.5. Later recordings are inserted sequentially into the compatible group
with minimum maximum distance to every member, or form a new group.
Existing groups/IDs remain stable. Membership is append-only, not a fresh
global clustering after every new file.

For a cluster judgment at revision `r`, eligible members are rows in
`records.csv.gz` with that `cluster_id` and `added_revision <= r`.
The same rule reconstructs the initially adopted memberships. A new member
is not evidence that its voice was personally heard in an earlier judgment.

## Splits and valid comparisons

The deterministic 70/15/15 split keeps the previously computed cosine
complete-link 0.8 groups together. New members inherit their existing
cluster's block; entirely new groups get their own block. Label values
do not determine the split. Counts: 6,302 / 1,351 / 1,351 recordings.
Public block IDs are random aliases.

Exact speaker identities are unavailable; acoustically grouped does not
guarantee speaker-disjoint. Historical weak-label groups also differ from
the current geometry. Results using weak labels require their own leakage
analysis and should not be compared to genuine-human targets as equivalent.

The provided Ridge baseline fits the mean genuine individual rating. It
selects regularization using validation only and reports test once.
Keep repeats, rater dependence, cluster propagation and target-scale changes
in mind when evaluating other approaches. ECAPA vectors support fixed-feature
models; full HuBERT fine-tuning requires separate waveforms.

## Privacy

This release was prepared at the maintainer's request using features and
ratings, with explicit exclusion of source audio. Real Telegram IDs, original
filenames, source paths, file hashes, recording/annotation timestamps,
messages and the identifier mapping are excluded. Only relative segment
positions are retained. Random IDs were not computed as hashes of Telegram IDs.

**The data is pseudonymized, not guaranteed anonymous.** ECAPA is specifically
a speaker-verification/identification representation. A party with other
recordings of a person may be able to link embeddings. Consistent rater
aliases retain a person's rating profile. Removal of audio and identifiers
does not establish a differential-privacy or formal anonymity guarantee.

For context, see the [encoder's intended use](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
and [research on inversion of speaker representations](https://www.isca-archive.org/spsc_2022/pizzi22_spsc.pdf).
That research does not establish exact waveform reconstruction for every
embedding in this release. Do not attempt re-identification or publish
private identifying information in project discussions.

## License

Feature and rating tables are released by the maintainer under CC BY 4.0;
attribute `liya243 / voice-passability-model`, the release version and URL.
This notice covers the distributed features/tables and conveys no rights
to the original audio. The ECAPA encoder is a separate Apache-2.0 project.
