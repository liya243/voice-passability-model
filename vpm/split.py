"""Split by voice, never by file: the same person must not be on both sides.

  test   every voice that has a gold segment, plus 10 % of the other voices (fixed by hash)
  val    another 10 % of voices; model selection and early stopping only
  train  the rest

The split depends only on the speaker id, so adding recordings never moves an existing voice.
"""
import hashlib
import pandas as pd
from .config import DATA, work
import json


def bucket(speaker):
    return int(hashlib.sha1(('split-v1/' + speaker).encode()).hexdigest(), 16) % 10


def assign(rec):
    gold = {g['relative_path'] for g in json.load(open(DATA / 'sys_new/gold_v1_catalog.json'))['items']}
    gold_spk = set(rec.speaker[rec.rel.isin(gold)])
    b = rec.speaker.map(bucket)
    return pd.Series(['test' if s in gold_spk or k == 1 else ('val' if k == 0 else 'train')
                      for s, k in zip(rec.speaker, b)], index=rec.index)


def recordings():
    rec = pd.read_csv(work('snapshot', 'recordings.csv'), keep_default_na=False)
    rec['split'] = assign(rec)
    return rec
