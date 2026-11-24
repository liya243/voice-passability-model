"""Gender labels for cis recordings from two independent binary classifiers.

  ecapa   JaesungHuh/voice-gender-classifier (ECAPA-TDNN, VoxCeleb2, MIT)
  audeer  audeering/wav2vec2-large-robust-24-ft-age-gender (CommonVoice, VoxCeleb2, aGender, TIMIT; CC-BY-NC-SA)

Each model sees up to MAX_WIN voiced 5 s windows of a recording; probabilities are averaged per recording
and then per speaker. A label is given only when both models agree with confidence >= the threshold;
the threshold is set by `validate` on corpora with metadata so that the lowest-voiced 10 % of women and
the highest-voiced 10 % of men are labelled with <= 1 % error. Trans voices are never labelled this way.

Usage:
  python -m vpm.cislabel score  [--source cis_tg]   -> work/cislabel/scores.csv   (p_female per model per recording)
  python -m vpm.cislabel validate                    -> threshold from cis corpora with gender_meta
  python -m vpm.cislabel label                       -> work/cislabel/labels.csv  (speaker, label)
"""
import argparse, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .config import DATA, SR, work
from .audio import load, window_starts, cut
from .voicecheck import load_cache, window_voiced

MAX_WIN = 6
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class _Audeer:
    name = 'audeer'

    def __init__(self):
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class Head(nn.Module):
            def __init__(self, c, n):
                super().__init__(); self.dense = nn.Linear(c.hidden_size, c.hidden_size)
                self.dropout = nn.Dropout(c.final_dropout); self.out_proj = nn.Linear(c.hidden_size, n)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class AgeGender(Wav2Vec2PreTrainedModel):     # as published in the model card
            def __init__(self, c):
                super().__init__(c); self.config = c; self.wav2vec2 = Wav2Vec2Model(c)
                self.age = Head(c, 1); self.gender = Head(c, 3); self.post_init()

            def forward(self, x):
                h = self.wav2vec2(x)[0].mean(dim=1)
                return self.age(h), torch.softmax(self.gender(h), dim=1)

        n = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
        self.proc = Wav2Vec2Processor.from_pretrained(n)
        self.m, info = AgeGender.from_pretrained(n, output_loading_info=True)
        assert not info['missing_keys'], info['missing_keys']
        self.m = self.m.to(dev).eval()

    @torch.no_grad()
    def p_female(self, W):
        x = self.proc(list(W), sampling_rate=SR, return_tensors='pt').input_values.to(dev)
        g = self.m(x)[1].float().cpu().numpy()          # female, male, child
        return g[:, 0] / np.maximum(g[:, 0] + g[:, 1], 1e-6)


class _Ecapa:
    name = 'ecapa'

    def __init__(self):
        from .third_party.jh_gender import ECAPA_gender
        self.m = ECAPA_gender.from_pretrained('JaesungHuh/voice-gender-classifier').to(dev).eval()

    @torch.no_grad()
    def p_female(self, W):
        x = torch.from_numpy(np.stack(W)).to(dev)
        lo = self.m.forward(x)                           # classes: ['male', 'female']
        return torch.softmax(lo, dim=1)[:, 1].float().cpu().numpy()


def windows(path, bl):
    x = load(DATA / path)
    st = [s for s in window_starts(len(x) / SR) if bl is None or window_voiced(bl, s)][:MAX_WIN]
    return [cut(x, s) for s in st]


def f0_median(W):
    import parselmouth
    f = parselmouth.Sound(np.concatenate(W).astype(np.float64), SR).to_pitch(pitch_floor=60, pitch_ceiling=500).selected_array['frequency']
    f = f[f > 0]
    return float(np.median(f)) if len(f) else np.nan


def score(source):
    rec = pd.read_csv(work('snapshot', 'recordings.csv'), keep_default_na=False)
    rec = rec[rec.source.str.startswith('cis_')]
    if source:
        rec = rec[rec.source == source]
    vc = load_cache()
    models = [_Ecapa(), _Audeer()]
    out = work('cislabel', 'scores.csv')
    done = set(pd.read_csv(out).rec) if out.exists() else set()
    rows = []
    for k, r in enumerate(rec.itertuples()):
        if r.rec in done:
            continue
        try:
            W = windows(r.path, vc.get(r.rec))
        except Exception:
            continue
        if not W:
            continue
        rows.append(dict(rec=r.rec, n_win=len(W), f0=f0_median(W), **{m.name: float(np.mean(m.p_female(W))) for m in models}))
        if len(rows) == 500:
            pd.DataFrame(rows).to_csv(out, mode='a', header=not out.exists(), index=False); rows = []
            print(k + 1, '/', len(rec), flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode='a', header=not out.exists(), index=False)


def per_speaker():
    rec = pd.read_csv(work('snapshot', 'recordings.csv'), keep_default_na=False)
    s = pd.read_csv(work('cislabel', 'scores.csv')).merge(rec[['rec', 'source', 'speaker', 'gender_meta']], on='rec')
    return s.groupby(['source', 'speaker']).agg(ecapa=('ecapa', 'mean'), audeer=('audeer', 'mean'), f0=('f0', 'median'), n=('rec', 'size'),
                                                gender_meta=('gender_meta', 'first')).reset_index()


def decide(sp, thr):
    f = (sp.ecapa >= thr) & (sp.audeer >= thr)
    m = (sp.ecapa <= 1 - thr) & (sp.audeer <= 1 - thr)
    return np.where(f, 'f', np.where(m, 'm', ''))


def validate():
    """Threshold from corpora with metadata, judged on the hard tails: lowest 10 % of women and highest 10 % of men by median F0."""
    sp = per_speaker()
    sp = sp[sp.gender_meta.isin(['f', 'm'])]
    if sp.empty:
        sys.exit('no speakers with gender_meta: add cis/<corpus>/speakers.csv for an official corpus first')
    f0 = sp.set_index('speaker').f0
    res = []
    for thr in np.arange(0.5, 0.99, 0.02):
        lab = decide(sp, thr)
        v = sp.speaker.map(f0)
        lo_f = (sp.gender_meta == 'f') & (v <= v[sp.gender_meta == 'f'].quantile(0.1))
        hi_m = (sp.gender_meta == 'm') & (v >= v[sp.gender_meta == 'm'].quantile(0.9))
        hard = (lo_f | hi_m).values
        given = lab != ''
        err = (lab != sp.gender_meta.values) & given
        res.append(dict(thr=round(thr, 2), coverage=given.mean(), error=err[given].mean() if given.any() else np.nan,
                        hard_coverage=given[hard].mean(), hard_error=err[hard & given].sum() / max(1, (hard & given).sum())))
    res = pd.DataFrame(res)
    print(res.round(3).to_string(index=False))
    ok = res[(res.hard_error <= 0.01) & (res.error <= 0.01)]
    thr = float(ok.thr.min()) if len(ok) else None
    json.dump(dict(threshold=thr, table=res.to_dict('records')), open(work('cislabel', 'threshold.json'), 'w'), indent=1)
    print('threshold', thr)


def label():
    thr = json.load(open(work('cislabel', 'threshold.json')))['threshold']
    if thr is None:
        sys.exit('validation found no threshold with <= 1 % error; do not label')
    sp = per_speaker()
    sp['label'] = decide(sp, thr)
    sp.loc[sp.gender_meta.isin(['f', 'm']), 'label'] = sp.gender_meta   # metadata wins where it exists
    sp.to_csv(work('cislabel', 'labels.csv'), index=False)
    print(sp.groupby(['source', 'label']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--source')
    a = ap.parse_args()
    {'score': lambda: score(a.source), 'validate': validate, 'label': label}[a.cmd]()
