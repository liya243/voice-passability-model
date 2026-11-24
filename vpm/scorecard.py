"""Scorecard: the fixed set of checks every model goes through, so models are compared on the same windows.

  gold         MAE to the consensus on the 500 gold segments: overall, 40-90, per 10-point band, worst band;
               Spearman; order accuracy on pairs whose consensus differs by >= 10
  heldout      the same on listened non-gold items of test voices
  consistency  spread of a recording's voiced windows around the recording's mean (humans: ~4.7)
  language     TidyVoice: the same person reading in different languages; how far a language moves the
               speaker's mean score, against the same-language noise floor (two halves of one language)
  cis          median score of cis women / cis men (labels from metadata or cislabel), share of cis women
               below 80 and of cis men above 20

Usage:
  python -m vpm.scorecard build                 -> work/scorecard/windows.csv  (rebuild only when the snapshot changes)
  python -m vpm.scorecard report preds.csv ...  -> work/scorecard/<name>.json and a table for all given models
preds.csv: columns w (window index into windows.csv) and pred (0..100).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import DATA, SR, work
from .audio import window_starts
from .split import recordings
from .voicecheck import load_cache, window_voiced

rng = np.random.default_rng(0)
N_CONSIST = 600          # recordings of our test voices with >= 3 voiced windows
N_LANG_SPK = 400         # TidyVoice test+val speakers with >= 2 languages
N_CIS_SPK = 400          # per gender


def build():
    rec = recordings()
    vc = load_cache()
    items = pd.read_csv(work('target', 'items.csv'))
    rows = []
    by_rel = rec[rec.rel != ''].set_index('rel')
    # gold and held-out items: exact windows that people heard
    for r in items.itertuples():
        if r.rel not in by_rel.index:
            continue
        x = by_rel.loc[r.rel]
        if isinstance(r.gold, str):
            rows.append(dict(set='gold', rec=x.rec, path=x.path, start=r.start, item=r.item))
        elif x.split == 'test' and r.n_listens > 0 and r.novoice_reports == 0 and '/english/' not in r.rel:
            rows.append(dict(set='heldout', rec=x.rec, path=x.path, start=r.start, item=r.item))

    def voiced_windows(row, cap=None):
        bl = vc.get(row.rec)
        if bl is None:
            return []
        st = [s for s in window_starts(row.duration) if window_voiced(bl, s)]
        return st[:cap] if cap else st

    ours = rec[rec.source.isin(['dataset', 'new_voices', 'additional_ru']) & (rec.split == 'test')]
    k = 0
    for r in ours.sample(frac=1, random_state=0).itertuples():
        st = voiced_windows(r, 12)
        if len(st) >= 3:
            rows += [dict(set='consistency', rec=r.rec, path=r.path, start=s) for s in st]; k += 1
        if k >= N_CONSIST:
            break
    tv = rec[(rec.source == 'tidyvoice') & rec.split.isin(['val', 'test'])]
    multi = tv.groupby('speaker').language.nunique()
    spk = multi[multi >= 2].index.to_numpy()
    for s in rng.permutation(spk)[:N_LANG_SPK]:
        for lang, g in tv[tv.speaker == s].groupby('language'):
            for r in g.sample(min(6, len(g)), random_state=0).itertuples():
                rows += [dict(set='language', rec=r.rec, path=r.path, start=st) for st in voiced_windows(r, 1)]
    lab = work('cislabel', 'labels.csv')
    if lab.exists():
        L = pd.read_csv(lab, keep_default_na=False)
        L = L[L.label.isin(['f', 'm'])]
        cis = rec[rec.source.str.startswith('cis_') & rec.split.isin(['val', 'test']) & rec.speaker.isin(L.speaker)]
        for g, Lg in L.groupby('label'):
            for s in rng.permutation(Lg.speaker.to_numpy())[:N_CIS_SPK]:
                c = cis[cis.speaker == s]
                for r in c.sample(min(3, len(c)), random_state=0).itertuples():
                    rows += [dict(set='cis', rec=r.rec, path=r.path, start=st) for st in voiced_windows(r, 2)]
    w = pd.DataFrame(rows)
    w.index.name = 'w'
    w.to_csv(work('scorecard', 'windows.csv'))
    print(w.groupby('set').agg(windows=('rec', 'size'), recordings=('rec', 'nunique')).to_string())


def _band_mae(y, p):
    e = np.abs(p - y); b = np.minimum(y // 10, 9).astype(int)
    bands = {f'{10 * k}-{10 * k + 10}': (float(e[b == k].mean()) if (b == k).sum() >= 5 else None, int((b == k).sum())) for k in range(10)}
    i, j = np.triu_indices(len(y), 1)
    far = np.abs(y[i] - y[j]) >= 10
    order = float((np.sign(p[i] - p[j])[far] == np.sign(y[i] - y[j])[far]).mean())
    mid = (y >= 40) & (y < 90)
    valid = [v for v, n in bands.values() if v is not None]
    return dict(n=len(y), mae=float(e.mean()), mae_40_90=float(e[mid].mean()), worst_band=max(valid),
                spearman=float(spearmanr(y, p).correlation), order_acc=order,
                bands={k: v for k, (v, n) in bands.items()}, band_n={k: n for k, (v, n) in bands.items()})


def report_one(w, p):
    d = w.assign(pred=p)
    items = pd.read_csv(work('target', 'items.csv')).set_index('item')
    out = {}
    for s in ('gold', 'heldout'):
        x = d[d.set == s].dropna(subset=['pred'])
        if len(x):
            out[s] = _band_mae(items.z.reindex(x.item).values, x.pred.values)
    c = d[d.set == 'consistency'].dropna(subset=['pred'])
    if len(c):
        dev = c.pred - c.groupby('rec').pred.transform('mean')
        n = c.groupby('rec').pred.transform('size')
        out['consistency'] = dict(recordings=int(c.rec.nunique()),
                                  within_sd=float(np.sqrt((dev ** 2 * n / (n - 1)).mean())),
                                  max_minus_min=float((c.groupby('rec').pred.max() - c.groupby('rec').pred.min()).median()))
    lg = d[d.set == 'language'].dropna(subset=['pred'])
    if len(lg):
        rec = recordings().set_index('rec')
        lg = lg.assign(speaker=rec.speaker.reindex(lg.rec).values, language=rec.language.reindex(lg.rec).values)
        sl = lg.groupby(['speaker', 'language']).pred.agg(['mean', 'size'])
        gap = sl.groupby(level=0)['mean'].agg(lambda m: m.max() - m.min())
        # noise floor: split each (speaker, language) cell into two random halves of its recordings
        fl = []
        for (_, _), g in lg.groupby(['speaker', 'language']):
            r = g.rec.unique()
            if len(r) >= 4:
                r = rng.permutation(r); h = len(r) // 2
                fl.append(abs(g[g.rec.isin(r[:h])].pred.mean() - g[g.rec.isin(r[h:])].pred.mean()))
        total = lg.pred.var()
        within_spk = lg.groupby('speaker').pred.transform(lambda v: v - v.mean())
        lang_eff = (lg.groupby(['speaker', 'language']).pred.transform('mean') - lg.groupby('speaker').pred.transform('mean'))
        out['language'] = dict(speakers=int(gap.size), median_language_gap=float(gap.median()),
                               same_language_half_gap=float(np.median(fl)) if fl else None,
                               share_of_variance_language=float(lang_eff.var() / total),
                               share_of_variance_speaker=float(1 - within_spk.var() / total))
    cs = d[d.set == 'cis'].dropna(subset=['pred'])
    if len(cs):
        rec = recordings().set_index('rec')
        L = pd.read_csv(work('cislabel', 'labels.csv'), keep_default_na=False).set_index('speaker').label
        cs = cs.assign(label=L.reindex(rec.speaker.reindex(cs.rec).values).values)
        per = cs.groupby('rec').agg(pred=('pred', 'mean'), label=('label', 'first'))
        f, m = per.pred[per.label == 'f'], per.pred[per.label == 'm']
        out['cis'] = dict(n_f=len(f), n_m=len(m), median_f=float(f.median()), median_m=float(m.median()),
                          f_below_80=float((f < 80).mean()), m_above_20=float((m > 20).mean()),
                          f_p10=float(f.quantile(0.1)), m_p90=float(m.quantile(0.9)))
    return out


def table(results):
    rows = []
    for name, r in results.items():
        g, h, c, l, s = (r.get(k, {}) for k in ('gold', 'heldout', 'consistency', 'language', 'cis'))
        rows.append({'model': name, 'gold MAE': g.get('mae'), 'gold 40-90': g.get('mae_40_90'), 'gold worst band': g.get('worst_band'),
                     'gold order': g.get('order_acc'), 'heldout MAE': h.get('mae'), 'heldout worst band': h.get('worst_band'),
                     'window SD': c.get('within_sd'), 'lang gap': l.get('median_language_gap'), 'lang floor': l.get('same_language_half_gap'),
                     'cis f med': s.get('median_f'), 'cis m med': s.get('median_m'), 'cis f<80': s.get('f_below_80')})
    t = pd.DataFrame(rows).set_index('model')
    bands = pd.DataFrame({n: r['gold']['bands'] for n, r in results.items() if 'gold' in r}).T
    return t, bands


def report(paths):
    w = pd.read_csv(work('scorecard', 'windows.csv'), index_col='w')
    res = {}
    for f in paths:
        p = pd.read_csv(f).set_index('w').pred.reindex(w.index).values
        res[Path(f).stem] = r = report_one(w, p)
        json.dump(r, open(work('scorecard', Path(f).stem + '.json'), 'w'), indent=1)
    t, b = table(res)
    pd.set_option('display.width', 250)
    print(t.round(3).to_string()); print('\ngold MAE by consensus band'); print(b.round(1).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('preds', nargs='*')
    a = ap.parse_args()
    build() if a.cmd == 'build' else report(a.preds)
