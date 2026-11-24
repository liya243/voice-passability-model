"""Which window length suits a backbone: frozen-feature probe at several lengths, centred on the rated 5 s.

People rated 5 s segments; a longer window adds context from the same recording around them (clipped at the
recording's edges), a shorter one keeps its middle. Same train/val voices and ridge as vpm.probe; reported on
val, gold and held-out test items, overall and by band.

Usage: python -m vpm.winlen whisper_small --lengths 3,5,8,12,20  ->  work/winlen/<backbone>.csv
"""
import argparse
import numpy as np
import pandas as pd
from .config import DATA, SR, WIN, work
from .audio import load, cut
from .split import recordings
from .probe import ridge_fit, ALPHAS
from . import backbones


def items():
    rec = recordings()
    by = rec[rec.rel != ''].set_index('rel')
    it = pd.read_csv(work('target', 'items.csv'))
    it = it[(it.n_listens > 0) & (it.novoice_reports == 0) & ~it.rel.str.contains('/english/') & it.rel.isin(by.index)].copy()
    it['split'] = np.where(it.gold.notna(), 'gold', by.split.reindex(it.rel).values)
    it['path'] = by.path.reindex(it.rel).values
    it['dur'] = by.duration.reindex(it.rel).values
    return it.reset_index(drop=True)


def feats(bb, it, L, batch=16):
    F = [None] * len(it)
    for path, g in it.groupby('path'):
        x = load(DATA / path)
        W = []
        for r in g.itertuples():
            s = min(max(0.0, r.start + WIN / 2 - L / 2), max(0.0, len(x) / SR - L))
            W.append(cut(x, s, L))
        for k in range(0, len(W), batch):
            f = bb(W[k:k + batch])
            for j, i in enumerate(g.index[k:k + batch]):
                F[i] = f[j].astype(np.float32)
    return np.stack(F)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('backbone'); ap.add_argument('--lengths', default='3,5,8,12,20')
    a = ap.parse_args()
    bb = backbones.load(a.backbone)
    it = items(); y = it.z.values; w = 1 / (it.z_sd.values ** 2 + 16)
    rows = []
    for L in [float(v) for v in a.lengths.split(',')]:
        F = feats(bb, it, L)
        tr = (it.split == 'train').values; va = (it.split == 'val').values
        best = None
        for l in range(F.shape[1]):
            for al in ALPHAS:
                f = ridge_fit(F[tr, l], y[tr], w[tr], al)
                m = float(np.average(np.abs(f(F[va, l]) - y[va]), weights=w[va]))
                if best is None or m < best[0]:
                    best = (m, l, al)
        _, l, al = best
        f = ridge_fit(F[tr | va, l], y[tr | va], w[tr | va], al)
        row = dict(length=L, layer=l, val_mae=round(best[0], 2))
        for s in ('gold', 'test'):
            k = (it.split == s).values; e = np.abs(f(F[k, l]) - y[k]); b = np.minimum(y[k] // 10, 9)
            row[f'{s}_mae'] = round(float(e.mean()), 2)
            row[f'{s}_worst_band'] = round(max(e[b == q].mean() for q in range(10) if (b == q).sum() >= 5), 1)
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(work('winlen', f'{a.backbone}.csv'), index=False)


if __name__ == '__main__':
    main()
