"""Cheap first test of a backbone: frozen features, one ridge per layer, then the scorecard.

Train windows: listened items of train voices (Russian, not gold, no "not a voice" report), weight 1/(z_sd^2+16).
The layer and ridge strength are chosen on val voices only. The chosen probe scores the scorecard windows,
so its row in the scorecard is directly comparable with ft3 and every other model.

Usage: python -m vpm.probe <backbone> [--batch 32]
  -> work/features/<backbone>/{train,score}.npy, work/preds/probe_<backbone>.csv, work/probe/<backbone>.json
"""
import argparse, json, time
import numpy as np
import pandas as pd
from .config import DATA, work
from .audio import load, cut
from .split import recordings
from . import backbones

ALPHAS = [1, 10, 100, 1000, 10000]


def train_windows():
    rec = recordings()
    by_rel = rec[rec.rel != ''].set_index('rel')
    it = pd.read_csv(work('target', 'items.csv'))
    it = it[it.gold.isna() & (it.n_listens > 0) & (it.novoice_reports == 0) & ~it.rel.str.contains('/english/')
            & it.rel.isin(by_rel.index)].copy()
    it['split'] = by_rel.split.reindex(it.rel).values
    it['path'] = by_rel.path.reindex(it.rel).values
    return it[it.split.isin(['train', 'val'])].reset_index(drop=True)


def extract(bb, df, out, batch):
    if out.exists():
        return np.load(out, mmap_mode='r')
    feats = [None] * len(df)
    t0 = time.time()
    for path, g in df.groupby('path'):
        try:
            x = load(DATA / path)
        except Exception:
            continue
        W = [cut(x, s) for s in g.start]
        for k in range(0, len(W), batch):
            f = bb(W[k:k + batch]).astype(np.float16)
            for j, i in enumerate(g.index[k:k + batch]):
                feats[i] = f[j]
    shape = next(f for f in feats if f is not None).shape
    F = np.stack([f if f is not None else np.full(shape, np.nan, np.float16) for f in feats])
    np.save(out, F)
    print(f'{out.name}: {F.shape} in {time.time() - t0:.0f}s', flush=True)
    return F


def ridge_fit(X, y, w, alpha):
    mu = np.average(X, 0, w); sd = X.std(0) + 1e-6
    Z = (X - mu) / sd; ym = np.average(y, weights=w)
    A = Z.T @ (Z * w[:, None]) + alpha * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Z.T @ (w * (y - ym)))
    return lambda X2: np.clip(((X2 - mu) / sd) @ b + ym, 0, 100)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('backbone'); ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()
    bb = backbones.load(a.backbone)
    tw = train_windows()
    sw = pd.read_csv(work('scorecard', 'windows.csv'), index_col='w').reset_index()
    Ft = np.asarray(extract(bb, tw, work('features', a.backbone, 'train.npy'), a.batch), np.float32)
    Fs = np.asarray(extract(bb, sw, work('features', a.backbone, 'score.npy'), a.batch), np.float32)
    ok = ~np.isnan(Ft).any((1, 2))
    tr = ok & (tw.split == 'train').values; va = ok & (tw.split == 'val').values
    y = tw.z.values; w = 1 / (tw.z_sd.values ** 2 + 16)
    best = None
    for L in range(Ft.shape[1]):
        for al in ALPHAS:
            f = ridge_fit(Ft[tr, L], y[tr], w[tr], al)
            mae = float(np.average(np.abs(f(Ft[va, L]) - y[va]), weights=w[va]))
            if best is None or mae < best[0]:
                best = (mae, L, al)
    mae, L, al = best
    f = ridge_fit(Ft[tr | va, L], y[tr | va], w[tr | va], al)
    pred = np.full(len(sw), np.nan)
    okS = ~np.isnan(Fs[:, L]).any(1)
    pred[okS] = f(Fs[okS, L])
    pd.DataFrame({'w': sw.w, 'pred': pred.round(3)}).to_csv(work('preds', f'probe_{a.backbone}.csv'), index=False)
    json.dump(dict(backbone=a.backbone, layer=L, alpha=al, val_mae=mae, n_train=int(tr.sum()), n_val=int(va.sum())),
              open(work('probe', f'{a.backbone}.json'), 'w'), indent=1)
    print(f'{a.backbone}: layer {L}, alpha {al}, val MAE {mae:.2f} (train {tr.sum()}, val {va.sum()})')


if __name__ == '__main__':
    main()
