"""Voice check per 0.5 s block (silero VAD), cached per recording.

The bot uses silero + YAMNet (rejects music and noise that silero lets through). Here only silero runs:
the cache decides which windows are used for training and for the consistency metrics, where a few
music windows matter far less than in a single user answer.

Cache: work/voicecheck/<source>.npz, key = rec, value = speech fraction of each 0.5 s block (float16).
Usage: python -m vpm.voicecheck [--source tidyvoice] [--jobs 16]
"""
import argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
from .config import DATA, SR, BLOCK, WIN, work
from .audio import load

FRAME = 512
WINDOW_MIN_SPEECH = 0.25   # a 5 s window counts as voiced when >= 25 % of it is speech (>= 1.25 s)
_model = None


def _init():
    global _model
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _model = load_silero_vad()


def blocks(x):
    """Speech fraction of each 0.5 s block."""
    import torch
    _model.reset_states()
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0, np.float16)
    t = torch.from_numpy(x[:n * FRAME].reshape(n, FRAME))
    with torch.no_grad():
        p = np.array([_model(t[i:i + 1], SR).item() for i in range(n)])
    centre = (np.arange(n) + 0.5) * FRAME / SR
    b = (centre // BLOCK).astype(int)
    nb = int(np.ceil(len(x) / (BLOCK * SR)))
    num = np.bincount(b, weights=(p > 0.5), minlength=nb)
    den = np.bincount(b, minlength=nb)
    return (num / np.maximum(den, 1)).astype(np.float16)


def _one(args):
    rec, path = args
    try:
        return rec, blocks(load(DATA / path))
    except Exception:
        return rec, None


def window_voiced(bl, start, length=WIN):
    """True when the window [start, start+length) has enough speech."""
    a = int(round(start / BLOCK)); b = a + int(round(length / BLOCK))
    seg = np.asarray(bl[a:b], dtype=np.float32)
    return bool(len(seg) and seg.sum() / (length / BLOCK) >= WINDOW_MIN_SPEECH)


def load_cache(sources=None):
    out = {}
    for f in sorted(work('voicecheck', 'x').parent.glob('*.npz')):
        if sources is None or f.stem in sources:
            z = np.load(f)
            out.update({k: z[k] for k in z.files})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=None)
    ap.add_argument('--jobs', type=int, default=16)
    a = ap.parse_args()
    rec = pd.read_csv(work('snapshot', 'recordings.csv'))
    for src, df in rec.groupby('source'):
        if a.source and src != a.source:
            continue
        f = work('voicecheck', f'{src}.npz')
        done = dict(np.load(f)) if f.exists() else {}
        todo = [(r, p) for r, p in zip(df.rec, df.path) if r not in done]
        with Pool(a.jobs, initializer=_init) as pool:
            for k, (r, b) in enumerate(pool.imap_unordered(_one, todo, chunksize=8)):
                if b is not None:
                    done[r] = b
                if (k + 1) % 5000 == 0:
                    np.savez(f, **done); print(src, k + 1, '/', len(todo), flush=True)
        np.savez(f, **done)
        sp = np.concatenate([v.astype(np.float32) for v in done.values()]) if done else np.zeros(1)
        print(f'{src}: {len(done)} recordings, speech in {100 * (sp > 0.5).mean():.0f}% of blocks', flush=True)


if __name__ == '__main__':
    main()
