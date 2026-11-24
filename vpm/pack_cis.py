"""Pack the labelled cis voices into one dataset for storage: cis_voices_v1/audio/<id>.opus + manifest.csv.

Kept: speakers with a label (f/m from corpus metadata, or both classifiers agreeing at the validated threshold).
Dropped: speakers without a confident label; recordings longer than MAX_DUR (streams, songs, montages);
recordings with speech in fewer than MIN_SPEECH of their 0.5 s blocks; and from the Telegram set, authors whose
recordings the classifiers put on different sides. Audio is re-encoded to Opus 64 kbit/s mono.

Usage: python -m vpm.pack_cis <out dir> [--jobs 16]
"""
import argparse, hashlib, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA, work
from .voicecheck import load_cache
from .cislabel import MIN_THR

MAX_DUR = 300.0
MIN_SPEECH = 0.30
LANG = {'cis_tg': 'ru', 'cis_fleurs_ru': 'ru', 'cis_fleurs_en': 'en', 'cis_librispeech': 'en'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--jobs', type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); (out / 'audio').mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(work('snapshot', 'recordings.csv'), keep_default_na=False)
    rec = rec[rec.source.str.startswith('cis_')]
    lab = pd.read_csv(work('cislabel', 'labels.csv'), keep_default_na=False)[['speaker', 'label']]
    sc = pd.read_csv(work('cislabel', 'scores.csv'))
    rec = rec.merge(lab, on='speaker', how='left').merge(sc[['rec', 'ecapa', 'audeer']], on='rec', how='left')
    vc = load_cache()
    rec['speech'] = [float((np.asarray(vc[r], np.float32) > 0.5).mean()) if r in vc else np.nan for r in rec.rec]
    n0 = len(rec); why = {}
    def drop(mask, reason):
        nonlocal rec
        why[reason] = int(mask.sum()); rec = rec[~mask]
    drop(~rec.label.isin(['f', 'm']), 'no confident speaker label')
    drop(rec.duration > MAX_DUR, f'longer than {MAX_DUR:.0f} s')
    drop(~(rec.speech >= MIN_SPEECH), f'speech in < {MIN_SPEECH:.0%} of blocks')
    rf = (rec.ecapa >= MIN_THR) & (rec.audeer >= MIN_THR); rm = (rec.ecapa <= 1 - MIN_THR) & (rec.audeer <= 1 - MIN_THR)
    per = pd.DataFrame({'speaker': rec.speaker, 'f': rf, 'm': rm}).groupby('speaker')[['f', 'm']].any()
    mixed = set(per.index[per.f & per.m])
    drop(rec.source.eq('cis_tg') & rec.speaker.isin(mixed), 'author with recordings on both sides')
    rec = rec.copy()
    rec['id'] = ['c' + hashlib.sha1(f'{s}/{p}'.encode()).hexdigest()[:16] for s, p in zip(rec.source, rec.path)]
    rec['file'] = 'audio/' + rec.id + '.opus'

    def enc(r):
        dst = out / r.file
        if dst.exists() and dst.stat().st_size > 0:
            return True
        tmp = dst.with_suffix('.tmp.opus')
        ok = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(DATA / r.path), '-ac', '1', '-ar', '16000',
                             '-c:a', 'libopus', '-b:a', '64k', str(tmp)]).returncode == 0
        if ok:
            tmp.rename(dst)
        return ok
    with ThreadPoolExecutor(a.jobs) as ex:
        ok = list(ex.map(enc, rec.itertuples()))
    rec = rec[ok]
    rec['language'] = rec.source.map(LANG)
    rec['label_from'] = np.where(rec.gender_meta.isin(['f', 'm']), 'corpus metadata', 'classifiers')
    cols = ['id', 'file', 'source', 'language', 'speaker', 'label', 'label_from', 'duration', 'speech', 'ecapa', 'audeer']
    rec[cols].round(3).to_csv(out / 'manifest.csv', index=False)
    print(f'kept {len(rec)} of {n0}; dropped: {why}')
    print(rec.groupby(['source', 'label']).agg(recordings=('id', 'size'), speakers=('speaker', 'nunique'),
                                               hours=('duration', lambda d: round(d.sum() / 3600, 1))).to_string())


if __name__ == '__main__':
    main()
