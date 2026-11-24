"""Snapshot: one frozen table of every recording the pipeline may use, so a run is reproducible while data keeps growing.

recordings.csv columns
  rec       stable id (hash of source + relative path)
  source    dataset | new_voices | additional_ru | tidyvoice | cis_tg | cis_<corpus>
  path      path relative to VPM_DATA
  speaker   voice group: ECAPA cluster for our data, speaker id for corpora, author folder for cis_tg (hashed)
  language  ISO code when known
  duration  seconds
  gender_meta  f / m from corpus metadata (official corpora only), else empty

Usage: python -m vpm.snapshot [--sources dataset,new_voices,...]  ->  work/snapshot/recordings.csv
The rating tables are copied next to it with their SHA-256, so a target can be rebuilt exactly.
"""
import argparse, hashlib, json, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .config import DATA, SOURCES, work
from .audio import duration

AUDIO = ('.wav', '.opus', '.ogg', '.flac', '.mp3', '.m4a')


def hid(*parts, n=12):
    return hashlib.sha1('/'.join(map(str, parts)).encode()).hexdigest()[:n]


def ours():
    """Our recordings; speaker = ECAPA voice cluster where known, else the file itself."""
    z = np.load(DATA / 'voice_AI/voice-passability-vectors/ecapa192_current.npz', allow_pickle=True)
    cl = {str(f).split('/')[-1]: int(c) for f, c in zip(z['filenames'], z['cluster_ids'])}
    rows = []
    for src, (root, pat, lang) in SOURCES.items():
        for p in sorted((DATA / root).glob(pat)):
            rel = str(p.relative_to(DATA / 'voice_AI'))
            spk = f'c{cl[p.name]}' if p.name in cl else 'f' + hid(rel)
            rows.append(dict(source=src, path=str(p.relative_to(DATA)), rel=rel, speaker=spk, language=lang, gender_meta=''))
    return rows


def tidyvoice():
    """TidyVoiceX_ASV: <split>/<speaker>/<language>/<clip>; one speaker reads in several languages."""
    root = DATA / 'tidyvoice'
    rows = []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in AUDIO:
            continue
        parts = p.relative_to(root).parts
        spk, lang = _tidy_speaker_lang(parts)
        rows.append(dict(source='tidyvoice', path=str(p.relative_to(DATA)), rel='', speaker='t' + hid(spk),
                         language=lang, gender_meta=''))
    return rows


def _tidy_speaker_lang(parts):
    # filled in from the real layout in TIDYVOICE.md once the archive is unpacked
    return parts[-3], parts[-2]


def cis():
    """cis/<corpus>/<speaker>/<file>. Corpus 'tg' is the own Telegram set (no metadata: labels come from cislabel)."""
    rows = []
    for corpus in sorted(p for p in (DATA / 'cis').glob('*') if p.is_dir()):
        meta = {}
        mf = corpus / 'speakers.csv'      # optional: speaker,gender (official corpora)
        if mf.exists():
            meta = dict(pd.read_csv(mf, dtype=str, keep_default_na=False).values)
        for p in sorted(corpus.rglob('*')):
            if p.suffix.lower() not in AUDIO:
                continue
            spk = p.relative_to(corpus).parts[0]
            rows.append(dict(source=f'cis_{corpus.name}', path=str(p.relative_to(DATA)), rel='',
                             speaker='s' + hid(corpus.name, spk), language='', gender_meta=meta.get(spk, '')))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default='ours,tidyvoice,cis')
    a = ap.parse_args()
    rows = []
    for s in a.sources.split(','):
        rows += {'ours': ours, 'tidyvoice': tidyvoice, 'cis': cis}[s]()
    df = pd.DataFrame(rows)
    df.insert(0, 'rec', [hid(s, p) for s, p in zip(df.source, df.path)])
    with ThreadPoolExecutor(16) as ex:
        df['duration'] = list(ex.map(lambda p: round(duration(DATA / p), 3), df.path))
    df = df[df.duration > 0]
    out = work('snapshot', 'recordings.csv')
    df.to_csv(out, index=False)
    sums = {}
    for f in sorted((DATA / 'ratings').glob('*.csv')):
        shutil.copy2(f, work('snapshot', 'ratings', f.name))
        sums[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    json.dump(sums, open(work('snapshot', 'ratings_sha256.json'), 'w'), indent=1)
    print(df.groupby('source').agg(recordings=('rec', 'size'), speakers=('speaker', 'nunique'),
                                   hours=('duration', lambda d: round(d.sum() / 3600, 1))).to_string())


if __name__ == '__main__':
    main()
