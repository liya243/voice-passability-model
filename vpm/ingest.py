"""Bring an official cis corpus into cis/<corpus>/<speaker>/<file> with cis/<corpus>/speakers.csv (speaker, gender f/m).

  fleurs <dir with data/<lang>/> <lang>   google/fleurs (CC-BY 4.0): gender per utterance, no speaker id,
                                          so each utterance is its own "speaker" (only used for labels and anchors)
  librispeech <dir with LibriSpeech/>     OpenSLR 12 (CC-BY 4.0): SPEAKERS.TXT gives sex per speaker

Files are symlinked, not copied. Usage: python -m vpm.ingest fleurs /data/dl/fleurs ru_ru
"""
import sys, tarfile
from pathlib import Path
import pandas as pd
from .config import DATA


def fleurs(root, lang):
    root = Path(root) / 'data' / lang
    out = DATA / 'cis' / f'fleurs_{lang.split("_")[0]}'
    rows = []
    for split in ('train', 'dev', 'test'):
        adir = root / 'audio' / split
        if not adir.exists():
            with tarfile.open(root / 'audio' / f'{split}.tar.gz') as t:
                t.extractall(root / 'audio', filter='data')
        t = pd.read_csv(root / f'{split}.tsv', sep='\t', header=None, quoting=3,
                        names=['id', 'file', 'raw', 'text', 'phon', 'n', 'gender'])
        for r in t.itertuples():
            src = adir / r.file
            if not src.exists():
                continue
            spk = Path(r.file).stem
            d = out / spk; d.mkdir(parents=True, exist_ok=True)
            (d / r.file).unlink(missing_ok=True); (d / r.file).symlink_to(src)
            rows.append((spk, {'FEMALE': 'f', 'MALE': 'm'}.get(r.gender, '')))
    pd.DataFrame(rows, columns=['speaker', 'gender']).drop_duplicates('speaker').to_csv(out / 'speakers.csv', index=False)
    print(out, len(rows))


def librispeech(root):
    root = Path(root) / 'LibriSpeech'
    out = DATA / 'cis' / 'librispeech'
    sex = {}
    for line in open(root / 'SPEAKERS.TXT'):
        if line.startswith(';'):
            continue
        p = [x.strip() for x in line.split('|')]
        if len(p) >= 2:
            sex[p[0]] = {'F': 'f', 'M': 'm'}.get(p[1], '')
    n = 0
    for f in root.glob('*/*/*/*.flac'):
        spk = f.parts[-3]
        d = out / spk; d.mkdir(parents=True, exist_ok=True)
        (d / f.name).unlink(missing_ok=True); (d / f.name).symlink_to(f); n += 1
    spk = sorted(p.name for p in out.iterdir() if p.is_dir())
    pd.DataFrame({'speaker': spk, 'gender': [sex.get(s, '') for s in spk]}).to_csv(out / 'speakers.csv', index=False)
    print(out, n, 'files', len(spk), 'speakers')


if __name__ == '__main__':
    {'fleurs': fleurs, 'librispeech': librispeech}[sys.argv[1]](*sys.argv[2:])
