"""One trans dataset instead of dataset/, new_voices/, additional_data/{russian,english}/.

  trans_voices_v1/     audio/<id>.<ext> + manifest.csv
  feedback_voices_v1/  voices people complained about in the bot ("should be higher/lower"), kept apart:
                       their labels mean something else; manifest has direction and the score shown then

id = 'v' + first 16 hex of the file's SHA-256, so the same file gets the same id on every machine and
duplicates across folders collapse into one row (all original paths are kept, ';'-joined).
Files are hard-linked when possible (no extra disk, old paths keep working for the bot), else copied.

  build  <voice_AI root> <out root> [items.csv]    canonical build (bot server); feedback voices that people
                                                   already rated (items.csv from vpm.target) go into trans too
  attach <manifest.csv> <out root> <dir> [dir...]  rebuild elsewhere from files already there, matched by
                                                   SHA-256; prints what is missing
"""
import csv, hashlib, json, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AUDIO = ('.wav', '.ogg', '.opus', '.flac', '.mp3', '.m4a')
SOURCES = [('dataset', 'dataset', 'ru'), ('new_voices', 'new_voices', 'ru'),
           ('additional_ru', 'additional_data/russian', 'ru'), ('additional_en', 'additional_data/english', 'en')]
FEEDBACK_KEEP = ('report_direction', 'mean_pct', 'mean_pred', 'duration', 'saved_at', 'model_name')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def place(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(root, out, items=None):
    root, out = Path(root), Path(out)
    rated = set()
    if items:
        rated = {r['rel'] for r in csv.DictReader(open(items)) if int(r['n_listens']) > 0}
    files = []
    for src, sub, lang in SOURCES:
        for p in sorted((root / sub).rglob('*')):
            if p.suffix.lower() in AUDIO and p.is_file() and not p.name.endswith('Zone.Identifier'):
                if src == 'new_voices' and p.with_suffix('.wav').exists() and p.suffix.lower() != '.wav':
                    continue                         # .ogg next to its converted .wav
                files.append((src, lang, p))
    with ThreadPoolExecutor(8) as ex:
        sums = list(ex.map(lambda t: sha(t[2]), files))
    trans, feedback = {}, {}
    for (src, lang, p), h in zip(files, sums):
        fb = p.name.startswith('pass_feedback_')
        rows = feedback if fb else trans
        vid = 'v' + h[:16]
        r = rows.setdefault(vid, dict(id=vid, file=f'audio/{vid}{p.suffix.lower()}', sha256=h, source=src,
                                      language=lang, bytes=p.stat().st_size, orig_paths=[]))
        r['orig_paths'].append(str(p.relative_to(root)))
        if fb:
            meta = p.with_suffix('.json')
            if meta.exists():
                m = json.load(open(meta))
                r.update({k: m.get(k, '') for k in FEEDBACK_KEEP})
        place(p, out / ('feedback_voices_v1' if fb else 'trans_voices_v1') / r['file'])
        if fb and str(p.relative_to(root)) in rated:     # rated by people: also part of the trans dataset
            t = trans.setdefault(vid, {**{k: v for k, v in r.items() if k in ('id', 'file', 'sha256', 'language', 'bytes')},
                                       'source': 'feedback_rated', 'orig_paths': []})
            t['orig_paths'].append(str(p.relative_to(root)))
            place(p, out / 'trans_voices_v1' / t['file'])
    for name, rows in (('trans_voices_v1', trans), ('feedback_voices_v1', feedback)):
        cols = ['id', 'file', 'sha256', 'source', 'language', 'bytes', 'orig_paths'] + (list(FEEDBACK_KEEP) if name.startswith('feedback') else [])
        with open(out / name / 'manifest.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader()
            for r in rows.values():
                w.writerow({**r, 'orig_paths': ';'.join(r['orig_paths'])})
        by = {}
        for r in rows.values():
            by[r['source'], r['language']] = by.get((r['source'], r['language']), 0) + 1
        print(name, len(rows), 'recordings', dict(sorted(by.items())), f'from {sum(len(r["orig_paths"]) for r in rows.values())} files')


def attach(manifest, out, *dirs):
    out = Path(out); manifest = Path(manifest)
    rows = list(csv.DictReader(open(manifest)))
    want = {r['sha256']: r for r in rows}
    files = [p for d in dirs for p in Path(d).rglob('*') if p.suffix.lower() in AUDIO and p.is_file()]
    found = 0
    with ThreadPoolExecutor(8) as ex:
        for p, h in zip(files, ex.map(sha, files)):
            if h in want:
                place(p, out / want[h]['file']); found += 1
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, out / 'manifest.csv')
    missing = [r for r in rows if not (out / r['file']).exists()]
    with open(out / 'missing.txt', 'w') as f:
        f.write('\n'.join(r['file'] for r in missing))
    print(f'{len(rows) - len(missing)} of {len(rows)} present, {len(missing)} missing -> {out / "missing.txt"}')


if __name__ == '__main__':
    {'build': build, 'attach': attach}[sys.argv[1]](*sys.argv[2:])
