"""Fine-tune a backbone with the pipeline loss; select on val voices; write scorecard predictions.

Each step mixes:  labelled items (scale + order)  |  window pairs of one recording (consistency)
                  TidyVoice speaker in two languages (language)  |  cis windows (cis)  |  miniapp pair votes (order)
                  windows without a voice (voice flag; score losses are never applied to them)
Schedule: first WARM of the steps train the heads only (backbone frozen) on scale, order and voice; then the
consistency, language and adversarial weights ramp from 0 to full by the end of epoch 1 while the top layers
unfreeze. Starting the invariance terms at full weight lets a fresh head collapse to one score for everyone.
Selection: mean of per-band MAE over the val bands that have >= 10 items, so no band can quietly fall apart.

Usage: python -m vpm.train --backbone mhubert147 --name mh1 [--epochs 6 --unfreeze 8 --w_lang 1 ...]
  -> work/runs/<name>/{config.json, model.pt, log.txt}, work/preds/<name>.csv
"""
import argparse, json, math, time
import numpy as np
import pandas as pd
import torch
from .config import DATA, SR, WIN, work
from .audio import load, cut, window_starts
from .split import recordings
from .voicecheck import load_cache, window_voiced

NOISE_MAX_SPEECH = 0.05     # a window with < 5 % speech by silero is a "no voice" example


def speech_share(bl, start):
    a = int(round(start / 0.5)); seg = np.asarray(bl[a:a + 10], dtype=np.float32)
    return float(seg.sum() / 10) if len(seg) else 0.0
from .probe import train_windows
from .model import PassModel
from . import loss as L

N = int(WIN * SR)
PAD = 0.75


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--unfreeze', type=int, default=8)
    ap.add_argument('--lr_bb', type=float, default=2e-5); ap.add_argument('--lr_head', type=float, default=5e-4)
    ap.add_argument('--b_lab', type=int, default=16); ap.add_argument('--b_rec', type=int, default=4)
    ap.add_argument('--b_lang', type=int, default=4); ap.add_argument('--b_cis', type=int, default=8)
    ap.add_argument('--w_order', type=float, default=0.5); ap.add_argument('--w_rec', type=float, default=1.0)
    ap.add_argument('--w_lang', type=float, default=1.0); ap.add_argument('--w_adv', type=float, default=0.1)
    ap.add_argument('--w_cis', type=float, default=0.5); ap.add_argument('--w_votes', type=float, default=1.0)
    ap.add_argument('--w_voice', type=float, default=0.3); ap.add_argument('--b_noise', type=int, default=6)
    ap.add_argument('--warm', type=float, default=0.05)
    ap.add_argument('--max_rec', type=int, default=4000); ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


class Bank:
    """Audio kept in RAM as float16: each entry is a window with PAD seconds on both sides for jitter."""
    def __init__(self):
        self.x, self.meta = [], []

    def add(self, x, start, pad=PAD, **meta):
        s0 = max(0.0, start - pad)
        seg = x[int(s0 * SR):int((start + WIN + pad) * SR)]
        self.x.append(seg.astype(np.float16)); self.meta.append(dict(off=int(round((start - s0) * SR)), **meta))

    def get(self, i, rng, jitter=True):
        x, off = self.x[i], self.meta[i]['off']
        o = off + (int(rng.integers(-PAD * SR, PAD * SR + 1)) if jitter else 0)
        o = max(0, min(o, max(0, len(x) - N)))
        w = x[o:o + N].astype(np.float32); w = np.pad(w, (0, N - len(w)))
        p = np.abs(w).max()
        return w / p if p > 0 else w


def build_banks(a, rng):
    rec = recordings(); vc = load_cache()
    rec_by = rec.set_index('rec')
    banks = {k: Bank() for k in ('lab', 'val', 'rec', 'lang', 'cis', 'votes', 'noise')}
    # labelled items
    tw = train_windows()
    for path, g in tw.groupby('path'):
        try:
            x = load(DATA / path)
        except Exception:
            continue
        for r in g.itertuples():
            banks['lab' if r.split == 'train' else 'val'].add(x, r.start, z=r.z, zsd=r.z_sd)
    # recording pairs: train voices of our data, >= 2 voiced windows
    ours = rec[rec.source.isin(['dataset', 'new_voices', 'additional_ru']) & (rec.split == 'train')].sample(frac=1, random_state=a.seed)
    k = 0
    for r in ours.itertuples():
        bl = vc.get(r.rec)
        if bl is None:
            continue
        ws = window_starts(r.duration)
        st = [s for s in ws if window_voiced(bl, s)]
        silent = [s for s in ws if speech_share(bl, s) < NOISE_MAX_SPEECH]
        if len(st) < 2 and not silent:
            continue
        x = load(DATA / r.path)
        if len(st) >= 2:
            for s in rng.permutation(st)[:4]:
                banks['rec'].add(x, float(s), pad=0, group=r.rec)
        for s in silent[:2]:
            banks['noise'].add(x, float(s), pad=0)
        k += 1
        if k >= a.max_rec:
            break
    # TidyVoice: train speakers with >= 2 languages, 2 clips per language
    tv = rec[(rec.source == 'tidyvoice') & (rec.split == 'train')]
    langs = sorted(tv.language.unique()); lid = {l: i for i, l in enumerate(langs)}
    for spk, g in tv.groupby('speaker'):
        if g.language.nunique() < 2:
            continue
        for lang, gl in g.groupby('language'):
            for r in gl.sample(min(2, len(gl)), random_state=a.seed).itertuples():
                x = load(DATA / r.path)
                banks['lang'].add(x, 0.0, pad=0, group=spk, lang=lid[lang])
    # cis: labelled train speakers
    lab = work('cislabel', 'labels.csv')
    if lab.exists():
        Lb = pd.read_csv(lab, keep_default_na=False).set_index('speaker').label
        cs = rec[rec.source.str.startswith('cis_') & (rec.split == 'train') & rec.speaker.map(Lb).isin(['f', 'm'])]
        for spk, g in cs.groupby('speaker'):
            for r in g.sample(min(2, len(g)), random_state=a.seed).itertuples():
                bl = vc.get(r.rec)
                st = [s for s in window_starts(r.duration) if bl is not None and window_voiced(bl, s)][:2]
                if st:
                    x = load(DATA / r.path)
                    for s in st:
                        banks['cis'].add(x, s, pad=0, y=1 if Lb[spk] == 'f' else 0)
    # windows people reported as "not a voice"; MUSAN music/noise if present under DATA/noise/
    nv = pd.read_csv(DATA / 'ratings/model_training_novoice_reports.csv')
    for r in nv.itertuples():
        try:
            banks['noise'].add(load(DATA / str(r.audio_path).replace('/root/voice_AI/', 'voice_AI/'), float(r.segment_start), WIN), 0.0, pad=0)
        except Exception:
            pass
    for f in sorted((DATA / 'noise').rglob('*.wav'))[:3000]:
        x = load(f)
        for s in window_starts(len(x) / SR)[:2]:
            banks['noise'].add(x, s, pad=0)
    # miniapp pair votes (windows taken as heard)
    pv = pd.read_csv(work('target', 'pairs.csv'))
    for r in pv.itertuples():
        for side in 'ab':
            p = str(getattr(r, f'{side}_path')).replace('/root/voice_AI/', 'voice_AI/')
            try:
                banks['votes'].add(load(DATA / p, getattr(r, f'{side}_start'), WIN), 0.0, pad=0, y=r.y)
            except Exception:
                banks['votes'].add(np.zeros(N, np.float32), 0.0, pad=0, y=np.nan)
    print({k: len(b.x) for k, b in banks.items()}, flush=True)
    return banks, langs


def pairs_of(bank, n, rng, by='group'):
    groups = {}
    for i, m in enumerate(bank.meta):
        groups.setdefault(m[by], []).append(i)
    groups = [g for g in groups.values() if len(g) >= 2]
    out = []
    for _ in range(n):
        g = groups[rng.integers(len(groups))]
        out.append(tuple(rng.choice(g, 2, replace=False)))
    return out


def evaluate(m, bank, dev, rng):
    m.eval(); p = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for k in range(0, len(bank.x), 64):
            x = torch.from_numpy(np.stack([bank.get(i, rng, jitter=False) for i in range(k, min(k + 64, len(bank.x)))])).to(dev)
            p.append(m(x).float().cpu().numpy())
    m.train()
    p = np.concatenate(p); y = np.array([mm['z'] for mm in bank.meta])
    e = np.abs(p - y); b = np.minimum(y // 10, 9).astype(int)
    bands = {int(k): float(e[b == k].mean()) for k in range(10) if (b == k).sum() >= 10}
    return dict(mae=float(e.mean()), band_mean=float(np.mean(list(bands.values()))), worst=max(bands.values()), bands=bands)


def main():
    a = args()
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda'
    out = work('runs', a.name, 'x').parent
    log = open(out / 'log.txt', 'a')
    def P(*s):
        s = ' '.join(map(str, s)); print(s, flush=True); log.write(s + '\n'); log.flush()
    P('args', json.dumps(vars(a)))
    banks, langs = build_banks(a, rng)
    m = PassModel(a.backbone, n_languages=len(langs) if a.w_adv > 0 else 0).to(dev)
    m.unfreeze_top(a.unfreeze)
    head = [p for n, p in m.named_parameters() if not n.startswith('bb.')]
    bb = [p for p in m.bb.parameters() if p.requires_grad]
    for p in bb:
        p.requires_grad = False         # frozen during the warm-up
    opt = torch.optim.AdamW([{'params': bb, 'lr': a.lr_bb}, {'params': head, 'lr': a.lr_head}], weight_decay=0.01)
    lab = banks['lab']; z = np.array([x['z'] for x in lab.meta]); zsd = np.array([x['zsd'] for x in lab.meta])
    steps = a.epochs * (len(lab.x) // a.b_lab); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    n_votes = len(banks['votes'].x) // 2
    epoch = len(lab.x) // a.b_lab; w0 = int(a.warm * steps)
    ramp = lambda s: 0.0 if s < w0 else min(1.0, (s - w0) / max(1, epoch - w0))
    best = None; t0 = time.time(); run = []
    for step in range(steps):
        if step == w0:
            for p in bb:
                p.requires_grad = True
            P('warm-up done: backbone top layers unfrozen')
        r = ramp(step)
        idx = rng.integers(0, len(lab.x), a.b_lab)
        X = [lab.get(i, rng) for i in idx]
        rp = pairs_of(banks['rec'], a.b_rec, rng) if a.w_rec and len(banks['rec'].x) else []
        lp = pairs_of(banks['lang'], a.b_lang, rng) if a.w_lang and len(banks['lang'].x) else []
        ci = rng.integers(0, len(banks['cis'].x), a.b_cis) if a.w_cis and len(banks['cis'].x) else []
        vi = rng.integers(0, n_votes, 2) if a.w_votes and n_votes else []
        X += [banks['rec'].get(i, rng, False) for pr in rp for i in pr]
        X += [banks['lang'].get(i, rng, False) for pr in lp for i in pr]
        X += [banks['cis'].get(i, rng, False) for i in ci]
        X += [banks['votes'].get(2 * v + s, rng, False) for v in vi for s in (0, 1)]
        ni = rng.integers(0, len(banks['noise'].x), a.b_noise) if a.w_voice and len(banks['noise'].x) else []
        n_voiced = len(X)
        X += [banks['noise'].get(i, rng, False) for i in ni]
        x = torch.from_numpy(np.stack(X)).to(dev)
        x = x + torch.randn_like(x) * (10 ** -rng.uniform(1.75, 3)) * float(rng.random() < 0.3)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, vlog, lg = m(x, heads=True, lam=r)
        p = p.float(); o = a.b_lab
        yv = torch.cat([torch.ones(n_voiced), torch.zeros(len(ni))]).to(dev)
        zt = torch.tensor(z[idx], device=dev, dtype=torch.float32); st = torch.tensor(zsd[idx], device=dev, dtype=torch.float32)
        parts = {'scale': L.scale(p[:o], zt, st), 'order': a.w_order * L.order_from_consensus(p[:o], zt, st)}
        if rp:
            q = p[o:o + 2 * len(rp)]; parts['rec'] = r * a.w_rec * L.same_voice(q[0::2], q[1::2]); o += 2 * len(rp)
        if lp:
            q = p[o:o + 2 * len(lp)]; parts['lang'] = r * a.w_lang * L.same_voice(q[0::2], q[1::2])
            if lg is not None and r > 0:
                yl = torch.tensor([banks['lang'].meta[i]['lang'] for pr in lp for i in pr], device=dev)
                parts['adv'] = a.w_adv * torch.nn.functional.cross_entropy(lg[o:o + 2 * len(lp)].float(), yl)
            o += 2 * len(lp)
        if len(ci):
            yc = torch.tensor([banks['cis'].meta[i]['y'] for i in ci], device=dev)
            parts['cis'] = a.w_cis * L.cis(p[o:o + len(ci)], yc); o += len(ci)
        if len(vi):
            q = p[o:o + 4]; yv = torch.tensor([banks['votes'].meta[2 * v]['y'] for v in vi], device=dev, dtype=torch.float32)
            ok = ~torch.isnan(yv)
            if ok.any():
                parts['votes'] = a.w_votes * L.order_from_votes(q[0::2][ok], q[1::2][ok], yv[ok])
        if len(ni):
            parts['voice'] = a.w_voice * torch.nn.functional.binary_cross_entropy_with_logits(vlog.float(), yv)
        loss = sum(parts.values())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        run.append({k: float(v) for k, v in parts.items()})
        if (step + 1) % 200 == 0:
            P(f'step {step + 1}/{steps} ramp {r:.2f}', {k: round(float(np.mean([q.get(k, 0) for q in run[-200:]])), 4) for k in parts}, f'{time.time() - t0:.0f}s')
        if (step + 1) % (len(lab.x) // a.b_lab) == 0 or step + 1 == steps:
            v = evaluate(m, banks['val'], dev, rng); P('val', json.dumps(v))
            if best is None or v['band_mean'] < best:
                best = v['band_mean']; m.save(out, val=v, step=step + 1, languages=langs); P('  saved')
    P('best val band_mean', best)


if __name__ == '__main__':
    main()
