"""Training target from human ratings: an expert-anchored consensus where every rater has their own monotone scale.

Each listened 5 s item i has a latent score z_i on the expert's scale (0..100).
Rater r answers  s = h_r(z) + noise_r,  h_r monotone piecewise-linear with knots every 10 points,
h_expert = identity (the anchor), other curves shrunk toward a pooled listener curve.
Seed labels (one per dataset file) are one more "rater" whose answer refers to the whole file.
Alternating fit: h_r and noise_r given z; z on a 0..100 grid given h (posterior mean and sd).
A rater's noise is the disagreement with the expert on shared items minus the expert's own retest noise.

Only the order of z matters for training (see loss.py); z_sd sets how much an item is trusted.

Soft cis anchors (when DATA/cis_links.csv and work/cislabel/labels.csv exist): an item recorded by a speaker
labelled cis female adds log sigmoid((z - 90) / CIS_S) to its posterior, cis male log sigmoid((20 - z) / CIS_S).
The expert's curve is then no longer fixed to the identity but fitted, monotone, with a strong pull toward it
(EXPERT_LAM), so the anchors can stretch the ends of the scale while the expert's order is kept.

Usage: python -m vpm.target  ->  work/target/items.csv, raters.csv, pairs.csv
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from .config import DATA, work

K = np.linspace(0, 100, 11)
G = np.arange(0, 101, 1.0)
OVERLAP = 2.5        # a vote joins an item when the windows overlap by >= 2.5 s
CIS_F, CIS_M, CIS_S = 90.0, 20.0, 3.0
EXPERT_LAM = 200.0


def _rel(p):
    return p.str.replace('/root/voice_AI/', '', regex=False)


def _basis(z):
    z = np.clip(z, 0, 100)
    k = np.minimum((z // 10).astype(int), 9)
    t = (z - K[k]) / 10
    B = np.zeros((len(z), 11))
    B[np.arange(len(z)), k] = 1 - t
    B[np.arange(len(z)), k + 1] = t
    return B


def fit_curve(z, s, w, prior=None, lam=0.0):
    """Monotone curve through (z, s): knots = c0 + cumsum(d), d >= 0."""
    T = np.tril(np.ones((11, 11)))
    A = _basis(z) @ T * np.sqrt(w)[:, None]
    b = s * np.sqrt(w)
    if prior is not None and lam > 0:
        A = np.vstack([A, np.sqrt(lam) * T])
        b = np.r_[b, np.sqrt(lam) * prior]
    x = lsq_linear(A, b, bounds=(np.r_[-np.inf, np.zeros(10)], np.inf)).x
    return np.clip(T @ x, 0, 100)


def load_votes(R):
    a = pd.read_csv(R / 'model_training_authors.csv')
    expert = int(a.author_id[a.is_expert.astype(str).str.lower().isin(['t', 'true'])].iloc[0])
    v = pd.read_csv(R / 'model_training_votes.csv'); v['rel'] = _rel(v.audio_path)
    seed = v[v.sample_id.astype(str).str.startswith('seed:')].copy()
    hv = v[~v.sample_id.astype(str).str.startswith('seed:')]
    cv = pd.read_csv(R / 'model_training_cluster_votes.csv'); cv['rel'] = _rel(cv.audio_path)
    gv = pd.read_csv(R / 'model_training_gold_votes.csv'); gv['rel'] = _rel(gv.audio_path)
    votes = pd.concat([hv.assign(src='individual'), cv.assign(src='cluster'), gv.assign(src='gold')])
    return expert, votes[['rel', 'segment_start', 'author_id', 'score', 'src']].reset_index(drop=True), seed, gv


def build_items(votes, seed, gold):
    """Gold windows first, then greedy by start: a vote joins an existing item of its file when overlap >= 2.5 s."""
    items, by_file = [], {}
    for g in gold.itertuples():
        by_file.setdefault(g.relative_path, []).append(len(items))
        items.append(dict(rel=g.relative_path, start=float(g.segment_start), gold=g.id))
    vi = np.empty(len(votes), int)
    for j, (r, s) in enumerate(zip(votes.rel.values, votes.segment_start.values)):
        best, bo = -1, OVERLAP
        for i in by_file.get(r, []):
            o = 5 - abs(items[i]['start'] - s)
            if o >= bo:
                best, bo = i, o
        if best < 0:
            best = len(items); by_file.setdefault(r, []).append(best)
            items.append(dict(rel=r, start=float(s), gold=None))
        vi[j] = best
    for r in seed.rel.unique():
        if r not in by_file:
            by_file[r] = [len(items)]; items.append(dict(rel=r, start=0.0, gold=None))
    return pd.DataFrame(items), vi, by_file


def consensus(expert, obs, N, retest, expert_boost=1.0, iters=25, anchors=None):
    """anchors: array of N with +1 (cis female), -1 (cis male), 0 (none)."""
    A = np.zeros(N) if anchors is None else np.asarray(anchors, float)
    soft = bool(np.any(A != 0))
    cis_ll = np.zeros((N, len(G)))
    lsig = lambda x: -np.logaddexp(0, -x)
    cis_ll[A > 0] = lsig((G - CIS_F) / CIS_S)
    cis_ll[A < 0] = lsig((CIS_M - G) / CIS_S)
    raters = obs.rater.unique()
    H = {r: K.copy() for r in raters}
    S2 = {r: 400.0 for r in raters}
    e = obs[obs.rater == expert].groupby('item').score
    exp_mean, exp_n = e.mean(), e.count()
    pool_s2 = 300.0
    z = obs.groupby('item').score.mean().reindex(range(N)).fillna(50).values
    listener = (obs.rater != expert) & (obs.rater != -1)
    for it in range(iters):
        pool = fit_curve(z[obs.item[listener]], obs.score[listener].values, obs.w0[listener].values)
        for r in raters:
            m = obs.rater == r
            if r == expert:
                m_e = obs.rater == expert
                H[r] = fit_curve(z[obs.item[m_e]], obs.score[m_e].values, obs.w0[m_e].values, K, EXPERT_LAM) if soft else K.copy()
                S2[r] = retest ** 2; continue
            H[r] = fit_curve(z[obs.item[m]], obs.score[m].values, obs.w0[m].values,
                             *((None, 0.0) if r == -1 else (pool, 30.0)))
            mm = m & obs.item.isin(exp_mean.index)
            ev = exp_mean.reindex(obs.item[mm]).values; en = exp_n.reindex(obs.item[mm]).values
            slope = np.gradient(H[r], K)[np.minimum((ev // 10).astype(int), 10)] if len(ev) else ev
            res2 = (obs.score[mm].values - np.interp(ev, K, H[r])) ** 2 - slope ** 2 * retest ** 2 / en
            n = obs.w0[mm].sum()
            S2[r] = max(25.0, (np.sum(obs.w0[mm].values * res2) + 10 * pool_s2) / (n + 10))
        pool_s2 = float(np.median([S2[r] for r in raters if r not in (expert, -1)]))
        w = obs.w0.values / np.array([S2[r] for r in obs.rater]) * np.where(obs.rater == expert, expert_boost, 1.0)
        L = np.tile(-((G - 50) ** 2) / (2 * 40 ** 2), (N, 1)) + cis_ll
        ridx = {r: k for k, r in enumerate(raters)}
        P = np.stack([np.interp(G, K, H[r]) for r in raters])[[ridx[r] for r in obs.rater]]
        np.add.at(L, obs.item.values, -0.5 * w[:, None] * (obs.score.values[:, None] - P) ** 2)
        L -= L.max(1, keepdims=True); p = np.exp(L); p /= p.sum(1, keepdims=True)
        zn = p @ G; zsd = np.sqrt(np.maximum(p @ G ** 2 - zn ** 2, 0))
        delta = np.abs(zn - z).max(); z = zn
        if delta < 0.05:
            break
    return z, zsd, H, S2, it + 1


def pass_pairs(R, items):
    """Pairwise votes from the miniapp ("which voice sounds more feminine": 'a', 'b', 'tie') -> y = 1, 0, 0.5.
    item_a / item_b point to a rated item when the window overlaps one, else -1 (the window is still usable)."""
    try:
        s = pd.read_csv(R / 'pass_pair_samples.csv'); v = pd.read_csv(R / 'pass_pair_votes.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['item_a', 'item_b', 'y', 'rater'])
    d = v.merge(s, on='sample_id', suffixes=('', '_s'))
    key = {}
    for i, r in items.iterrows():
        key.setdefault(r.rel, []).append((r.start, i))
    def find(path, start):
        rel = str(path).replace('/root/voice_AI/', '')
        c = [(abs(s0 - start), i) for s0, i in key.get(rel, []) if abs(s0 - start) <= 5 - OVERLAP]
        return min(c)[1] if c else -1
    rows = []
    for r in d.itertuples():
        a, b = find(r.a_path, r.a_start), find(r.b_path, r.b_start)
        rows.append(dict(item_a=a, item_b=b, a_path=r.a_path, a_start=r.a_start, b_path=r.b_path, b_start=r.b_start,
                         y={'a': 1.0, 'b': 0.0}.get(r.choice, 0.5), rater=r.author_id, pair_kind=r.pair_kind))
    return pd.DataFrame(rows)


def cis_anchors(items):
    links, lab = DATA / 'cis_links.csv', work('cislabel', 'labels.csv')
    a = np.zeros(len(items))
    if links.exists() and lab.exists():
        L = pd.read_csv(lab, keep_default_na=False).set_index('speaker').label
        c = pd.read_csv(links)
        c['label'] = c.speaker.map(L)
        sign = dict(zip(c.rel, c.label.map({'f': 1, 'm': -1}).fillna(0)))
        a = items.rel.map(sign).fillna(0).values
    return a


def main(expert_boost=1.0):
    R = DATA / 'ratings'
    expert, votes, seed, gv = load_votes(R)
    gold = pd.DataFrame(json.load(open(DATA / 'sys_new/gold_v1_catalog.json'))['items'])
    items, vi, by_file = build_items(votes, seed, gold)
    votes['item'] = vi
    sv = [(i, sc, 1 / len(by_file[r])) for r, sc in zip(seed.rel, seed.score) for i in by_file[r]]
    obs = pd.DataFrame({'item': np.r_[votes.item.values, [x[0] for x in sv]].astype(int),
                        'score': np.r_[votes.score.values, [x[1] for x in sv]].astype(float),
                        'rater': np.r_[votes.author_id.values, np.full(len(sv), -1)].astype(int),
                        'w0': np.r_[np.ones(len(votes)), [x[2] for x in sv]]})
    rep = gv[gv.author_id == expert].groupby('item_id').score.agg(['std', 'count'])
    retest = float(np.sqrt(np.mean(rep['std'][rep['count'] >= 2] ** 2)))
    anchors = cis_anchors(items)
    z, zsd, H, S2, n_it = consensus(expert, obs, len(items), retest, expert_boost, anchors=anchors)
    items['cis_anchor'] = anchors.astype(int)
    items['z'] = z.round(2); items['z_sd'] = zsd.round(2)
    cnt = obs[obs.rater != -1].groupby('item').rater
    items['n_listens'] = cnt.count().reindex(range(len(items))).fillna(0).astype(int).values
    with_expert = set(obs.item[obs.rater == expert])
    items['has_expert'] = [i in with_expert for i in range(len(items))]
    items['has_seed'] = items.rel.isin(set(seed.rel))
    nv = pd.read_csv(R / 'model_training_novoice_reports.csv')
    bad = {(_r, s) for _r, s in zip(_rel(nv.audio_path.fillna('')), nv.segment_start.fillna(-1))}
    items['novoice_reports'] = [int(any(r == br and abs(s - bs) < 5 - OVERLAP for br, bs in bad)) for r, s in zip(items.rel, items.start)]
    items.index.name = 'item'
    items.to_csv(work('target', 'items.csv'))
    rep_all = gv.groupby(['author_id', 'item_id']).score.agg(['std', 'count'])
    trt = rep_all[rep_all['count'] >= 2].groupby('author_id')['std'].apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
    rows = [dict(rater='seed' if r == -1 else ('expert' if r == expert else str(r)),
                 votes=int((obs.rater == r).sum()), noise_sd=round(float(np.sqrt(S2[r])), 1),
                 retest_sd=round(float(trt.get(r, np.nan)), 1),
                 **{f'h({int(k)})': round(float(x), 1) for k, x in zip(K, H[r])}) for r in H]
    pd.DataFrame(rows).sort_values('votes', ascending=False).to_csv(work('target', 'raters.csv'), index=False)
    pairs = pass_pairs(R, items)
    pairs.to_csv(work('target', 'pairs.csv'), index=False)
    for v, n in ((1, 'cis f'), (-1, 'cis m')):
        k = (items.cis_anchor == v) & (items.n_listens > 0)
        if k.any():
            print(f'{n}: {int(k.sum())} rated items, z median {items.z[k].median():.1f}')
    e = H[expert]
    print('expert curve h(z) at 0,20,..,100:', [round(float(x)) for x in e[::2]])
    print(f'consensus: {n_it} iterations, {len(items)} items ({int((items.n_listens > 0).sum())} listened, '
          f'{int(items.gold.notna().sum())} gold), expert retest sd {retest:.1f}, pair votes {len(pairs)}')


if __name__ == '__main__':
    main()
