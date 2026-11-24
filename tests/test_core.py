import numpy as np
import pandas as pd
import torch
from vpm.audio import window_starts, cut
from vpm.target import fit_curve, consensus, K
from vpm import loss as L


def test_windows_match_bot():
    assert window_starts(12.0) == [0.0, 5.0, 10.0]
    assert window_starts(11.9) == [0.0, 5.0]
    assert window_starts(1.5) == []
    w = cut(np.ones(16000 * 3, np.float32) * 0.5, 0.0)
    assert len(w) == 80000 and w.max() == 1.0 and w[-1] == 0


def test_curve_is_monotone():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 100, 300); s = np.clip(30 + 0.5 * z + rng.normal(0, 10, 300), 0, 100)
    h = fit_curve(z, s, np.ones_like(z))
    assert np.all(np.diff(h) >= -1e-9)
    assert abs(np.interp(50, K, h) - 55) < 5


def test_consensus_recovers_order_through_a_nonlinear_rater():
    rng = np.random.default_rng(1)
    true = rng.uniform(0, 100, 200); rows = []
    for i, t in enumerate(true):
        rows.append((i, t + rng.normal(0, 6), 1))                        # expert
        rows.append((i, 100 * (t / 100) ** 2 + rng.normal(0, 6), 2))     # rater with a squashed bottom
    obs = pd.DataFrame(rows, columns=['item', 'score', 'rater']); obs['w0'] = 1.0
    z, zsd, H, S2, _ = consensus(1, obs, len(true), retest=6.0)
    assert np.corrcoef(z, true)[0, 1] > 0.97
    assert np.interp(50, K, H[2]) < 35                                    # the curve learned the squash


def test_losses():
    p = torch.tensor([80.0, 20.0]); z = torch.tensor([80.0, 20.0]); sd = torch.tensor([3.0, 3.0])
    assert L.scale(p, z, sd) < 1e-6
    assert L.order_from_consensus(p, z, sd) < L.order_from_consensus(p.flip(0), z, sd)
    assert L.same_voice(torch.tensor([50.0]), torch.tensor([54.0])) == 0
    assert L.same_voice(torch.tensor([50.0]), torch.tensor([70.0])) > 0
    assert L.cis(torch.tensor([90.0, 10.0]), torch.tensor([1, 0])) == 0
