"""The four parts of the training loss. All take scores on the 0..100 scale.

  scale        Huber to the consensus z, weight 1/(z_sd^2 + 16): keeps the model on the expert's scale
  order        RankNet over pairs: within the labelled batch, target P(i > j) from the consensus and its sd;
               plus the miniapp pair votes (y = 1, 0, 0.5)
  consistency  two voiced windows of one recording: penalise |p_a - p_b| beyond DELTA (same voice, same score)
  language     one TidyVoice speaker in two languages: the same hinge; optionally a language classifier behind
               gradient reversal, so the embedding carries no language
  cis          cis women at least CIS_F_FLOOR, cis men at most CIS_M_CEIL (hinge; labels from metadata or cislabel)
"""
import torch
import torch.nn.functional as F

DELTA = 5.0
TAU = 10.0            # points per logit in the RankNet probability
CIS_F_FLOOR = 90.0
CIS_M_CEIL = 20.0


def scale(p, z, zsd):
    w = 1 / (zsd ** 2 + 16); w = w / w.mean()
    return (F.smooth_l1_loss(p / 100, z / 100, beta=0.05, reduction='none') * w).mean()


def order_from_consensus(p, z, zsd):
    i, j = torch.triu_indices(len(p), len(p), 1, device=p.device)
    s = torch.sqrt(zsd[i] ** 2 + zsd[j] ** 2 + 1.0)
    target = 0.5 * (1 + torch.erf((z[i] - z[j]) / (s * 2 ** 0.5)))
    keep = (target - 0.5).abs() > 0.1          # pairs whose order the consensus actually knows
    if keep.sum() == 0:
        return p.sum() * 0
    return F.binary_cross_entropy_with_logits(((p[i] - p[j]) / TAU)[keep], target[keep])


def order_from_votes(pa, pb, y):
    return F.binary_cross_entropy_with_logits((pa - pb) / TAU, y)


def _hinge(excess):
    """Points beyond a limit -> the same Huber scale as `scale` (so 10 points over costs about as much as 10 points of error)."""
    e = F.relu(excess) / 100
    return F.smooth_l1_loss(e, torch.zeros_like(e), beta=0.05)


def same_voice(pa, pb, delta=DELTA):
    return _hinge((pa - pb).abs() - delta)


def cis(p, label):
    f = label == 1; m = label == 0
    lf = _hinge(CIS_F_FLOOR - p[f]) if f.any() else p.sum() * 0
    lm = _hinge(p[m] - CIS_M_CEIL) if m.any() else p.sum() * 0
    return lf + lm
