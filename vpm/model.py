"""Pass model = any HF backbone + learned mix of its layers + attentive statistics pooling + small head -> 0..100."""
import json
from pathlib import Path
import torch
import torch.nn as nn
from .backbones import BACKBONES


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class PassModel(nn.Module):
    def __init__(self, backbone, n_languages=0, hidden=256):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        self.cfg = dict(backbone=backbone, n_languages=n_languages, hidden=hidden)
        hub = BACKBONES[backbone][1]
        self.bb = AutoModel.from_pretrained(hub)
        c = AutoConfig.from_pretrained(hub)
        L, D = c.num_hidden_layers + 1, c.hidden_size
        self.norm_input = bool(getattr(c, 'feat_extract_norm', 'layer') == 'layer')
        self.mix = nn.Parameter(torch.zeros(L))
        self.att = nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 1))
        self.emb = nn.Sequential(nn.LayerNorm(2 * D), nn.Linear(2 * D, hidden), nn.GELU(), nn.Dropout(0.1))
        self.out = nn.Linear(hidden, 1)
        self.lang = nn.Linear(hidden, n_languages) if n_languages else None

    def embed(self, x):
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
        hs = self.bb(x, output_hidden_states=True).hidden_states
        h = (torch.stack(hs, 0) * torch.softmax(self.mix, 0)[:, None, None, None]).sum(0)   # B, T, D
        a = torch.softmax(self.att(h).squeeze(-1), 1)[..., None]
        mu = (a * h).sum(1)
        sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-6).sqrt()
        return self.emb(torch.cat([mu, sd], -1))

    def forward(self, x, with_lang=False, lam=1.0):
        e = self.embed(x)
        score = torch.sigmoid(self.out(e).squeeze(-1)) * 100
        if with_lang and self.lang is not None:
            return score, self.lang(GradReverse.apply(e, lam))
        return score

    def unfreeze_top(self, n):
        for p in self.bb.parameters():
            p.requires_grad = False
        layers = self.bb.encoder.layers
        for l in layers[len(layers) - n:] if n else []:
            for p in l.parameters():
                p.requires_grad = True

    def save(self, run, **extra):
        run = Path(run); run.mkdir(parents=True, exist_ok=True)
        json.dump({**self.cfg, **extra}, open(run / 'config.json', 'w'), indent=1)
        torch.save(self.state_dict(), run / 'model.pt')


def load_run(run):
    run = Path(run)
    c = json.load(open(run / 'config.json'))
    m = PassModel(c['backbone'], c.get('n_languages', 0), c.get('hidden', 256))
    m.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
    return m
