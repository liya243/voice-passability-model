"""Score the scorecard windows with a model.

Models are adapters with .score(batch: float32 [B, 80000]) -> 0..100. Add new ones to MODELS.
  legacy:<checkpoint.pt>   the bot's HubertRegressor (code from voice_AI/train_scripts), e.g. ft3
  onnx:<model.onnx>        an exported model as the worker runs it
  pipeline:<run dir>       a model trained by vpm.train
Optional --calib <json with x, y>: the monotone output map the bot applies after the model.

Usage: python -m vpm.predict --model legacy:model_archive_20261121/runs/ft3/best.pt --calib ... --name ft3_half
"""
import argparse, json, sys
import numpy as np
import pandas as pd
import torch
from .config import DATA, SR, WIN, work
from .audio import load, cut

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class Legacy:
    def __init__(self, ckpt):
        sys.path.insert(0, str(DATA / 'voice_AI'))
        from train_scripts.model import HubertRegressor
        self.m = HubertRegressor('facebook/hubert-base-ls960', num_bins=20, dropout=0.1)
        self.m.load_state_dict(torch.load(DATA / ckpt, map_location='cpu', weights_only=False)['model'], strict=False)
        self.m.to(dev).eval()

    @torch.no_grad()
    def score(self, x):
        x = torch.from_numpy(x).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            return self.m(x, torch.ones_like(x, dtype=torch.bool))['reg_probs'].squeeze(-1).float().cpu().numpy() * 100


class Onnx:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(DATA / path), providers=['CPUExecutionProvider'])
        self.i = self.s.get_inputs()[0].name

    def score(self, x):
        return np.concatenate([np.asarray(self.s.run(None, {self.i: x[k:k + 1]})[0]).reshape(-1) for k in range(len(x))]) * 100


class Pipeline:
    def __init__(self, run):
        from .model import load_run
        self.m = load_run(DATA / run).to(dev).eval()

    @torch.no_grad()
    def score(self, x):
        with torch.autocast(dev, dtype=torch.bfloat16):
            return self.m(torch.from_numpy(x).to(dev)).float().cpu().numpy()


MODELS = {'legacy': Legacy, 'onnx': Onnx, 'pipeline': Pipeline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--calib'); ap.add_argument('--name', required=True)
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    kind, arg = a.model.split(':', 1)
    m = MODELS[kind](arg)
    cal = json.load(open(DATA / a.calib)) if a.calib else None
    w = pd.read_csv(work('scorecard', 'windows.csv'), index_col='w')
    pred = np.full(len(w), np.nan)
    for path, g in w.groupby('path'):
        try:
            x = load(DATA / path)
        except Exception:
            continue
        W = np.stack([cut(x, s) for s in g.start])
        for k in range(0, len(W), a.batch):
            pred[g.index[k:k + a.batch]] = m.score(W[k:k + a.batch])
    if cal:
        pred = np.interp(pred, cal['x'], cal['y'])
    pd.DataFrame({'w': w.index, 'pred': pred.round(3)}).to_csv(work('preds', a.name + '.csv'), index=False)
    print(a.name, 'scored', int(np.isfinite(pred).sum()), 'of', len(w))


if __name__ == '__main__':
    main()
