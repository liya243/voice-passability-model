"""Export a trained run to ONNX for the bot's CPU worker, then quantize to uint8 like the current model.

Input: log-mel features [B, 80, 500] of one 5 s window (Whisper's own feature extractor, computed by the worker);
outputs: score 0..100 and p_voice. Checks the uint8 model against torch on the gold windows.

Usage: python -m vpm.export work/runs/ws5_s0  ->  <run>/model_fp32.onnx, model_uint8.onnx, export_check.json
"""
import json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .config import DATA, work
from .audio import load, cut
from .model import load_run


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, mel):
        e = self.m.bb
        h = F.gelu(e.conv1(mel)); h = F.gelu(e.conv2(h)).permute(0, 2, 1)
        h = h + e.embed_positions.weight[:h.shape[1]]
        hs = [h]
        for layer in e.layers:
            o = layer(h, None); h = o[0] if isinstance(o, tuple) else o
            hs.append(h)
        hs[-1] = e.layer_norm(h)
        m = self.m
        h = (torch.stack(hs, 0) * torch.softmax(m.mix, 0)[:, None, None, None]).sum(0)
        a = torch.softmax(m.att(h).squeeze(-1), 1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-6).sqrt()
        emb = m.emb(torch.cat([mu, sd], -1))
        return torch.sigmoid(m.out(emb).squeeze(-1)) * 100, torch.sigmoid(m.voice(emb).squeeze(-1))


def main(run):
    run = DATA / run
    m = load_run(run).eval()
    assert m.whisper, 'export is written for the Whisper path'
    w = Wrap(m).eval()
    dummy = torch.zeros(1, 80, 500)
    fp32, u8 = run / 'model_fp32.onnx', run / 'model_uint8.onnx'
    torch.onnx.export(w, dummy, fp32, input_names=['mel'], output_names=['score', 'p_voice'],
                      dynamic_axes={'mel': {0: 'b'}, 'score': {0: 'b'}, 'p_voice': {0: 'b'}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(fp32), str(u8), op_types_to_quantize=['MatMul'], weight_type=QuantType.QUInt8)
    import onnxruntime as ort
    s = ort.InferenceSession(str(u8), providers=['CPUExecutionProvider'])
    win = pd.read_csv(work('scorecard', 'windows.csv'), index_col='w')
    g = win[win.set == 'gold']
    X = np.stack([cut(load(DATA / r.path), r.start) for r in g.itertuples()])
    mel = m.fe(list(X), sampling_rate=16000, return_tensors='np').input_features[:, :, :500].astype(np.float32)
    with torch.no_grad():
        pt = w(torch.from_numpy(mel))[0].numpy()
    on = np.concatenate([s.run(None, {'mel': mel[i:i + 1]})[0] for i in range(len(mel))])
    d = np.abs(on - pt)
    res = dict(windows=len(d), mean_abs_diff=float(d.mean()), p99_abs_diff=float(np.quantile(d, 0.99)), max_abs_diff=float(d.max()),
               bytes_uint8=u8.stat().st_size)
    json.dump(res, open(run / 'export_check.json', 'w'), indent=1)
    print(res)


if __name__ == '__main__':
    main(sys.argv[1])
