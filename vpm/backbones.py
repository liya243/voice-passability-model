"""Pretrained backbones behind one interface: 5 s windows in, per-layer pooled features out.

Every backbone gives, for each hidden layer, the time-mean and time-std of its frames -> [B, L, 2D].
Add a model by adding a line to BACKBONES; the probe, fine-tuning and scorecard need nothing else.
"""
import numpy as np
import torch

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

#  name: (kind, hub id, note)
BACKBONES = {
    'hubert_base': ('hf', 'facebook/hubert-base-ls960', 'the bot today (ft3 starts from it); English'),
    'mhubert147': ('hf', 'utter-project/mHuBERT-147', 'HuBERT trained on 147 languages'),
    'xlsr300': ('hf', 'facebook/wav2vec2-xls-r-300m', '128 languages'),
    'wavlm_base_plus': ('hf', 'microsoft/wavlm-base-plus', 'speaker and emotion benchmarks; English'),
    'wavlm_large': ('hf', 'microsoft/wavlm-large', 'speaker and emotion benchmarks; English'),
    'whisper_small': ('whisper', 'openai/whisper-small', 'encoder only; multilingual, robust to noise'),
    'ecapa': ('speechbrain', 'speechbrain/spkrec-ecapa-voxceleb', 'speaker embedding; one "layer"'),
}


class HF:
    def __init__(self, hub):
        from transformers import AutoModel, AutoFeatureExtractor
        self.fe = AutoFeatureExtractor.from_pretrained(hub)
        self.m = AutoModel.from_pretrained(hub).to(dev).eval()

    @torch.no_grad()
    def __call__(self, W):
        x = torch.from_numpy(np.stack(W)).to(dev)
        if getattr(self.fe, 'do_normalize', False):
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
        with torch.autocast(dev, dtype=torch.bfloat16):
            hs = self.m(x, output_hidden_states=True).hidden_states
        h = torch.stack(hs, 1).float()                      # B, L, T, D
        return torch.cat([h.mean(2), h.std(2)], -1).cpu().numpy()


class Whisper:
    def __init__(self, hub):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(hub)
        self.m = WhisperModel.from_pretrained(hub).encoder.to(dev).eval()

    @torch.no_grad()
    def __call__(self, W):
        f = self.fe(list(W), sampling_rate=16000, return_tensors='pt').input_features.to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            hs = self.m(f, output_hidden_states=True).hidden_states
        h = torch.stack(hs, 1).float()[:, :, :250]          # 30 s input; the 5 s window is the first 250 frames
        return torch.cat([h.mean(2), h.std(2)], -1).cpu().numpy()


class Ecapa:
    def __init__(self, hub):
        from speechbrain.inference.speaker import EncoderClassifier
        self.m = EncoderClassifier.from_hparams(source=hub, run_opts={'device': dev})

    @torch.no_grad()
    def __call__(self, W):
        e = self.m.encode_batch(torch.from_numpy(np.stack(W)).to(dev)).squeeze(1).float().cpu().numpy()
        return e[:, None, :]                                 # B, 1, 192


def load(name):
    kind, hub, _ = BACKBONES[name]
    return {'hf': HF, 'whisper': Whisper, 'speechbrain': Ecapa}[kind](hub)
