"""Offline ECAPA regressor. No pretrained files or network downloads are needed."""
import torch
from torch import nn
import torchaudio

# SpeechBrain 1.0.3 checks these removed torchaudio APIs at import time.
# This model uses tensors only; audio decoding is handled by soundfile.
for name,fn in [('list_audio_backends',lambda:['soundfile']),
                ('get_audio_backend',lambda:'soundfile'),('set_audio_backend',lambda _:None)]:
    if not hasattr(torchaudio,name):setattr(torchaudio,name,fn)
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class EcapaRegressor(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.frontend=Fbank(n_mels=80)
        self.normalization=InputNormalization(norm_type='sentence',std_norm=False)
        self.encoder=ECAPA_TDNN(input_size=80,channels=[width]*4+[width*3],
            attention_channels=128 if width==1024 else 64,
            se_channels=128 if width==1024 else 64,lin_neurons=192)
        self.head=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,128),nn.GELU(),
                                nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,wave):
        lengths=torch.ones(len(wave),device=wave.device)
        features=self.normalization(self.frontend(wave.float()),lengths)
        encoded=self.encoder(features,lengths).squeeze(1)
        return self.head(encoded).squeeze(-1).sigmoid()
