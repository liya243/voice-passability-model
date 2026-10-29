"""Production-compatible peak normalization and actual-length tail filtering."""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

def load_audio(path: str | Path) -> np.ndarray:
    audio,sr=sf.read(path,dtype='float32',always_2d=True)
    audio=audio.mean(axis=1)
    if sr!=16000:audio=signal.resample(audio,int(len(audio)*16000/sr))
    audio=np.nan_to_num(audio).astype(np.float32)
    if not len(audio):raise ValueError('Empty recording')
    peak=float(np.abs(audio).max())
    return audio/peak if peak>0 else audio

def windows(audio: np.ndarray) -> list[np.ndarray]:
    out=[];fallback=None
    for start in range(0,len(audio),80000):
        clip=audio[start:start+80000]
        if len(clip)<48000:break
        if len(clip)<80000:clip=np.pad(clip,(0,80000-len(clip)))
        if fallback is None:fallback=clip
        if float(np.abs(clip).mean())>=1e-4:out.append(clip)
    if not out and fallback is not None:out=[fallback]
    return out
