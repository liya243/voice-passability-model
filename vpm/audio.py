"""Audio loading and the bot's windowing. Any format ffmpeg reads (wav, opus, flac, mp3)."""
import subprocess
import numpy as np
import soundfile as sf
from .config import SR, WIN, MIN_WIN


def load(path, start=0.0, dur=None):
    """Mono float32 at 16 kHz. soundfile for wav/flac, ffmpeg for the rest."""
    path = str(path)
    try:
        info = sf.info(path)
        frames = -1 if dur is None else int(round(dur * info.samplerate))
        x, sr = sf.read(path, start=int(round(start * info.samplerate)), frames=frames, dtype='float32', always_2d=True)
        x = x.mean(1)
        if sr != SR:
            import torch, torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
        return x
    except sf.LibsndfileError:
        cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-ss', str(start), '-i', path]
        if dur is not None:
            cmd += ['-t', str(dur)]
        cmd += ['-f', 'f32le', '-ac', '1', '-ar', str(SR), '-']
        return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32).copy()


def duration(path):
    try:
        return sf.info(str(path)).duration
    except sf.LibsndfileError:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                             capture_output=True, text=True).stdout.strip()
        return float(out) if out else 0.0


def window_starts(dur):
    """Start times (s) of the bot's non-overlapping 5 s windows; the last one is kept if >= MIN_WIN."""
    starts, s = [], 0.0
    while dur - s >= MIN_WIN:
        starts.append(s)
        s += WIN
    return starts


def cut(x, start, length=WIN, peak_norm=True):
    """Exact window as the model sees it: zero-padded to 5 s, peak-normalised per window."""
    n = int(length * SR)
    w = x[int(round(start * SR)):][:n]
    w = np.pad(w, (0, n - len(w)))
    if peak_norm:
        p = np.abs(w).max()
        if p > 0:
            w = w / p
    return w.astype(np.float32)
