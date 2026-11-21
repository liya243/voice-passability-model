"""Voice weight v2: Davidson ranking model on 15 Praat features of 0.5 s blocks, trained on expert pair votes.
Percent = position of the recording's median block score among dataset voices (each ECAPA voice counted once).
Used by the bot (/weight) and the miniapp live weight. Bundle is plain JSON (no pickle)."""
import io, json, math
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import resample_poly

RATE = 16000
BLOCK = 8000
FEATURES = ['f0_median', 'f0_sd_st', 'voiced_frac', 'F1', 'F2', 'F3', 'F4', 'hnr', 'cpp', 'h1_h2', 'h1_a3',
            'alpha_ratio_db', 'hammarberg_db', 'ltas_slope_db_per_khz', 'centroid_hz']


def encode_block(wave):
    """Exactly the miniapp weight-training preprocessing (mono16k-rms012-peak095-fade3ms-v1) for 16 kHz input."""
    w = np.asarray(wave, dtype='float32').copy()
    w -= w.mean()
    rms = float(np.sqrt(np.mean(w * w)))
    if rms < .003:
        return None
    peak = float(np.max(np.abs(w)))
    w *= min(.12 / rms, .95 / peak)
    f = round(.003 * RATE)
    r = np.linspace(0, 1, f, dtype='float32')
    w[:f] *= r
    w[-f:] *= r[::-1]
    return (np.round(np.clip(w, -1, 1) * 32767) / 32768).astype('float32')   # PCM16 round trip as served


def _band(sdb, fr, lo, hi):
    m = (fr >= lo) & (fr < hi)
    return sdb[m]


def block_features(x):
    x = np.asarray(x, dtype=np.float64)
    r = {}
    snd = parselmouth.Sound(x, sampling_frequency=RATE)
    pitch = snd.to_pitch_ac(time_step=0.005, pitch_floor=60, pitch_ceiling=700)
    f0 = pitch.selected_array['frequency']
    v = f0 > 0
    r['voiced_frac'] = float(v.mean())
    if v.sum() < 5:
        return None
    r.update(f0_median=float(np.median(f0[v])), f0_sd_st=float(np.std(12 * np.log2(f0[v] / 100))))
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5500)
    ts = pitch.xs()[v]
    for k in (1, 2, 3, 4):
        vals = np.array([fm.get_value_at_time(k, t) for t in ts])
        vals = vals[np.isfinite(vals)]
        r[f'F{k}'] = float(np.median(vals)) if len(vals) else np.nan
    r['hnr'] = float(call(snd.to_harmonicity_cc(0.01, 60, 0.1, 1.0), 'Get mean', 0, 0))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 16384)) + 1e-12
    fr = np.fft.rfftfreq(16384, 1 / RATE)
    sdb = 20 * np.log10(spec)
    f0m = r['f0_median']
    h = lambda f: float(_band(sdb, fr, f * 0.9, f * 1.1).max())
    r['h1_h2'] = h(f0m) - h(2 * f0m)
    r['h1_a3'] = h(f0m) - float(_band(sdb, fr, r['F3'] * 0.9, r['F3'] * 1.1).max()) if np.isfinite(r.get('F3', np.nan)) else np.nan
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-20
    fr = np.fft.rfftfreq(len(x), 1 / RATE)
    e = lambda lo, hi: float(spec[(fr >= lo) & (fr < hi)].sum())
    r['alpha_ratio_db'] = 10 * np.log10(e(1000, 5000) / e(50, 1000))
    sdb = 10 * np.log10(spec)
    r['hammarberg_db'] = float(sdb[(fr >= 0) & (fr < 2000)].max() - sdb[(fr >= 2000) & (fr < 5000)].max())
    m = (fr >= 100) & (fr <= 4000)
    r['ltas_slope_db_per_khz'] = float(np.polyfit(fr[m] / 1000, sdb[m], 1)[0])
    r['centroid_hz'] = float((fr * spec).sum() / spec.sum())
    r['cpp'] = float(call(call(snd, 'To PowerCepstrogram', 60, 0.002, 5000, 50), 'Get CPPS', 'no', 0.01, 0.001, 60, 330, 0.05,
                          'parabolic', 0.001, 0, 'Straight', 'Robust'))
    return r


def to_mono16k(audio, sample_rate):
    a = np.asarray(audio, dtype='float32')
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sample_rate != RATE:
        d = math.gcd(int(sample_rate), RATE)
        a = resample_poly(a, RATE // d, int(sample_rate) // d).astype('float32')
    return a


class WeightModelV2:
    def __init__(self, path):
        b = json.load(open(path))
        assert b['format'] == 'weight-davidson-acoustic-v2' and b['features'] == FEATURES
        self.bundle = b
        self.mean = np.array(b['mean'])
        self.std = np.array(b['std'])
        self.fill = np.array(b['fill'])
        self.w = np.array(b['w'])
        self.q_scores = np.array(b['percent_table']['scores'])
        self.q_pct = np.array(b['percent_table']['percent'])
        self.min_voiced = float(b['min_voiced_frac'])
        self.version = b['version']

    def block_score(self, feats):
        x = np.array([feats.get(k, np.nan) for k in FEATURES], dtype=np.float64)
        x[0] = math.log(x[0])
        x = np.where(np.isfinite(x), x, self.fill)
        return float(((x - self.mean) / self.std) @ self.w)

    def percent(self, score):
        return float(np.clip(np.interp(score, self.q_scores, self.q_pct), 0, 100))

    def blocks(self, audio16k, max_blocks=None):
        """[(start_sec, score)] for every voiced 0.5 s block."""
        out = []
        n = len(audio16k) // BLOCK
        for k in range(n if max_blocks is None else min(n, max_blocks)):
            enc = encode_block(audio16k[k * BLOCK:(k + 1) * BLOCK])
            if enc is None:
                continue
            f = block_features(enc)
            if f is None or f['voiced_frac'] < self.min_voiced:
                continue
            out.append((k * 0.5, self.block_score(f)))
        return out

    def predict(self, audio, sample_rate, max_blocks=240):
        a = to_mono16k(audio, sample_rate)
        bl = self.blocks(a, max_blocks)
        if not bl:
            return dict(has_voice=False, percent=None, blocks=[], version=self.version)
        s = np.array([x[1] for x in bl])
        med = float(np.median(s))
        pcts = [self.percent(x) for x in s]
        return dict(has_voice=True, percent=self.percent(med), raw_score=med, block_count=len(bl),
                    percent_p25=float(np.percentile(pcts, 25)), percent_p75=float(np.percentile(pcts, 75)),
                    blocks=[(t, p) for (t, _), p in zip(bl, pcts)], version=self.version)
