"""Voice weight v2 for one recording: python weight/predict_weight.py recording.wav
Needs: pip install numpy scipy soundfile praat-parselmouth"""
import argparse, json, sys
from pathlib import Path
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_v2 import WeightModelV2

p = argparse.ArgumentParser()
p.add_argument('audio')
p.add_argument('--bundle', default=str(Path(__file__).resolve().parent / 'weight_v2_20261121.json'))
p.add_argument('--max-blocks', type=int, default=240, help='0.5 s blocks to score (240 = 2 minutes)')
a = p.parse_args()
audio, rate = sf.read(a.audio, dtype='float32', always_2d=True)
r = WeightModelV2(a.bundle).predict(audio, rate, a.max_blocks)
if not r['has_voice']:
    sys.exit('No voiced 0.5 s block found')
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != 'blocks'}, ensure_ascii=False))
