"""Paths and constants. Everything data-related lives under VPM_DATA (default ~/data); nothing private is in this repo.

Layout under VPM_DATA:
  voice_AI/            our recordings (dataset/, new_voices/, additional_data/russian/)
  ratings/             CSV export of the rating tables (model_training_*, pass_pair_*)
  sys_new/gold_v1_catalog.json
  tidyvoice/           TidyVoiceX_ASV, extracted
  cis/<source>/<speaker>/<file>   cis voices: own Telegram set and official corpora
  work/                everything the pipeline writes (snapshots, caches, runs)
"""
import os
from pathlib import Path

DATA = Path(os.environ.get('VPM_DATA', Path.home() / 'data'))
WORK = Path(os.environ.get('VPM_WORK', DATA / 'work'))
SR = 16000
WIN = 5.0            # seconds per scored window, as in the bot
BLOCK = 0.5          # voice check resolution, seconds
MIN_WIN = 2.0        # the bot drops a trailing window shorter than this

# sources: name -> (root relative to DATA, glob, language or None)
SOURCES = {
    'dataset': ('voice_AI/dataset', '*.wav', 'ru'),
    'new_voices': ('voice_AI/new_voices', '**/*.wav', 'ru'),
    'additional_ru': ('voice_AI/additional_data/russian', '*.wav', 'ru'),
}


def work(*parts):
    p = WORK.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
