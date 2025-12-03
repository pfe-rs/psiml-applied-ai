import os
import sys
os.environ.pop("MPLBACKEND", None)

# ---- PyTorch 2.6 safe unpickle patch for XTTS v2 ----
import torch
from torch.serialization import add_safe_globals

from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig

add_safe_globals([
    XttsConfig,
    XttsArgs,
    XttsAudioConfig,
    BaseDatasetConfig,
    BaseAudioConfig,
])
# ------------------------------------------------------

from TTS.api import TTS

# --------------------------------------
# Minimal CLI argument parsing (sys.argv)
# --------------------------------------
#
# Usage:
#   python xtts.py speaker.wav output.wav "Hello world" en
#
# Defaults:

DEFAULT_SPEAKER = "/content/psiml-applied-ai/notebooks/wavs/audio_2.wav"
DEFAULT_OUTPUT = "out.wav"
DEFAULT_TEXT = "Hello from XTTS!"
DEFAULT_LANGUAGE = "en"
DEFAULT_DEVICE = 'cpu'

speaker_wav = DEFAULT_SPEAKER
output_wav = DEFAULT_OUTPUT
text = DEFAULT_TEXT
language = DEFAULT_LANGUAGE
device = DEFAULT_DEVICE

if len(sys.argv) > 1 and sys.argv[1] != "":
    speaker_wav = sys.argv[1]

if len(sys.argv) > 2 and sys.argv[2] != "":
    output_wav = sys.argv[2]

if len(sys.argv) > 3 and sys.argv[3] != "":
    text = sys.argv[3]

if len(sys.argv) > 4 and sys.argv[4] != "":
    language = sys.argv[4]
    
if len(sys.argv) > 5 and sys.argv[5] != "":
    device = sys.argv[5]

print("=== XTTS CLI ===")
print("Speaker WAV:", speaker_wav)
print("Output WAV:", output_wav)
print("Text:", text)
print("Language:", language)
print("=================")

# --------------------------------------
# Load model
# --------------------------------------
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

tts = TTS(model_name)
tts.to(device)

print("XTTS loaded!")

# --------------------------------------
# Run XTTS
# --------------------------------------
tts.tts_to_file(
    text=text,
    file_path=output_wav,
    speaker_wav=speaker_wav,
    language=language
)

print("Saved:", output_wav)


