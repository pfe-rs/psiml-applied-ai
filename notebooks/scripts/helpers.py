from typing import Optional

from dataclasses import dataclass
from functools import lru_cache

import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


OPUS_MODELS = {
    ("sr", "en"): "Helsinki-NLP/opus-mt-sr-en",
    ("en", "sr"): "Helsinki-NLP/opus-mt-en-sr",
}

def ensure_wav_16k_mono(in_path: str, out_path: str) -> str:
    wav, sr = librosa.load(in_path, sr=16000, mono=True)
    sf.write(out_path, wav, 16000)
    return out_path


@dataclass
class PipelineConfig:
    asr_model_id: str = "openai/whisper-small"  # change to whisper-base for extra speed
    asr_force_lang: Optional[str] = None        # e.g., "sr" to force Serbian
    nmt_src: str = "sr"
    nmt_tgt: str = "en"
    tts_lang: str = "en"

def validate_pair(src: str, tgt: str) -> str:
    key = (src, tgt)
    if key not in OPUS_MODELS:
        raise ValueError(f"Unsupported translation pair {src}->{tgt}. "
                         f"Supported pairs: {sorted(OPUS_MODELS.keys())}")
    return OPUS_MODELS[key]


@lru_cache(maxsize=2)
def load_opus(model_id: str, device: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    return tok, mdl
