"""Adapter around the faster-whisper model bundled with the competition package."""
from __future__ import annotations

from pathlib import Path
import numpy as np

TARGET_SAMPLE_RATE = 16_000
_MODEL_CACHE: dict[tuple[str, str, str, int], object] = {}

def load_model(
    model_path: str,
    backend: str = 'cpu',
    compute_type: str = 'int8',
    cpu_threads: int = 4,
):
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f'faster-whisper model directory does not exist: {path}')
    cpu_threads = max(1, int(cpu_threads))
    key = (str(path), backend, compute_type, cpu_threads)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError('Python package `faster-whisper` is not installed') from exc
    model = WhisperModel(
        str(path),
        device=backend,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    _MODEL_CACHE[key] = model
    return model

def _resample_linear(audio: np.ndarray, source_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == TARGET_SAMPLE_RATE or audio.size == 0:
        return audio
    target_length = max(1, round(audio.size * TARGET_SAMPLE_RATE / source_rate))
    old_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)

def transcribe_pcm(
    model,
    pcm: bytes,
    sample_rate: int,
    beam_size: int = 1,
) -> str:
    if not pcm or len(pcm) < 2:
        return ''
    sample_bytes = len(pcm) - (len(pcm) % 2)
    audio = np.frombuffer(pcm[:sample_bytes], dtype='<i2').astype(np.float32)
    audio /= 32768.0
    audio = _resample_linear(audio, sample_rate)
    segments, _ = model.transcribe(
        audio,
        language='zh',
        task='transcribe',
        beam_size=max(1, int(beam_size)),
        temperature=0.0,
        vad_filter=True,
        vad_parameters={'min_silence_duration_ms': 350},
        condition_on_previous_text=False,
        initial_prompt=(
            '以下是机器人比赛中的普通话指令：请引领到目标点A、'
            '请引领到目标点B、请引领到目标点C、开始识别、'
            '请寻找指定目标、是否可以通行。'
        ),
    )
    return ''.join(segment.text for segment in segments).strip()
