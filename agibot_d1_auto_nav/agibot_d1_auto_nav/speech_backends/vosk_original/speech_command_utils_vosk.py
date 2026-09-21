"""Original Vosk streaming recognition implementation kept as a selectable backup."""
from __future__ import annotations

import json
import subprocess
import time

from agibot_d1_auto_nav.speech_command_utils import (
    build_arecord_command,
    device_id_to_arecord_device,
    downmix_pcm_to_mono,
    ensure_arecord_available,
    maximize_capture_volume,
    measure_environment_noise,
    pcm_rms,
    recognized_text_matches,
    validate_model_path,
)

_MODEL_CACHE: dict[str, object] = {}
_LISTENER_CACHE: dict[tuple[str, str | None, int, int, str, int], 'VoskVoiceCommandListener'] = {}


def _runtime_classes():
    try:
        from vosk import KaldiRecognizer, Model
    except ImportError as exc:
        raise RuntimeError(
            'Python package `vosk` is not installed. Run: python3 -m pip install vosk'
        ) from exc
    return KaldiRecognizer, Model


def _cached_model(model_path: str):
    path = str(validate_model_path(model_path))
    if path not in _MODEL_CACHE:
        _, model_class = _runtime_classes()
        _MODEL_CACHE[path] = model_class(path)
    return _MODEL_CACHE[path]


def _parse_text(payload: str, field: str) -> str:
    try:
        value = json.loads(payload).get(field, '')
    except json.JSONDecodeError:
        return ''
    return value.strip() if isinstance(value, str) else ''


class VoskVoiceCommandListener:
    def __init__(self, model_path, device_id, sample_rate, channels, sample_format, chunk_size, device=None):
        ensure_arecord_available()
        self.model_path = str(validate_model_path(model_path))
        self.device = device if device is not None else device_id_to_arecord_device(device_id)
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_format = sample_format
        self.chunk_size = chunk_size
        self.model = _cached_model(self.model_path)
        self.noise_floor = None
        self._noise_calibration_attempts = 0
        maximize_capture_volume(self.device)

    def calibrate_environment_noise(self, duration_sec=0.5):
        if self.noise_floor is not None:
            return self.noise_floor
        if self._noise_calibration_attempts >= 2:
            return 0.0
        self._noise_calibration_attempts += 1
        self.noise_floor = measure_environment_noise(
            device=self.device,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_format=self.sample_format,
            chunk_size=self.chunk_size,
            duration_sec=duration_sec,
        )
        return self.noise_floor

    def wait_for_command(
        self,
        expected_phrase: str,
        timeout_sec: float = 0.0,
        on_text=None,
        accept_any_speech: bool = False,
        accept_any_recognized_text: bool = False,
        speech_rms_threshold: float = 100.0,
        speech_consecutive_chunks: int = 3,
        noise_calibration_sec: float = 0.5,
        threshold_multiplier: float = 3.0,
        pre_roll_sec: float = 0.3,
        end_silence_sec: float = 0.8,
        min_utterance_sec: float = 0.35,
        max_utterance_sec: float = 10.0,
        max_digital_gain: float = 4.0,
        text_match_mode: str = '',
        text_match_percentage: float = 50.0,
    ) -> str:
        del pre_roll_sec
        del end_silence_sec, min_utterance_sec, max_utterance_sec, max_digital_gain
        maximize_capture_volume(self.device)
        noise_floor = self.calibrate_environment_noise(noise_calibration_sec)
        speech_start_threshold = max(
            float(speech_rms_threshold),
            noise_floor * max(float(threshold_multiplier), 1.0),
        )
        recognizer_class, _ = _runtime_classes()
        recognizer = recognizer_class(self.model, self.sample_rate)
        recognizer.SetWords(True)
        process = subprocess.Popen(
            build_arecord_command(self.device, self.sample_rate, self.channels, self.sample_format),
            stdout=subprocess.PIPE,
            stderr=None,
        )
        started = time.time()
        matched = None
        last_partial = ''
        speech_chunks = 0
        try:
            assert process.stdout is not None
            while True:
                if timeout_sec > 0.0 and time.time() - started > timeout_sec:
                    raise TimeoutError(
                        f'waiting for voice command timed out after {timeout_sec:.1f} seconds'
                    )
                raw_chunk = process.stdout.read(self.chunk_size)
                if not raw_chunk:
                    break
                chunk = downmix_pcm_to_mono(
                    raw_chunk,
                    self.sample_format,
                    self.channels,
                )
                if accept_any_speech:
                    speech_chunks = (
                        speech_chunks + 1
                        if pcm_rms(chunk, self.sample_format) >= speech_start_threshold
                        else 0
                    )
                    if speech_chunks >= max(1, speech_consecutive_chunks):
                        matched = '[speech activity detected]'
                        if on_text is not None:
                            on_text(matched)
                        break
                if recognizer.AcceptWaveform(chunk):
                    text = _parse_text(recognizer.Result(), 'text')
                    if not text:
                        continue
                    if on_text is not None:
                        on_text(text)
                    if recognized_text_matches(
                        text=text,
                        expected_phrase=expected_phrase,
                        mode=text_match_mode,
                        percentage=text_match_percentage,
                        legacy_accept_any=accept_any_recognized_text,
                    ):
                        matched = text
                        break
                else:
                    text = _parse_text(recognizer.PartialResult(), 'partial')
                    if text == last_partial:
                        continue
                    last_partial = text
                    if text and on_text is not None:
                        on_text(text)
                    if text and recognized_text_matches(
                        text=text,
                        expected_phrase=expected_phrase,
                        mode=text_match_mode,
                        percentage=text_match_percentage,
                        legacy_accept_any=accept_any_recognized_text,
                    ):
                        matched = text
                        break
            if matched is None:
                text = _parse_text(recognizer.FinalResult(), 'text')
                if text and on_text is not None:
                    on_text(text)
                if text and recognized_text_matches(
                    text=text,
                    expected_phrase=expected_phrase,
                    mode=text_match_mode,
                    percentage=text_match_percentage,
                    legacy_accept_any=accept_any_recognized_text,
                ):
                    matched = text
            if matched is None:
                raise RuntimeError(
                    'realtime Vosk recognition ended before matching the expected command'
                )
            return matched
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()


def get_cached_listener(model_path, device_id, sample_rate, channels, sample_format, chunk_size, device=None):
    path = str(validate_model_path(model_path))
    resolved_device = device if device is not None else device_id_to_arecord_device(device_id)
    key = (path, resolved_device, sample_rate, channels, sample_format, chunk_size)
    if key not in _LISTENER_CACHE:
        _LISTENER_CACHE[key] = VoskVoiceCommandListener(
            path, device_id, sample_rate, channels, sample_format, chunk_size, resolved_device
        )
    return _LISTENER_CACHE[key]
