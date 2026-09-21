from __future__ import annotations

import collections
import difflib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from .chinese_asr_backend import load_model, transcribe_pcm


_SPEECH_MODEL_ROOT = Path(__file__).resolve().parent / 'models' / 'speech'
_LEGACY_SMALL_MODEL_PATH = (
    _SPEECH_MODEL_ROOT
    / 'models--Systran--faster-whisper-small'
    / 'snapshots'
    / '536b0662742c02347bc0e980a01041f333bce120'
)
_PORTABLE_SMALL_MODEL_PATH = _SPEECH_MODEL_ROOT / 'faster-whisper-small'
DEFAULT_SMALL_MODEL_PATH = os.environ.get(
    'AGIBOT_WHISPER_SMALL_MODEL_PATH',
    str(
        _PORTABLE_SMALL_MODEL_PATH
        if _PORTABLE_SMALL_MODEL_PATH.exists()
        else _LEGACY_SMALL_MODEL_PATH
    ),
)
DEFAULT_BASE_MODEL_PATH = os.environ.get(
    'AGIBOT_WHISPER_BASE_MODEL_PATH',
    str(_SPEECH_MODEL_ROOT / 'faster-whisper-base'),
)
DEFAULT_VOSK_MODEL_PATH = os.environ.get(
    'AGIBOT_VOSK_MODEL_PATH',
    str(_SPEECH_MODEL_ROOT / 'vosk-model-cn-0.22'),
)
DEFAULT_MODEL_PATH = DEFAULT_SMALL_MODEL_PATH
DEFAULT_BACKEND = 'faster_whisper'
SUPPORTED_BACKENDS = ('faster_whisper', 'vosk')
SUPPORTED_TEXT_MATCH_MODES = ('command', 'first_nonempty', 'percentage')
DEFAULT_DEVICE_ID = 0
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_FORMAT = 'S16_LE'
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_FASTER_WHISPER_BEAM_SIZE = 1
DEFAULT_FASTER_WHISPER_CPU_THREADS = 4
PCM_TARGET_PEAK = 32767


_MODEL_CACHE: dict[tuple[str, int], object] = {}
_LISTENER_CACHE: dict[
    tuple[str, str | None, int, int, str, int, int, int],
    'VoiceCommandListener',
] = {}


def ensure_arecord_available() -> None:
    if shutil.which('arecord') is None:
        raise RuntimeError('`arecord` is not installed. Please install `alsa-utils` first.')


def normalize_backend(backend: str) -> str:
    normalized = str(backend).strip().lower().replace('-', '_')
    if normalized == 'whisper':
        normalized = 'faster_whisper'
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            f'unsupported voice_command_backend {backend!r}; '
            f'choose one of {SUPPORTED_BACKENDS}'
        )
    return normalized


def normalize_text_match_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace('-', '_')
    if normalized not in SUPPORTED_TEXT_MATCH_MODES:
        raise ValueError(
            f'unsupported voice_text_match_mode {mode!r}; '
            f'choose one of {SUPPORTED_TEXT_MATCH_MODES}'
        )
    return normalized


def validate_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f'speech recognition model directory does not exist: {path}')
    return path


def get_cached_model(model_path: str, cpu_threads: int = DEFAULT_FASTER_WHISPER_CPU_THREADS):
    validated_model_path = validate_model_path(model_path)
    cpu_threads = max(1, int(cpu_threads))
    cache_key = (str(validated_model_path), cpu_threads)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded_model = load_model(cache_key[0], cpu_threads=cpu_threads)
    _MODEL_CACHE[cache_key] = loaded_model
    return loaded_model


class VoiceCommandListener:
    def __init__(
        self,
        model_path: str,
        device_id: int | None,
        sample_rate: int,
        channels: int,
        sample_format: str,
        chunk_size: int,
        device: str | None = None,
        beam_size: int = DEFAULT_FASTER_WHISPER_BEAM_SIZE,
        cpu_threads: int = DEFAULT_FASTER_WHISPER_CPU_THREADS,
    ) -> None:
        ensure_arecord_available()
        validated_model_path = validate_model_path(model_path)
        resolved_device = device
        if resolved_device is None:
            resolved_device = device_id_to_arecord_device(device_id)

        self.model_path = str(validated_model_path)
        self.device = resolved_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_format = sample_format
        self.chunk_size = chunk_size
        self.beam_size = max(1, int(beam_size))
        self.cpu_threads = max(1, int(cpu_threads))
        self.model = get_cached_model(self.model_path, self.cpu_threads)
        self.noise_floor: float | None = None
        self._noise_calibration_attempts = 0

        maximize_capture_volume(self.device)

    def calibrate_environment_noise(self, duration_sec: float = 0.5) -> float:
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
        maximize_capture_volume(self.device)
        noise_floor = self.calibrate_environment_noise(noise_calibration_sec)
        arecord_command = build_arecord_command(
            device=self.device,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_format=self.sample_format,
        )

        process = subprocess.Popen(
            arecord_command,
            stdout=subprocess.PIPE,
            stderr=None,
        )

        start_time = time.time()
        final_matched_text: str | None = None
        speech_chunk_count = 0
        speaking = False
        silence_chunks = 0
        utterance_chunks: list[bytes] = []
        block_seconds = self.chunk_size / max(2 * self.channels * self.sample_rate, 1)
        pre_roll_chunks = max(
            1, round(max(pre_roll_sec, 0.0) / max(block_seconds, 0.01))
        )
        end_silence_chunks = max(
            1, round(max(end_silence_sec, 0.1) / max(block_seconds, 0.01))
        )
        min_utterance_chunks = max(
            1, round(max(min_utterance_sec, 0.1) / max(block_seconds, 0.01))
        )
        max_utterance_chunks = max(
            min_utterance_chunks,
            round(max(max_utterance_sec, min_utterance_sec) / max(block_seconds, 0.01)),
        )
        pre_roll = collections.deque(maxlen=pre_roll_chunks)
        start_threshold = max(
            float(speech_rms_threshold),
            noise_floor * max(float(threshold_multiplier), 1.0),
        )
        stop_threshold = max(
            float(speech_rms_threshold) * 0.7,
            noise_floor * max(float(threshold_multiplier) * 0.65, 1.0),
        )

        try:
            assert process.stdout is not None
            while True:
                if timeout_sec > 0.0 and time.time() - start_time > timeout_sec:
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
                level = pcm_rms(chunk, self.sample_format)

                if accept_any_speech:
                    if level >= start_threshold:
                        speech_chunk_count += 1
                    else:
                        speech_chunk_count = 0
                    if speech_chunk_count >= max(speech_consecutive_chunks, 1):
                        final_matched_text = '[speech activity detected]'
                        if on_text is not None:
                            on_text(final_matched_text)
                        break

                if not speaking:
                    pre_roll.append(chunk)
                    if level < start_threshold:
                        continue
                    speaking = True
                    silence_chunks = 0
                    utterance_chunks = list(pre_roll)
                    pre_roll.clear()
                else:
                    utterance_chunks.append(chunk)
                    if level < stop_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                utterance_finished = (
                    silence_chunks >= end_silence_chunks
                    or len(utterance_chunks) >= max_utterance_chunks
                )
                if speaking and utterance_finished:
                    speech_chunks = len(utterance_chunks) - silence_chunks
                    if speech_chunks >= min_utterance_chunks:
                        audio = amplify_pcm_for_recognition(
                            b''.join(utterance_chunks),
                            self.sample_format,
                            max_gain=max_digital_gain,
                        )
                        text = transcribe_pcm(
                            self.model,
                            audio,
                            self.sample_rate,
                            beam_size=self.beam_size,
                        )
                        if text and on_text is not None:
                            on_text(text)
                        if text and recognized_text_matches(
                            text=text,
                            expected_phrase=expected_phrase,
                            mode=text_match_mode,
                            percentage=text_match_percentage,
                            legacy_accept_any=accept_any_recognized_text,
                        ):
                            final_matched_text = text
                            break
                    speaking = False
                    silence_chunks = 0
                    utterance_chunks = []
                    pre_roll.clear()

            if final_matched_text is None:
                raise RuntimeError(
                    'realtime speech recognition ended before matching the expected command'
                )

            return final_matched_text
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()


def get_cached_listener(
    model_path: str,
    device_id: int | None,
    sample_rate: int,
    channels: int,
    sample_format: str,
    chunk_size: int,
    device: str | None = None,
    backend: str = DEFAULT_BACKEND,
    beam_size: int = DEFAULT_FASTER_WHISPER_BEAM_SIZE,
    cpu_threads: int = DEFAULT_FASTER_WHISPER_CPU_THREADS,
):
    selected_backend = normalize_backend(backend)
    if selected_backend == 'vosk':
        from .speech_backends.vosk_original import get_cached_listener as get_vosk_listener
        return get_vosk_listener(
            model_path=model_path,
            device_id=device_id,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            chunk_size=chunk_size,
            device=device,
        )
    validated_model_path = str(validate_model_path(model_path))
    resolved_device = device
    if resolved_device is None:
        resolved_device = device_id_to_arecord_device(device_id)

    cache_key = (
        validated_model_path,
        resolved_device,
        sample_rate,
        channels,
        sample_format,
        chunk_size,
        max(1, int(beam_size)),
        max(1, int(cpu_threads)),
    )
    cached = _LISTENER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    listener = VoiceCommandListener(
        model_path=validated_model_path,
        device_id=device_id,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        chunk_size=chunk_size,
        device=resolved_device,
        beam_size=beam_size,
        cpu_threads=cpu_threads,
    )
    _LISTENER_CACHE[cache_key] = listener
    return listener


def preload_listener(
    model_path: str = DEFAULT_MODEL_PATH,
    device_id: int | None = DEFAULT_DEVICE_ID,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_format: str = DEFAULT_SAMPLE_FORMAT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    device: str | None = None,
    backend: str = DEFAULT_BACKEND,
    beam_size: int = DEFAULT_FASTER_WHISPER_BEAM_SIZE,
    cpu_threads: int = DEFAULT_FASTER_WHISPER_CPU_THREADS,
    noise_calibration_sec: float = 0.5,
):
    listener = get_cached_listener(
        model_path=model_path,
        device_id=device_id,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        chunk_size=chunk_size,
        device=device,
        backend=backend,
        beam_size=beam_size,
        cpu_threads=cpu_threads,
    )
    listener.calibrate_environment_noise(noise_calibration_sec)
    return listener


def build_arecord_command(
    device: str | None,
    sample_rate: int,
    channels: int,
    sample_format: str,
) -> list[str]:
    command = [
        'arecord',
        '-q',
        '-t',
        'raw',
        '-f',
        sample_format,
        '-r',
        str(sample_rate),
        '-c',
        str(channels),
    ]
    if device:
        command.extend(['-D', device])
    command.append('-')
    return command


def measure_environment_noise(
    device: str | None,
    sample_rate: int,
    channels: int,
    sample_format: str,
    chunk_size: int,
    duration_sec: float,
) -> float:
    if duration_sec <= 0.0:
        return 0.0
    maximize_capture_volume(device)
    bytes_per_sample = 2 if sample_format == 'S16_LE' else 2
    block_seconds = chunk_size / max(
        bytes_per_sample * channels * sample_rate,
        1,
    )
    chunk_count = max(
        1,
        round(duration_sec / max(block_seconds, 0.01)),
    )
    process = subprocess.Popen(
        build_arecord_command(
            device=device,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
        ),
        stdout=subprocess.PIPE,
        stderr=None,
    )
    levels: list[float] = []
    try:
        assert process.stdout is not None
        for _ in range(chunk_count):
            raw_chunk = process.stdout.read(chunk_size)
            if not raw_chunk:
                break
            chunk = downmix_pcm_to_mono(raw_chunk, sample_format, channels)
            levels.append(pcm_rms(chunk, sample_format))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
    if not levels:
        raise RuntimeError('microphone returned no audio during noise calibration')
    levels.sort()
    return levels[len(levels) // 2]


def device_id_to_arecord_device(device_id: int | None) -> str | None:
    if device_id is None or device_id < 0:
        return None
    return f'hw:{device_id},0'


def maximize_capture_volume(device: str | None) -> None:
    if shutil.which('amixer') is None:
        return

    commands = [
        ['amixer', 'sset', 'Capture', '100%'],
        ['amixer', 'sset', 'Mic', '100%'],
        ['amixer', 'sset', 'Digital', '100%'],
    ]
    if device:
        commands = [
            ['amixer', '-D', device, 'sset', 'Capture', '100%'],
            ['amixer', '-D', device, 'sset', 'Mic', '100%'],
            ['amixer', '-D', device, 'sset', 'Digital', '100%'],
        ] + commands

    for command in commands:
        subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def amplify_pcm_to_full_scale(chunk: bytes, sample_format: str) -> bytes:
    """Peak-normalize signed 16-bit PCM before passing it to the recognizer."""
    if sample_format.upper() != 'S16_LE' or len(chunk) < 2:
        return chunk

    sample_bytes = len(chunk) - (len(chunk) % 2)
    samples = struct.unpack(f'<{sample_bytes // 2}h', chunk[:sample_bytes])
    peak = max(abs(sample) for sample in samples)
    if peak == 0 or peak >= PCM_TARGET_PEAK:
        return chunk

    gain = PCM_TARGET_PEAK / peak
    amplified = (
        max(-32768, min(PCM_TARGET_PEAK, round(sample * gain)))
        for sample in samples
    )
    result = struct.pack(f'<{len(samples)}h', *amplified)
    return result + chunk[sample_bytes:]


def amplify_pcm_for_recognition(
    chunk: bytes,
    sample_format: str,
    max_gain: float = 4.0,
    target_peak: int = 28000,
) -> bytes:
    """Apply one bounded gain to a complete utterance without changing its SNR."""
    if sample_format.upper() != 'S16_LE' or len(chunk) < 2:
        return chunk
    sample_bytes = len(chunk) - (len(chunk) % 2)
    samples = struct.unpack(f'<{sample_bytes // 2}h', chunk[:sample_bytes])
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        return chunk
    gain = min(max(float(max_gain), 1.0), target_peak / peak)
    if gain <= 1.0:
        return chunk
    amplified = (
        max(-32768, min(32767, round(sample * gain))) for sample in samples
    )
    result = struct.pack(f'<{len(samples)}h', *amplified)
    return result + chunk[sample_bytes:]


def downmix_pcm_to_mono(
    chunk: bytes,
    sample_format: str,
    channels: int,
) -> bytes:
    """Average interleaved signed 16-bit channels into a true mono PCM stream."""
    if channels <= 1 or sample_format.upper() != 'S16_LE' or len(chunk) < 2:
        return chunk
    frame_bytes = 2 * channels
    usable_bytes = len(chunk) - (len(chunk) % frame_bytes)
    if usable_bytes == 0:
        return b''
    samples = struct.unpack(f'<{usable_bytes // 2}h', chunk[:usable_bytes])
    mono = []
    for start in range(0, len(samples), channels):
        frame = samples[start:start + channels]
        mono.append(round(sum(frame) / channels))
    return struct.pack(f'<{len(mono)}h', *mono)


def pcm_rms(chunk: bytes, sample_format: str) -> float:
    """Return the RMS level of signed 16-bit PCM without applying digital gain."""
    if sample_format.upper() != 'S16_LE' or len(chunk) < 2:
        return 0.0
    sample_bytes = len(chunk) - (len(chunk) % 2)
    samples = struct.unpack(f'<{sample_bytes // 2}h', chunk[:sample_bytes])
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def parse_vosk_text(payload: str, field: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ''
    value = data.get(field, '')
    return value.strip() if isinstance(value, str) else ''


def normalize_command_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r'[\s，。,\.!！?？:：;；"“”\'‘’]', '', lowered)


POINT_ALIASES = {
    'a': ('a', '诶', '哎', '欸', '唉'),
    'b': ('b', '比', '逼', '必', '壁', '币', '避', '毕'),
    'c': ('c', '西', '希', '溪', '夕', '锡'),
}

POINT_MARKERS = ('目标点', '目标店', '目标电')
NAVIGATION_KEYWORDS = (
    '引领',
    '领到',
    '带到',
    '带我去',
    '带去',
    '去',
    '去往',
    '前往',
    '导航',
    '请到',
    '到目标',
)
COMMAND_NEGATIONS = ('不要', '不用', '别去', '别带', '停止', '取消')
FUZZY_RATIO_THRESHOLD = 0.50
SHORT_FUZZY_RATIO_THRESHOLD = 0.45
GENERIC_COMMAND_FUZZY_THRESHOLD = 0.62
NAVIGATION_COMMAND_FUZZY_THRESHOLD = 0.65

START_RECOGNITION_COMMAND = '开始识别'
START_RECOGNITION_VARIANTS = (
    '开始识别',
    '开始辨别',
    '开始识辨',
    '开始检测',
    '启动识别',
    '进行识别',
    '可以识别',
    '开始看',
)
START_RECOGNITION_FRAGMENTS = (
    '开始',
    '开使',
    '识别',
    '时别',
    '十别',
    '辨别',
)
START_RECOGNITION_NEGATIONS = (
    '停止',
    '暂停',
    '取消',
    '不要',
    '不用',
    '别开始',
)
START_RECOGNITION_FUZZY_THRESHOLD = 0.34

COMMAND_PREFIXES = (
    '请引领到',
    '引领到',
    '请带到',
    '带到',
    '请到',
    '到',
    '请去',
    '去',
    '请前往',
    '前往',
    '',
)


def build_command_variants(expected_phrase: str) -> list[str]:
    normalized = normalize_command_text(expected_phrase)
    variants = {normalized}

    target_point = ''
    if normalized:
        last_char = normalized[-1]
        if last_char in POINT_ALIASES:
            target_point = last_char

    if not target_point:
        return [variant for variant in variants if variant]

    for prefix in COMMAND_PREFIXES:
        for marker in POINT_MARKERS:
            for alias in POINT_ALIASES[target_point]:
                variants.add(f'{prefix}{marker}{alias}')

    return [variant for variant in variants if variant]


def fuzzy_contains(text: str, expected: str, threshold: float | None = None) -> bool:
    if not text or not expected:
        return False
    if expected in text:
        return True

    if threshold is None:
        threshold = (
            SHORT_FUZZY_RATIO_THRESHOLD
            if min(len(text), len(expected)) <= 4
            else FUZZY_RATIO_THRESHOLD
        )

    if len(text) <= len(expected):
        return difflib.SequenceMatcher(None, text, expected).ratio() >= threshold

    window = len(expected)
    for start in range(0, len(text) - window + 1):
        chunk = text[start:start + window]
        if difflib.SequenceMatcher(None, chunk, expected).ratio() >= threshold:
            return True
    return False


def _navigation_target_matches(text: str, target_point: str) -> bool:
    expected_aliases = POINT_ALIASES[target_point]
    all_aliases = {
        alias: point
        for point, aliases in POINT_ALIASES.items()
        for alias in aliases
    }

    for marker in POINT_MARKERS:
        marker_index = text.find(marker)
        if marker_index < 0:
            continue
        suffix = text[marker_index + len(marker):]
        if not suffix:
            return False
        for alias, point in all_aliases.items():
            if suffix.startswith(alias):
                return point == target_point
        return False

    return any(text.endswith(alias) for alias in expected_aliases)


def _matches_navigation_command(
    text: str,
    expected_phrase: str,
    target_point: str,
) -> bool:
    if any(negation in text for negation in COMMAND_NEGATIONS):
        return False
    if not _navigation_target_matches(text, target_point):
        return False
    if any(keyword in text for keyword in NAVIGATION_KEYWORDS):
        return True
    marker_positions = [
        text.find(marker) for marker in POINT_MARKERS if marker in text
    ]
    if marker_positions and min(marker_positions) < 2:
        # A bare target such as "目标点A" identifies a place but does not
        # contain an instruction to move there.
        return False
    return fuzzy_contains(
        text,
        normalize_command_text(expected_phrase),
        threshold=NAVIGATION_COMMAND_FUZZY_THRESHOLD,
    )


def matches_start_recognition_command(text: str) -> bool:
    normalized_text = normalize_command_text(text)
    if len(normalized_text) < 2:
        return False
    if any(negation in normalized_text for negation in START_RECOGNITION_NEGATIONS):
        return False
    if any(fragment in normalized_text for fragment in START_RECOGNITION_FRAGMENTS):
        return True
    return any(
        difflib.SequenceMatcher(None, normalized_text, variant).ratio()
        >= START_RECOGNITION_FUZZY_THRESHOLD
        for variant in START_RECOGNITION_VARIANTS
    )


def matches_expected_command(text: str, expected_phrase: str) -> bool:
    normalized_text = normalize_command_text(text)
    normalized_expected = normalize_command_text(expected_phrase)
    if not normalized_text or not normalized_expected:
        return False
    if normalized_expected == START_RECOGNITION_COMMAND:
        return matches_start_recognition_command(normalized_text)

    target_point = normalized_expected[-1]
    if target_point in POINT_ALIASES and any(
        marker in normalized_expected for marker in POINT_MARKERS
    ):
        return _matches_navigation_command(
            normalized_text,
            normalized_expected,
            target_point,
        )

    return fuzzy_contains(
        normalized_text,
        normalized_expected,
        threshold=GENERIC_COMMAND_FUZZY_THRESHOLD,
    )


def command_match_percentage(text: str, expected_phrase: str) -> float:
    normalized_text = normalize_command_text(text)
    normalized_expected = normalize_command_text(expected_phrase)
    if not normalized_text or not normalized_expected:
        return 0.0
    if normalized_expected in normalized_text:
        return 100.0
    return 100.0 * difflib.SequenceMatcher(
        None,
        normalized_text,
        normalized_expected,
    ).ratio()


def recognized_text_matches(
    text: str,
    expected_phrase: str,
    mode: str = '',
    percentage: float = 50.0,
    legacy_accept_any: bool = False,
) -> bool:
    if not normalize_command_text(text):
        return False
    if not str(mode).strip():
        return legacy_accept_any or matches_expected_command(text, expected_phrase)
    selected_mode = normalize_text_match_mode(mode)
    if selected_mode == 'command':
        return matches_expected_command(text, expected_phrase)
    if selected_mode == 'first_nonempty':
        return True
    threshold = min(max(float(percentage), 0.0), 100.0)
    return command_match_percentage(text, expected_phrase) >= threshold


def wait_for_command(
    expected_phrase: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device_id: int | None = DEFAULT_DEVICE_ID,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_format: str = DEFAULT_SAMPLE_FORMAT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout_sec: float = 0.0,
    on_text=None,
    api_key: str = '',
    base_websocket_api_url: str = '',
    model_name: str = '',
    block_size: int = 0,
    device_index: int | None = None,
    device: str | None = None,
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
    backend: str = DEFAULT_BACKEND,
    beam_size: int = DEFAULT_FASTER_WHISPER_BEAM_SIZE,
    cpu_threads: int = DEFAULT_FASTER_WHISPER_CPU_THREADS,
) -> str:
    del api_key
    del base_websocket_api_url
    del model_name
    del block_size
    del device_index
    listener = get_cached_listener(
        model_path=model_path,
        device_id=device_id,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        chunk_size=chunk_size,
        device=device,
        backend=backend,
        beam_size=beam_size,
        cpu_threads=cpu_threads,
    )
    return listener.wait_for_command(
        expected_phrase=expected_phrase,
        timeout_sec=timeout_sec,
        on_text=on_text,
        accept_any_speech=accept_any_speech,
        accept_any_recognized_text=accept_any_recognized_text,
        speech_rms_threshold=speech_rms_threshold,
        speech_consecutive_chunks=speech_consecutive_chunks,
        noise_calibration_sec=noise_calibration_sec,
        threshold_multiplier=threshold_multiplier,
        pre_roll_sec=pre_roll_sec,
        end_silence_sec=end_silence_sec,
        min_utterance_sec=min_utterance_sec,
        max_utterance_sec=max_utterance_sec,
        max_digital_gain=max_digital_gain,
        text_match_mode=text_match_mode,
        text_match_percentage=text_match_percentage,
    )
