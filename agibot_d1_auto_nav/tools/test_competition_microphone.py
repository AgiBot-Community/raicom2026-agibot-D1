#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from agibot_d1_auto_nav.speech_command_utils import build_arecord_command
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_BASE_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_SMALL_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_VOSK_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import device_id_to_arecord_device
from agibot_d1_auto_nav.speech_command_utils import normalize_backend
from agibot_d1_auto_nav.speech_command_utils import preload_listener
from agibot_d1_auto_nav.speech_command_utils import wait_for_command


DEFAULT_CONFIG = PACKAGE_ROOT / 'config' / 'auto_nav_params.yaml'


def load_parameters(config_path: Path) -> dict[str, object]:
    with config_path.expanduser().open(encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    try:
        parameters = document['/**']['ros__parameters']
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f'cannot find /**.ros__parameters in {config_path}'
        ) from exc
    if not isinstance(parameters, dict):
        raise RuntimeError(f'invalid ROS parameter mapping in {config_path}')
    return parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Test the exact microphone and speech-recognition pipeline used by '
            'competition/agibot_d1_auto_nav.'
        ),
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--backend', choices=('vosk', 'faster_whisper'))
    parser.add_argument(
        '--model-size',
        choices=('base', 'small'),
        help=(
            'Temporarily select the faster-whisper base or small model; '
            'defaults to voice_command_faster_whisper_model_size from config.'
        ),
    )
    parser.add_argument('--expected', default='请引领到目标点A')
    parser.add_argument('--device-id', type=int)
    parser.add_argument('--sample-rate', type=int)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--timeout', type=float)
    parser.add_argument(
        '--match-mode',
        choices=('command', 'first_nonempty', 'percentage'),
        help='Temporarily override voice_text_match_mode.',
    )
    parser.add_argument(
        '--match-percentage',
        type=float,
        help='Temporarily override voice_text_match_percentage.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Require a complete normalized match with --expected.',
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Start another recognition round after every successful result.',
    )
    parser.add_argument(
        '--no-preload',
        action='store_true',
        help='Skip the explicit model preload timing step.',
    )
    return parser.parse_args()


def parameter(parameters: dict[str, object], name: str, default):
    return parameters[name] if name in parameters else default


def main() -> int:
    args = parse_args()
    parameters = load_parameters(args.config)
    backend = normalize_backend(
        args.backend or str(parameter(parameters, 'voice_command_backend', 'vosk'))
    )
    if backend == 'vosk':
        if args.model_size is not None:
            raise ValueError('--model-size is only valid with faster_whisper')
        model_size = None
        model_parameter = 'voice_command_vosk_model_path'
        default_model_path = DEFAULT_VOSK_MODEL_PATH
    else:
        model_size = args.model_size or str(
            parameter(parameters, 'voice_command_faster_whisper_model_size', 'small')
        ).strip().lower()
        if model_size not in {'base', 'small'}:
            raise ValueError(
                'voice_command_faster_whisper_model_size must be base or small; '
                f'got {model_size!r}'
            )
        model_parameter = f'voice_command_faster_whisper_{model_size}_model_path'
        default_model_path = (
            DEFAULT_BASE_MODEL_PATH
            if model_size == 'base'
            else DEFAULT_SMALL_MODEL_PATH
        )
    model_path = str(parameter(parameters, model_parameter, default_model_path))
    device_id = (
        args.device_id
        if args.device_id is not None
        else int(parameter(parameters, 'voice_command_device_id', 0))
    )
    sample_rate = (
        args.sample_rate
        if args.sample_rate is not None
        else int(parameter(parameters, 'voice_command_sample_rate', 44100))
    )
    channels = (
        args.channels
        if args.channels is not None
        else int(parameter(parameters, 'voice_command_channels', 1))
    )
    sample_format = str(
        parameter(parameters, 'voice_command_sample_format', 'S16_LE')
    )
    chunk_size = int(parameter(parameters, 'voice_command_chunk_size', 4000))
    beam_size = int(
        parameter(parameters, 'voice_command_faster_whisper_beam_size', 1)
    )
    cpu_threads = int(
        parameter(parameters, 'voice_command_faster_whisper_cpu_threads', 4)
    )
    timeout_sec = (
        args.timeout
        if args.timeout is not None
        else float(parameter(parameters, 'voice_command_timeout_sec', 0.0))
    )
    text_match_mode = args.match_mode or str(
        parameter(parameters, 'voice_text_match_mode', 'command')
    )
    text_match_percentage = (
        args.match_percentage
        if args.match_percentage is not None
        else float(parameter(parameters, 'voice_text_match_percentage', 65.0))
    )
    if args.strict:
        text_match_mode = 'percentage'
        text_match_percentage = 100.0
    device = device_id_to_arecord_device(device_id)
    arecord_command = build_arecord_command(
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
    )

    print(f'config: {args.config.resolve()}')
    print(f'backend: {backend}')
    if model_size is not None:
        print(f'model size: {model_size}')
    print(f'model: {model_path}')
    if backend == 'faster_whisper':
        print(f'faster-whisper: beam_size={beam_size}, cpu_threads={cpu_threads}')
    print(
        f'audio: device={device or "default"}, rate={sample_rate}, '
        f'channels={channels}, format={sample_format}, chunk={chunk_size}'
    )
    print(f'arecord: {" ".join(arecord_command)}')
    print(f'expected text: {args.expected}')
    print(
        f'matching: mode={text_match_mode}, '
        f'percentage={text_match_percentage:.1f}%'
    )

    if not args.no_preload:
        started_at = time.monotonic()
        print('loading speech model...', flush=True)
        preload_listener(
            model_path=model_path,
            device_id=device_id,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            chunk_size=chunk_size,
            backend=backend,
            beam_size=beam_size,
            cpu_threads=cpu_threads,
            noise_calibration_sec=float(
                parameter(parameters, 'voice_noise_calibration_sec', 0.5)
            ),
        )
        print(f'model ready in {time.monotonic() - started_at:.2f} seconds')

    round_number = 0
    while True:
        round_number += 1
        last_text = {'value': ''}

        def print_text(text: str) -> None:
            if text == last_text['value']:
                return
            last_text['value'] = text
            print(f'heard: {text}', flush=True)

        if args.no_preload and round_number == 1:
            instruction = 'keep quiet briefly for one-time calibration, then speak'
        else:
            instruction = 'speak now'
        print(
            f'\nround {round_number}: {instruction}; press Ctrl+C to stop.',
            flush=True,
        )
        result = wait_for_command(
            expected_phrase=args.expected,
            model_path=model_path,
            device_id=device_id,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            chunk_size=chunk_size,
            timeout_sec=timeout_sec,
            on_text=print_text,
            accept_any_recognized_text=(
                False
                if args.strict
                else bool(
                    parameter(parameters, 'voice_accept_any_recognized_text', False)
                )
            ),
            speech_rms_threshold=float(
                parameter(parameters, 'voice_any_speech_rms_threshold', 100.0)
            ),
            speech_consecutive_chunks=int(
                parameter(parameters, 'voice_any_speech_consecutive_chunks', 3)
            ),
            noise_calibration_sec=float(
                parameter(parameters, 'voice_noise_calibration_sec', 0.5)
            ),
            threshold_multiplier=float(
                parameter(parameters, 'voice_threshold_multiplier', 3.0)
            ),
            pre_roll_sec=float(parameter(parameters, 'voice_pre_roll_sec', 0.3)),
            end_silence_sec=float(
                parameter(parameters, 'voice_end_silence_sec', 0.8)
            ),
            min_utterance_sec=float(
                parameter(parameters, 'voice_min_utterance_sec', 0.35)
            ),
            max_utterance_sec=float(
                parameter(parameters, 'voice_max_utterance_sec', 10.0)
            ),
            max_digital_gain=float(
                parameter(parameters, 'voice_max_digital_gain', 4.0)
            ),
            text_match_mode=text_match_mode,
            text_match_percentage=text_match_percentage,
            backend=backend,
            beam_size=beam_size,
            cpu_threads=cpu_threads,
        )
        print(f'accepted: {result}', flush=True)
        if not args.continuous:
            return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nstopped by user')
        raise SystemExit(130)
    except Exception as exc:
        print(f'\nmicrophone test failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
