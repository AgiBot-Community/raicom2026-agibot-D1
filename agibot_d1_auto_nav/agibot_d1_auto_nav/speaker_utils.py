from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_VOICE = 'cmn'
DEFAULT_SPEED = 175
DEFAULT_PITCH = 50
DEFAULT_VOLUME = 100


def ensure_command_available(command: str, install_hint: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f'`{command}` is not installed. {install_hint}')


def choose_player(preferred: str) -> str:
    if preferred != 'auto':
        if shutil.which(preferred) is None:
            raise RuntimeError(f'Requested player `{preferred}` is not installed.')
        return preferred

    for player in ('paplay', 'aplay'):
        if shutil.which(player) is not None:
            return player

    raise RuntimeError(
        'No supported audio player was found. Please install `pulseaudio-utils` '
        'or `alsa-utils`.'
    )


def build_tts_command(
    text: str,
    voice: str,
    speed: int,
    pitch: int,
    volume: int,
    output_path: Path,
) -> list[str]:
    return [
        'espeak-ng',
        '-v',
        voice,
        '-s',
        str(speed),
        '-p',
        str(pitch),
        '-a',
        str(volume),
        '-w',
        str(output_path),
        text,
    ]


def synthesize_to_wav(
    text: str,
    voice: str,
    speed: int,
    pitch: int,
    volume: int,
    output_path: Path,
) -> None:
    ensure_command_available(
        'espeak-ng',
        'Please install it first, for example: sudo apt install espeak-ng',
    )
    command = build_tts_command(
        text=text,
        voice=voice,
        speed=speed,
        pitch=pitch,
        volume=volume,
        output_path=output_path,
    )
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'`espeak-ng` exited with status code {result.returncode}')


def play_wav(audio_path: Path, player: str, device: str | None) -> None:
    if player == 'paplay':
        command = ['paplay']
        if device:
            command.extend(['--device', device])
        command.append(str(audio_path))
    elif player == 'aplay':
        command = ['aplay']
        if device:
            command.extend(['-D', device])
        command.append(str(audio_path))
    else:
        raise RuntimeError(f'Unsupported player: {player}')

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'`{player}` exited with status code {result.returncode}')


def speak_text(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: int = DEFAULT_SPEED,
    pitch: int = DEFAULT_PITCH,
    volume: int = DEFAULT_VOLUME,
    player: str = 'auto',
    device: str | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix='speaker_tts_') as temp_dir:
        output_path = Path(temp_dir) / 'speech.wav'
        synthesize_to_wav(
            text=text,
            voice=voice,
            speed=speed,
            pitch=pitch,
            volume=volume,
            output_path=output_path,
        )
        play_wav(output_path, choose_player(player), device)
