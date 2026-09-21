#!/usr/bin/env python3
"""Download the competition package's faster-whisper model outside Git."""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_REPOSITORIES = {
    'base': 'Systran/faster-whisper-base',
    'small': 'Systran/faster-whisper-small',
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = (
    REPOSITORY_ROOT
    / 'agibot_d1_auto_nav'
    / 'agibot_d1_auto_nav'
    / 'models'
    / 'speech'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download the local faster-whisper model used by the competition node.'
    )
    parser.add_argument('--model', choices=MODEL_REPOSITORIES, default='base')
    parser.add_argument(
        '--output',
        type=Path,
        help='Destination directory. Defaults to the package models/speech directory.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'huggingface-hub is required; install the package requirements first'
        ) from exc

    destination = args.output or DEFAULT_MODEL_ROOT / f'faster-whisper-{args.model}'
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPOSITORIES[args.model],
        local_dir=str(destination),
    )
    print(f'{args.model} model ready: {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

