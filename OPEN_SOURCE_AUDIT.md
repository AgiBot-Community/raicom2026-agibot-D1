# Competition repository release audit

Scope: the `competition/` directory only. Inspection date: 2026-09-21.
No existing directory was deleted automatically.

The directory is now an independent Git repository on branch `main`. With the
current ignore rules, the initial publishable set is 53 files totaling about
17.1 MiB; the required visual classifier accounts for about 16.8 MiB.

## Required and retained

- `agibot_d1_auto_nav/agibot_d1_auto_nav/`: task state machine, local speech
  recognition, command matching and vision code.
- `agibot_d1_auto_nav/config/`: competition, Nav2 and waypoint parameters.
- `agibot_d1_auto_nav/launch/`: launch entry points.
- `agibot_d1_auto_nav/test/`: unit and configuration tests.
- `agibot_d1_auto_nav/tools/`: microphone and offline validation tools.
- `agibot_d1_auto_nav/agibot_d1_auto_nav/models/*.npz`: approximately 17 MiB
  visual character model required at runtime.
- Package manifests, setup files, README and validation CSV summaries.

## Do not publish

| Path | Approximate size | Reason |
| --- | ---: | --- |
| `agibot_d1_auto_nav/agibot_d1_auto_nav/models/speech/` | 600 MiB | Downloaded faster-whisper weights; above normal Git hosting limits and reproducible with the download script. |
| `agibot_d1_auto_nav/tmp/` | 95 MiB | Runtime camera captures and recognition output. |
| `agibot_d1_auto_nav/{build,install,log}/` | less than 1 MiB here | Generated colcon artifacts containing machine-specific paths. |
| `agibot_d1_auto_nav/{__pycache__,.pytest_cache}/` and nested caches | small | Generated Python/test cache. |
| `agibot_d1_auto_nav/validation/*/` | 3.5 MiB | Replay images and annotated results; keep the CSV summaries, publish images only after privacy review. |

These paths are covered by `.gitignore`. They still exist in the working folder,
so publish with Git rather than uploading a raw ZIP of the current directory.

## Sensitive information

- No cloud key is required by the competition runtime; speech recognition is local.
- No common live-token signature remains in publishable competition source.
- Personal absolute paths were removed from launch files and runtime defaults.
- RTSP URLs and robot-local IP addresses are hardware configuration, not account
  credentials. They remain in the example YAML and should be changed in an
  ignored `*.local.yaml` copy.
- Speech-model and map paths can be supplied through environment variables shown
  in `.env.example`.

## External prerequisites

This is a ROS 2 overlay repository, not the complete robot SDK. A runnable D1
base workspace must already provide:

- `agibot_d1_navigation2`
- `agibot_d1_localization`
- `agibot_d1_map_server`
- the robot driver, laser/odometry topics, TF tree and a Nav2-compatible map

The root README contains commands to verify these prerequisites before build.

## Before making the repository public

1. Run `git status --ignored` and confirm speech weights, captures and colcon
   output are ignored.
2. Review waypoint coordinates, the map used at runtime and validation imagery
   for competition or venue confidentiality.
3. Replace the generic maintainer identity in `package.xml`/`setup.py` with the
   team or repository contact if desired.
4. Execute the README build, unit tests, microphone test and a low-speed hardware
   smoke test on Ubuntu 22.04 / ROS 2 Humble.
5. Create the initial commit and add the chosen public Git remote.
