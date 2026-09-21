# Third-party notices

The MIT license in this repository covers the competition package's original
code. Dependencies and downloaded model weights keep their upstream licenses.

| Component | How it is used | License / source |
| --- | --- | --- |
| ROS 2 and Nav2 | Middleware, messages, actions, localization and navigation | Their respective upstream ROS package licenses. |
| faster-whisper / CTranslate2 | Default local speech-recognition backend | Their upstream package licenses. |
| `Systran/faster-whisper-base` and `small` | Downloaded local model weights; not tracked by Git | The upstream Hugging Face model cards declare MIT. |
| Vosk and `vosk-model-cn-0.22` | Optional local speech backend; not tracked by Git | Vosk's official model list declares the Chinese 0.22 model as Apache-2.0. |
| OpenCV / NumPy | Local vision pipeline | Their respective upstream licenses. |

The base Agibot D1 workspace is an external runtime dependency and is not part
of this competition repository. Its source and vendor SDK licenses must be
handled by the base workspace distributor.

