import struct

from agibot_d1_auto_nav.speech_command_utils import amplify_pcm_to_full_scale
from agibot_d1_auto_nav.speech_command_utils import amplify_pcm_for_recognition
from agibot_d1_auto_nav.speech_command_utils import command_match_percentage
from agibot_d1_auto_nav.speech_command_utils import downmix_pcm_to_mono
from agibot_d1_auto_nav.speech_command_utils import matches_expected_command
from agibot_d1_auto_nav.speech_command_utils import normalize_backend
from agibot_d1_auto_nav.speech_command_utils import pcm_rms
from agibot_d1_auto_nav.speech_command_utils import recognized_text_matches
from agibot_d1_auto_nav.speech_command_utils import VoiceCommandListener


def unpack_pcm(chunk):
    return struct.unpack(f'<{len(chunk) // 2}h', chunk)


def test_environment_noise_calibration_is_cached(monkeypatch):
    measurements = []

    def fake_measure(**kwargs):
        measurements.append(kwargs)
        return 123.0

    monkeypatch.setattr(
        'agibot_d1_auto_nav.speech_command_utils.measure_environment_noise',
        fake_measure,
    )
    listener = VoiceCommandListener.__new__(VoiceCommandListener)
    listener.device = 'plughw:1,0'
    listener.sample_rate = 44100
    listener.channels = 2
    listener.sample_format = 'S16_LE'
    listener.chunk_size = 4000
    listener.noise_floor = None
    listener._noise_calibration_attempts = 0

    assert listener.calibrate_environment_noise(0.5) == 123.0
    assert listener.calibrate_environment_noise(2.0) == 123.0
    assert len(measurements) == 1
    assert measurements[0]['duration_sec'] == 0.5


def test_pcm_audio_is_amplified_to_full_scale():
    quiet_audio = struct.pack('<4h', -1000, -250, 500, 1000)

    amplified = unpack_pcm(amplify_pcm_to_full_scale(quiet_audio, 'S16_LE'))

    assert amplified == (-32767, -8192, 16384, 32767)


def test_pcm_amplification_preserves_silence_and_full_scale_audio():
    silence = struct.pack('<3h', 0, 0, 0)
    full_scale = struct.pack('<3h', -32768, 0, 32767)

    assert amplify_pcm_to_full_scale(silence, 'S16_LE') == silence
    assert amplify_pcm_to_full_scale(full_scale, 'S16_LE') == full_scale


def test_pcm_amplification_ignores_unsupported_format():
    audio = b'\x01\x02\x03\x04'

    assert amplify_pcm_to_full_scale(audio, 'S24_LE') == audio


def test_pcm_rms_uses_unamplified_signal_level():
    audio = struct.pack('<4h', -100, 100, -100, 100)

    assert pcm_rms(audio, 'S16_LE') == 100.0


def test_stereo_pcm_is_averaged_to_mono_before_recognition():
    stereo = struct.pack('<8h', 1000, 3000, -2000, 2000, 500, 1500, -400, -600)

    mono = unpack_pcm(downmix_pcm_to_mono(stereo, 'S16_LE', channels=2))

    assert mono == (2000, 0, 1000, -500)


def test_recognition_gain_is_bounded_instead_of_forcing_full_scale():
    quiet_audio = struct.pack('<4h', -1000, -250, 500, 1000)

    amplified = unpack_pcm(
        amplify_pcm_for_recognition(quiet_audio, 'S16_LE', max_gain=4.0)
    )

    assert amplified == (-4000, -1000, 2000, 4000)


def test_start_recognition_accepts_loose_matches():
    accepted = (
        '开始识别',
        '请开始识别',
        '开始',
        '识别',
        '开始时别',
        '启动识别',
        '开始检测',
        '可以开始了',
    )
    for text in accepted:
        assert matches_expected_command(text, '开始识别'), text


def test_start_recognition_rejects_unrelated_navigation_speech():
    rejected = (
        '',
        '到',
        '请到目标点',
        '继续走',
        '前方有障碍物',
        '停止识别',
        '不要开始',
    )
    for text in rejected:
        assert not matches_expected_command(text, '开始识别'), text


def test_voice_backend_parameter_accepts_both_models():
    assert normalize_backend('faster_whisper') == 'faster_whisper'
    assert normalize_backend('whisper') == 'faster_whisper'
    assert normalize_backend('vosk') == 'vosk'


def test_first_nonempty_mode_accepts_first_text_but_not_empty_text():
    assert recognized_text_matches('前方有人说话', '请引领到目标点A', 'first_nonempty')
    assert not recognized_text_matches('  ', '请引领到目标点A', 'first_nonempty')


def test_command_mode_accepts_fuzzy_navigation_phrases_for_the_expected_point():
    accepted = (
        '请引领到目标点A',
        '引领到目标点诶',
        '请带我去目标店A',
        '前往目标点A',
        '请引零到目标点A',
    )
    for text in accepted:
        assert recognized_text_matches(
            text,
            '请引领到目标点A',
            'command',
        ), text


def test_command_mode_rejects_noise_unrelated_speech_and_wrong_points():
    rejected = (
        '',
        '有人在说话',
        '今天天气很好',
        '前方有障碍物',
        '目标点A',
        '请引领到目标点B',
        '请引领到目标点',
        '不要引领到目标点A',
    )
    for text in rejected:
        assert not recognized_text_matches(
            text,
            '请引领到目标点A',
            'command',
        ), text


def test_default_matching_requires_the_expected_command():
    assert recognized_text_matches('请引领到目标点A', '请引领到目标点A')
    assert not recognized_text_matches('有人在说话', '请引领到目标点A')


def test_percentage_mode_uses_configured_similarity_threshold():
    text = '请引领到目标点'
    expected = '请引领到目标点A'
    score = command_match_percentage(text, expected)

    assert 0.0 < score < 100.0
    assert recognized_text_matches(text, expected, 'percentage', score)
    assert not recognized_text_matches(text, expected, 'percentage', score + 0.1)
