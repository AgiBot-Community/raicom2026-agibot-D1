from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / 'config'


def test_fast_dynamic_replan_tree_has_no_long_wait():
    root = ET.parse(CONFIG_DIR / 'bt_fast_dynamic_replan.xml').getroot()

    rate_controller = root.find('.//RateController')
    wait = root.find('.//Wait')

    assert rate_controller is not None
    assert float(rate_controller.attrib['hz']) >= 3.0
    assert wait is not None
    assert float(wait.attrib['wait_duration']) <= 1.0


def test_b_to_c_profile_prefers_detour_over_original_path():
    with (CONFIG_DIR / 'auto_nav_params.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)['/**']['ros__parameters']

    assert parameters['b_to_c_global_inflation_radius'] > 0.0
    assert parameters['b_to_c_path_align_scale'] < 24.0
    assert parameters['b_to_c_path_dist_scale'] < 24.0
    assert parameters['b_to_c_goal_dist_scale'] > parameters['b_to_c_path_dist_scale']
    assert parameters['endpoint_inflation_radius'] >= 0.16
    assert (
        parameters['b_to_c_global_inflation_radius']
        > parameters['b_to_c_inflation_radius']
    )
    assert parameters['b_to_c_min_linear_speed'] > 0.0
    assert 0.0 < parameters['b_departure_profile_distance'] < 1.0
    assert 0.0 < parameters['c_departure_profile_distance'] < 1.0
    assert parameters['c_departure_inflation_radius'] < parameters['endpoint_inflation_radius']
    assert parameters['c_departure_max_angular_speed'] > parameters['endpoint_max_angular_speed']


def test_static_and_dynamic_nav2_xy_goal_tolerances_match():
    with (CONFIG_DIR / 'auto_nav_params.yaml').open(encoding='utf-8') as stream:
        auto_nav = yaml.safe_load(stream)['/**']['ros__parameters']
    controller_text = (
        CONFIG_DIR / 'nav2_controller_competition.yaml'
    ).read_text(encoding='utf-8')
    controller_document = controller_text.split('\n/**:', maxsplit=1)[0]
    controller = yaml.safe_load(controller_document)['/**']['ros__parameters']

    expected = auto_nav['nav2_xy_goal_tolerance']
    assert controller['general_goal_checker']['xy_goal_tolerance'] == expected
    assert controller['FollowPath']['xy_goal_tolerance'] == expected


def test_stage_reach_confirmation_has_boolean_switch():
    with (CONFIG_DIR / 'auto_nav_params.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)['/**']['ros__parameters']

    assert isinstance(parameters['stage_reach_confirmation_enabled'], bool)


def test_b_arrival_speech_is_configured():
    with (CONFIG_DIR / 'auto_nav_params.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)['/**']['ros__parameters']

    assert parameters['b_arrival_speech'] == '已到达b点'


def test_obstacle_speech_range_precedes_global_replanning_range():
    with (CONFIG_DIR / 'auto_nav_params.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)['/**']['ros__parameters']
    with (
        CONFIG_DIR / 'nav2_planner_competition.yaml'
    ).open(encoding='utf-8') as stream:
        global_costmap = yaml.safe_load(stream)['/**']['global_costmap'][
            'ros__parameters'
        ]

    scan = global_costmap['obstacle_layer']['scan']
    speech_range = parameters['moving_obstacle_distance_threshold']
    assert speech_range > scan['obstacle_max_range']
    assert speech_range <= scan['raytrace_max_range']


def test_voice_frontend_parameters_are_valid():
    with (CONFIG_DIR / 'auto_nav_params.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)['/**']['ros__parameters']

    assert parameters['voice_command_channels'] >= 1
    assert parameters['voice_noise_calibration_sec'] > 0.0
    assert parameters['voice_threshold_multiplier'] > 1.0
    assert parameters['voice_pre_roll_sec'] > 0.0
    assert parameters['voice_end_silence_sec'] > 0.0
    assert parameters['voice_min_utterance_sec'] > 0.0
    assert (
        parameters['voice_max_utterance_sec']
        > parameters['voice_min_utterance_sec']
    )
    assert 1.0 <= parameters['voice_max_digital_gain'] <= 8.0
    assert parameters['voice_text_match_mode'] in {
        'command',
        'first_nonempty',
        'percentage',
    }
    assert parameters['voice_text_match_mode'] == 'command'
    assert parameters['voice_accept_any_recognized_text'] is False
    assert 0.0 <= parameters['voice_text_match_percentage'] <= 100.0
    assert parameters['voice_command_faster_whisper_model_size'] in {'base', 'small'}
    assert parameters['voice_command_faster_whisper_model_size'] == 'base'
    assert 'voice_command_faster_whisper_base_model_path' not in parameters
    assert 'voice_command_faster_whisper_small_model_path' not in parameters
    assert 'voice_command_vosk_model_path' not in parameters
    assert parameters['voice_command_faster_whisper_beam_size'] == 1
    assert parameters['voice_command_faster_whisper_cpu_threads'] == 4
