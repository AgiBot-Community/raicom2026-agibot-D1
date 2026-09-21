import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action import NavigateThroughPoses
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from agibot_d1_auto_nav.speech_command_utils import DEFAULT_CHANNELS as DEFAULT_ASR_CHANNELS
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_BACKEND as DEFAULT_ASR_BACKEND
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_BASE_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_CHUNK_SIZE as DEFAULT_ASR_CHUNK_SIZE
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_DEVICE_ID as DEFAULT_ASR_DEVICE_ID
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_FASTER_WHISPER_BEAM_SIZE
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_FASTER_WHISPER_CPU_THREADS
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_MODEL_PATH as DEFAULT_ASR_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_SAMPLE_FORMAT as DEFAULT_ASR_SAMPLE_FORMAT
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_SAMPLE_RATE as DEFAULT_ASR_SAMPLE_RATE
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_SMALL_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import DEFAULT_VOSK_MODEL_PATH
from agibot_d1_auto_nav.speech_command_utils import preload_listener
from agibot_d1_auto_nav.speech_command_utils import normalize_backend
from agibot_d1_auto_nav.speech_command_utils import wait_for_command
from agibot_d1_auto_nav.character_classifier import CharacterClassifier
from agibot_d1_auto_nav.qrcode_utils import capture_and_recognize_qrcode
from agibot_d1_auto_nav.speaker_utils import speak_text
from agibot_d1_auto_nav.traffic_light_utils import capture_and_recognize_traffic_light
from agibot_d1_auto_nav.vision_utils import DEFAULT_RTSP_URLS
from agibot_d1_auto_nav.vision_utils import capture_and_recognize_sign
from agibot_d1_auto_nav.vision_utils import color_to_chinese


WAYPOINT_PARAMETER_NAMES = (
    'initial_x', 'initial_y', 'initial_yaw',
    'a_x', 'a_y', 'a_yaw',
    'b_x', 'b_y', 'b_yaw',
    'c_x', 'c_y', 'c_yaw',
    'path1_mid_x', 'path1_mid_y', 'path1_mid_yaw',
    'path2_mid_x', 'path2_mid_y', 'path2_mid_yaw',
)


class AutoNavNode(Node):
    def __init__(self) -> None:
        super().__init__('agibot_d1_auto_nav')

        self.declare_parameter('initial_x', math.nan)
        self.declare_parameter('initial_y', math.nan)
        self.declare_parameter('initial_yaw', math.nan)
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('a_x', math.nan)
        self.declare_parameter('a_y', math.nan)
        self.declare_parameter('a_yaw', math.nan)
        self.declare_parameter('wait_at_a_sec', 2.0)
        self.declare_parameter('a_arrival_speech', '已到达a点')
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('voice_command_enabled', True)
        self.declare_parameter('voice_command_a', '请引领到目标点A')
        self.declare_parameter('voice_command_b', '请引领到目标点B')
        self.declare_parameter('voice_command_c', '请引领到目标点C')
        self.declare_parameter('voice_command_start_recognition', '开始识别')
        self.declare_parameter('voice_command_find_target', '请寻找指定目标')
        self.declare_parameter('voice_command_can_pass', '是否可以通行')
        self.declare_parameter('voice_command_backend', DEFAULT_ASR_BACKEND)
        self.declare_parameter('voice_command_model_path', DEFAULT_ASR_MODEL_PATH)
        self.declare_parameter('voice_command_faster_whisper_model_path', DEFAULT_ASR_MODEL_PATH)
        self.declare_parameter('voice_command_faster_whisper_model_size', 'small')
        self.declare_parameter(
            'voice_command_faster_whisper_base_model_path',
            DEFAULT_BASE_MODEL_PATH,
        )
        self.declare_parameter(
            'voice_command_faster_whisper_small_model_path',
            DEFAULT_SMALL_MODEL_PATH,
        )
        self.declare_parameter(
            'voice_command_faster_whisper_beam_size',
            DEFAULT_FASTER_WHISPER_BEAM_SIZE,
        )
        self.declare_parameter(
            'voice_command_faster_whisper_cpu_threads',
            DEFAULT_FASTER_WHISPER_CPU_THREADS,
        )
        self.declare_parameter(
            'voice_command_vosk_model_path',
            DEFAULT_VOSK_MODEL_PATH,
        )
        self.declare_parameter('voice_command_device_id', DEFAULT_ASR_DEVICE_ID)
        self.declare_parameter('voice_command_sample_rate', DEFAULT_ASR_SAMPLE_RATE)
        self.declare_parameter('voice_command_channels', DEFAULT_ASR_CHANNELS)
        self.declare_parameter('voice_command_sample_format', DEFAULT_ASR_SAMPLE_FORMAT)
        self.declare_parameter('voice_command_chunk_size', DEFAULT_ASR_CHUNK_SIZE)
        self.declare_parameter('voice_command_timeout_sec', 0.0)
        self.declare_parameter('voice_any_speech_rms_threshold', 100.0)
        self.declare_parameter('voice_any_speech_consecutive_chunks', 3)
        self.declare_parameter('voice_noise_calibration_sec', 0.5)
        self.declare_parameter('voice_threshold_multiplier', 3.0)
        self.declare_parameter('voice_pre_roll_sec', 0.3)
        self.declare_parameter('voice_end_silence_sec', 0.8)
        self.declare_parameter('voice_min_utterance_sec', 0.35)
        self.declare_parameter('voice_max_utterance_sec', 10.0)
        self.declare_parameter('voice_max_digital_gain', 4.0)
        self.declare_parameter('voice_text_match_mode', 'command')
        self.declare_parameter('voice_text_match_percentage', 65.0)
        self.declare_parameter('voice_accept_any_recognized_text', False)
        self.declare_parameter('mid_obstacle_speech', '前方发现障碍物，正在绕行，请小心行走')
        self.declare_parameter('sign_recognition_enabled', True)
        self.declare_parameter('sign_capture_output_dir', '')
        self.declare_parameter('sign_camera_rtsp_urls', DEFAULT_RTSP_URLS)
        self.declare_parameter('sign_camera_warmup_frames', 10)
        self.declare_parameter('sign_camera_open_timeout_sec', 5.0)
        self.declare_parameter('sign_vote_frame_count', 20)
        self.declare_parameter('sign_vote_frame_interval_sec', 0.1)
        self.declare_parameter('sign_min_valid_frames', 3)
        self.declare_parameter('sign_min_char_vote_ratio', 0.55)
        self.declare_parameter('sign_min_color_vote_ratio', 0.50)
        self.declare_parameter('sign_save_all_frames', False)
        self.declare_parameter('sign_preprocess_enabled', False)
        self.declare_parameter('sign_preprocess_saturation_threshold', 100)
        self.declare_parameter('sign_digit_position', 'right')
        self.declare_parameter('speech_voice', 'cmn')
        self.declare_parameter('speech_speed', 175)
        self.declare_parameter('speech_pitch', 50)
        self.declare_parameter('speech_volume', 100)
        self.declare_parameter('speech_player', 'auto')
        self.declare_parameter('speech_device', '')
        self.declare_parameter('moving_obstacle_speech', '注意前方有移动物体，正在避让')
        self.declare_parameter('moving_obstacle_scan_topic', '/scan')
        self.declare_parameter('moving_obstacle_enable_on_a_to_b', True)
        self.declare_parameter('moving_obstacle_enable_on_b_to_c', True)
        self.declare_parameter('a_to_b_obstacle_speech_max_count', 1)
        self.declare_parameter('b_to_c_obstacle_speech_max_count', 1)
        self.declare_parameter('moving_obstacle_distance_threshold', 2.8)
        self.declare_parameter('moving_obstacle_delta_threshold', 0.08)
        self.declare_parameter('moving_obstacle_front_angle_deg', 24.0)
        self.declare_parameter('moving_obstacle_cooldown_sec', 4.0)
        self.declare_parameter('dynamic_avoidance_profiles_enabled', True)
        self.declare_parameter('avoidance_parameter_timeout_sec', 3.0)
        self.declare_parameter('endpoint_inflation_radius', 0.18)
        self.declare_parameter('endpoint_cost_scaling_factor', 25.0)
        self.declare_parameter('endpoint_obstacle_scale', 0.01)
        self.declare_parameter('endpoint_max_linear_speed', 0.16)
        self.declare_parameter('endpoint_max_angular_speed', 0.25)
        self.declare_parameter('nav2_xy_goal_tolerance', 0.20)
        self.declare_parameter('endpoint_yaw_goal_tolerance', 0.20)
        self.declare_parameter('endpoint_profile_switch_distance', 0.90)
        self.declare_parameter('c_departure_profile_distance', 0.45)
        self.declare_parameter('c_departure_inflation_radius', 0.16)
        self.declare_parameter('c_departure_cost_scaling_factor', 20.0)
        self.declare_parameter('c_departure_max_linear_speed', 0.20)
        self.declare_parameter('c_departure_max_angular_speed', 0.38)
        self.declare_parameter('b_to_c_footprint_padding', 0.04)
        self.declare_parameter('b_to_c_inflation_radius', 0.55)
        self.declare_parameter('b_to_c_global_inflation_radius', 0.70)
        self.declare_parameter('b_to_c_cost_scaling_factor', 5.5)
        self.declare_parameter('b_to_c_obstacle_scale', 0.5)
        self.declare_parameter('b_to_c_path_align_scale', 4.0)
        self.declare_parameter('b_to_c_goal_align_scale', 6.0)
        self.declare_parameter('b_to_c_path_dist_scale', 6.0)
        self.declare_parameter('b_to_c_goal_dist_scale', 32.0)
        self.declare_parameter('b_to_c_min_linear_speed', 0.04)
        self.declare_parameter('b_to_c_max_linear_speed', 0.13)
        self.declare_parameter('b_to_c_max_angular_speed', 0.30)
        self.declare_parameter('b_to_c_forward_point_distance', 0.35)
        self.declare_parameter('b_to_c_sim_time', 2.0)
        self.declare_parameter('b_departure_profile_distance', 0.60)
        self.declare_parameter('b_x', math.nan)
        self.declare_parameter('b_y', math.nan)
        self.declare_parameter('b_yaw', math.nan)
        self.declare_parameter('b_arrival_speech', '已到达b点')
        self.declare_parameter('wait_at_b_sec', 2.0)
        self.declare_parameter('c_x', math.nan)
        self.declare_parameter('c_y', math.nan)
        self.declare_parameter('c_yaw', math.nan)
        self.declare_parameter('c_arrival_speech', '已到达c点')
        self.declare_parameter('task5_enabled', True)
        self.declare_parameter('qrcode_capture_output_dir', '')
        self.declare_parameter('qrcode_camera_rtsp_urls', DEFAULT_RTSP_URLS)
        self.declare_parameter('qrcode_camera_warmup_frames', 10)
        self.declare_parameter('qrcode_camera_open_timeout_sec', 5.0)
        self.declare_parameter('qrcode_vote_frame_count', 20)
        self.declare_parameter('qrcode_vote_frame_interval_sec', 0.1)
        self.declare_parameter('qrcode_min_valid_votes', 2)
        self.declare_parameter('qrcode_recognition_max_attempts', 3)
        self.declare_parameter('task6_enabled', True)
        self.declare_parameter('traffic_light_capture_output_dir', '')
        self.declare_parameter('traffic_light_camera_rtsp_urls', DEFAULT_RTSP_URLS)
        self.declare_parameter('traffic_light_camera_warmup_frames', 10)
        self.declare_parameter('traffic_light_camera_open_timeout_sec', 5.0)
        self.declare_parameter('traffic_light_vote_frame_count', 15)
        self.declare_parameter('traffic_light_vote_frame_interval_sec', 0.1)
        self.declare_parameter('traffic_light_min_valid_votes', 3)
        self.declare_parameter('traffic_light_min_confidence', 0.55)
        self.declare_parameter('traffic_light_saturation_threshold', 80)
        self.declare_parameter('traffic_light_value_threshold', 145)
        self.declare_parameter('traffic_light_min_component_area', 20)
        self.declare_parameter('traffic_light_recognition_max_attempts', 3)
        self.declare_parameter('path_option', 1)
        self.declare_parameter('path1_mid_x', math.nan)
        self.declare_parameter('path1_mid_y', math.nan)
        self.declare_parameter('path1_mid_yaw', math.nan)
        self.declare_parameter('path2_mid_x', math.nan)
        self.declare_parameter('path2_mid_y', math.nan)
        self.declare_parameter('path2_mid_yaw', math.nan)
        self.declare_parameter('nav_timeout_sec', 300.0)
        self.declare_parameter('initial_pose_republish_count', 5)
        self.declare_parameter('initial_pose_republish_interval_sec', 1.0)
        self.declare_parameter('wait_for_amcl_timeout_sec', 20.0)
        self.declare_parameter('wait_for_nav_server_timeout_sec', 60.0)
        self.declare_parameter('wait_for_nav_active_timeout_sec', 60.0)
        self.declare_parameter('stage_reach_confirmation_enabled', True)
        self.declare_parameter('stage_reach_xy_tolerance', 0.30)
        self.declare_parameter('stage_reach_fallback_xy_tolerance', 0.45)
        self.declare_parameter('stage_reach_stable_sec', 0.5)
        self.declare_parameter('stage_reach_timeout_sec', 30.0)

        self._amcl_pose = None
        self._goal_feedback_distance = None
        self._latest_scan = None
        self._moving_obstacle_monitor_enabled = False
        self._last_moving_obstacle_min_range = None
        self._last_moving_obstacle_speech_time = 0.0
        self._moving_obstacle_speech_parameter = 'moving_obstacle_speech'
        self._moving_obstacle_segment = ''
        self._moving_obstacle_speech_count = 0
        self._moving_obstacle_speech_max_count = -1
        self._active_avoidance_profile = ''
        self._nav_parameter_clients = {}
        self._lifecycle_state_clients = {}
        self._voice_listener_preloaded = False
        self._sign_classifier: CharacterClassifier | None = None
        self._qrcode_detector: cv2.QRCodeDetector | None = None

        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            transient_qos,
        )
        self._amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_pose_callback,
            10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._scan_sub = self.create_subscription(
            LaserScan,
            str(self.get_parameter('moving_obstacle_scan_topic').value),
            self._scan_callback,
            sensor_qos,
        )
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._nav_through_poses_client = ActionClient(
            self,
            NavigateThroughPoses,
            'navigate_through_poses',
        )
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._preload_vision_models()

        if (
            bool(self.get_parameter('voice_command_enabled').value)
            and not bool(self.get_parameter('debug_mode').value)
        ):
            self._preload_voice_listener()

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._amcl_pose = msg

    def _scan_callback(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._check_for_moving_obstacle_speech(msg)

    def _feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self._goal_feedback_distance = feedback.distance_remaining
        self.get_logger().info(
            f'navigation feedback: distance_remaining={feedback.distance_remaining:.3f} m'
        )

    def _preload_vision_models(self) -> None:
        started_at = time.monotonic()
        self.get_logger().info('preloading visual recognition models into memory')
        self._sign_classifier = CharacterClassifier()
        self._qrcode_detector = cv2.QRCodeDetector()
        elapsed = time.monotonic() - started_at
        self.get_logger().info(
            f'visual recognition models preloaded in {elapsed:.2f} seconds: '
            'sign character classifier and QR code detector ready; '
            'traffic light recognition uses the pre-imported OpenCV color pipeline'
        )

    def _preload_voice_listener(self) -> None:
        if self._voice_listener_preloaded:
            return

        try:
            backend = normalize_backend(
                str(self.get_parameter('voice_command_backend').value)
            )
            preload_listener(
                model_path=self._voice_model_path(backend),
                device_id=int(self.get_parameter('voice_command_device_id').value),
                sample_rate=int(self.get_parameter('voice_command_sample_rate').value),
                channels=int(self.get_parameter('voice_command_channels').value),
                sample_format=str(self.get_parameter('voice_command_sample_format').value),
                chunk_size=int(self.get_parameter('voice_command_chunk_size').value),
                backend=backend,
                beam_size=int(
                    self.get_parameter('voice_command_faster_whisper_beam_size').value
                ),
                cpu_threads=int(
                    self.get_parameter('voice_command_faster_whisper_cpu_threads').value
                ),
                noise_calibration_sec=float(
                    self.get_parameter('voice_noise_calibration_sec').value
                ),
            )
            self._voice_listener_preloaded = True
            self.get_logger().info(f'voice recognition model preloaded: backend={backend}')
        except Exception as exc:
            self.get_logger().warning(f'failed to preload voice recognition model: {exc}')

    def _voice_model_path(self, backend: str) -> str:
        if backend == 'vosk':
            return str(self.get_parameter('voice_command_vosk_model_path').value).strip()

        model_size = str(
            self.get_parameter('voice_command_faster_whisper_model_size').value
        ).strip().lower()
        if model_size not in {'base', 'small'}:
            raise ValueError(
                'voice_command_faster_whisper_model_size must be base or small; '
                f'got {model_size!r}'
            )
        parameter_name = f'voice_command_faster_whisper_{model_size}_model_path'
        configured = str(self.get_parameter(parameter_name).value).strip()
        if configured:
            return configured
        return str(
            self.get_parameter('voice_command_faster_whisper_model_path').value
        ).strip()

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
        half_yaw = yaw * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def validate_waypoint_parameters(self) -> None:
        invalid = [
            name
            for name in WAYPOINT_PARAMETER_NAMES
            if not math.isfinite(float(self.get_parameter(name).value))
        ]
        if invalid:
            raise RuntimeError(
                'waypoint configuration is missing or invalid parameters: '
                + ', '.join(invalid)
            )
        self.get_logger().info('all waypoint parameters loaded from configuration')

    def publish_initial_pose(self) -> None:
        self.validate_waypoint_parameters()
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(self.get_parameter('initial_x').value)
        msg.pose.pose.position.y = float(self.get_parameter('initial_y').value)
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = self._yaw_to_quaternion(
            float(self.get_parameter('initial_yaw').value)
        )
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942,
        ]

        republish_count = int(self.get_parameter('initial_pose_republish_count').value)
        republish_interval = float(
            self.get_parameter('initial_pose_republish_interval_sec').value
        )
        for idx in range(republish_count):
            msg.header.stamp = self.get_clock().now().to_msg()
            self._initial_pose_pub.publish(msg)
            self.get_logger().info(
                f'published initial pose {idx + 1}/{republish_count}: '
                f'x={msg.pose.pose.position.x:.3f}, '
                f'y={msg.pose.pose.position.y:.3f}, '
                f'yaw={float(self.get_parameter("initial_yaw").value):.3f}'
            )
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(republish_interval)

    def wait_for_amcl_pose(self) -> bool:
        timeout = float(self.get_parameter('wait_for_amcl_timeout_sec').value)
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._amcl_pose is not None:
                pose = self._amcl_pose.pose.pose.position
                self.get_logger().info(
                    f'AMCL pose received: x={pose.x:.3f}, y={pose.y:.3f}'
                )
                return True
        self.get_logger().warning(
            f'no /amcl_pose received within {timeout:.1f} seconds'
        )
        return False

    def wait_for_navigation_server(self) -> bool:
        timeout = float(self.get_parameter('wait_for_nav_server_timeout_sec').value)
        ready = self._nav_client.wait_for_server(timeout_sec=timeout)
        if not ready:
            self.get_logger().error(
                f'navigate_to_pose action server not ready within {timeout:.1f} seconds'
            )
            return False
        through_poses_ready = self._nav_through_poses_client.wait_for_server(
            timeout_sec=timeout
        )
        if not through_poses_ready:
            self.get_logger().error(
                f'navigate_through_poses action server not ready within {timeout:.1f} seconds'
            )
            return False
        self.get_logger().info('navigate_to_pose action server is available')
        return self.wait_for_navigation_nodes_active()

    def _lifecycle_state(self, node_name: str, timeout_sec: float) -> tuple[int, str] | None:
        client = self._lifecycle_state_clients.get(node_name)
        if client is None:
            client = self.create_client(
                GetState,
                f'{node_name.rstrip("/")}/get_state',
            )
            self._lifecycle_state_clients[node_name] = client
        if not client.wait_for_service(timeout_sec=min(timeout_sec, 1.0)):
            return None
        future = client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=min(timeout_sec, 1.0),
        )
        if not future.done() or future.result() is None:
            return None
        state = future.result().current_state
        return int(state.id), str(state.label)

    def wait_for_navigation_nodes_active(self) -> bool:
        node_names = (
            '/map_server',
            '/amcl',
            '/bt_navigator',
            '/planner_server',
            '/controller_server',
        )
        timeout = float(self.get_parameter('wait_for_nav_active_timeout_sec').value)
        deadline = time.time() + timeout
        last_states = {}
        while rclpy.ok() and time.time() < deadline:
            all_active = True
            for node_name in node_names:
                remaining = max(deadline - time.time(), 0.0)
                state = self._lifecycle_state(node_name, remaining)
                last_states[node_name] = 'unavailable' if state is None else state[1]
                if state is None or state[0] != State.PRIMARY_STATE_ACTIVE:
                    all_active = False
            if all_active:
                self.get_logger().info('all navigation lifecycle nodes are active')
                return True
            rclpy.spin_once(self, timeout_sec=0.2)
            time.sleep(0.3)
        state_text = ', '.join(
            f'{name}={last_states.get(name, "unknown")}' for name in node_names
        )
        self.get_logger().error(
            f'navigation lifecycle nodes did not become active within {timeout:.1f} seconds: '
            f'{state_text}'
        )
        return False

    def _set_nav2_parameters(self, node_name: str, values: dict) -> bool:
        client = self._nav_parameter_clients.get(node_name)
        if client is None:
            service_name = f'{node_name.rstrip("/")}/set_parameters'
            client = self.create_client(SetParameters, service_name)
            self._nav_parameter_clients[node_name] = client
        timeout = float(self.get_parameter('avoidance_parameter_timeout_sec').value)
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warning(
                f'parameter service unavailable for {node_name}; keeping previous avoidance values'
            )
            return False
        request = SetParameters.Request()
        request.parameters = [
            Parameter(name=name, value=value).to_parameter_msg()
            for name, value in values.items()
        ]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().warning(f'timed out setting avoidance parameters on {node_name}')
            return False
        response = future.result()
        results = getattr(response, 'results', response if isinstance(response, list) else [])
        if len(results) != len(values):
            self.get_logger().warning(
                f'{node_name} returned {len(results)} results for {len(values)} parameters'
            )
            return False
        rejected = [
            name
            for name, result in zip(values, results)
            if not getattr(result, 'successful', False)
        ]
        if rejected:
            self.get_logger().warning(
                f'{node_name} rejected avoidance parameters: {", ".join(rejected)}'
            )
            return False
        return True

    def apply_avoidance_profile(self, profile: str) -> None:
        if not bool(self.get_parameter('dynamic_avoidance_profiles_enabled').value):
            return
        if profile == self._active_avoidance_profile:
            return
        xy_tolerance = float(self.get_parameter('nav2_xy_goal_tolerance').value)
        if profile == 'endpoint':
            footprint_padding = 0.01
            inflation_radius = float(self.get_parameter('endpoint_inflation_radius').value)
            global_inflation_radius = inflation_radius
            cost_scaling = float(self.get_parameter('endpoint_cost_scaling_factor').value)
            obstacle_scale = float(self.get_parameter('endpoint_obstacle_scale').value)
            path_align_scale = 8.0
            goal_align_scale = 8.0
            path_dist_scale = 12.0
            goal_dist_scale = 24.0
            min_linear = 0.0
            forward_point_distance = 0.20
            max_linear = float(self.get_parameter('endpoint_max_linear_speed').value)
            max_angular = float(self.get_parameter('endpoint_max_angular_speed').value)
            sim_time = 1.2
            yaw_tolerance = float(self.get_parameter('endpoint_yaw_goal_tolerance').value)
        elif profile == 'c_departure':
            footprint_padding = 0.01
            inflation_radius = float(self.get_parameter('c_departure_inflation_radius').value)
            global_inflation_radius = inflation_radius
            cost_scaling = float(self.get_parameter('c_departure_cost_scaling_factor').value)
            obstacle_scale = 0.03
            path_align_scale = 16.0
            goal_align_scale = 12.0
            path_dist_scale = 16.0
            goal_dist_scale = 22.0
            min_linear = 0.0
            forward_point_distance = 0.20
            max_linear = float(self.get_parameter('c_departure_max_linear_speed').value)
            max_angular = float(self.get_parameter('c_departure_max_angular_speed').value)
            sim_time = 1.0
            yaw_tolerance = 0.20
        elif profile == 'b_to_c':
            footprint_padding = float(
                self.get_parameter('b_to_c_footprint_padding').value
            )
            inflation_radius = float(self.get_parameter('b_to_c_inflation_radius').value)
            global_inflation_radius = float(
                self.get_parameter('b_to_c_global_inflation_radius').value
            )
            cost_scaling = float(self.get_parameter('b_to_c_cost_scaling_factor').value)
            obstacle_scale = float(self.get_parameter('b_to_c_obstacle_scale').value)
            path_align_scale = float(
                self.get_parameter('b_to_c_path_align_scale').value
            )
            goal_align_scale = float(
                self.get_parameter('b_to_c_goal_align_scale').value
            )
            path_dist_scale = float(
                self.get_parameter('b_to_c_path_dist_scale').value
            )
            goal_dist_scale = float(
                self.get_parameter('b_to_c_goal_dist_scale').value
            )
            min_linear = float(
                self.get_parameter('b_to_c_min_linear_speed').value
            )
            forward_point_distance = float(
                self.get_parameter('b_to_c_forward_point_distance').value
            )
            max_linear = float(self.get_parameter('b_to_c_max_linear_speed').value)
            max_angular = float(self.get_parameter('b_to_c_max_angular_speed').value)
            sim_time = float(self.get_parameter('b_to_c_sim_time').value)
            yaw_tolerance = 0.20
        elif profile == 'normal':
            footprint_padding = 0.01
            inflation_radius = 0.18
            global_inflation_radius = 0.20
            cost_scaling = 13.0
            obstacle_scale = 0.08
            path_align_scale = 24.0
            goal_align_scale = 16.0
            path_dist_scale = 24.0
            goal_dist_scale = 18.0
            min_linear = 0.0
            forward_point_distance = 0.20
            max_linear = 0.18
            max_angular = 0.35
            sim_time = 1.2
            yaw_tolerance = 0.20
        else:
            raise ValueError(f'unknown avoidance profile: {profile}')

        updates = (
            self._set_nav2_parameters('/controller_server', {
                'FollowPath.BaseObstacle.scale': obstacle_scale,
                'FollowPath.PathAlign.scale': path_align_scale,
                'FollowPath.GoalAlign.scale': goal_align_scale,
                'FollowPath.PathDist.scale': path_dist_scale,
                'FollowPath.GoalDist.scale': goal_dist_scale,
                'FollowPath.PathAlign.forward_point_distance': forward_point_distance,
                'FollowPath.GoalAlign.forward_point_distance': forward_point_distance,
                'FollowPath.min_vel_x': min_linear,
                'FollowPath.min_speed_xy': min_linear,
                'FollowPath.max_vel_x': max_linear,
                'FollowPath.max_speed_xy': max_linear,
                'FollowPath.max_vel_theta': max_angular,
                'FollowPath.sim_time': sim_time,
                'general_goal_checker.xy_goal_tolerance': xy_tolerance,
                'general_goal_checker.yaw_goal_tolerance': yaw_tolerance,
            }),
            self._set_nav2_parameters('/local_costmap/local_costmap', {
                'footprint_padding': footprint_padding,
                'inflation_layer.inflation_radius': inflation_radius,
                'inflation_layer.cost_scaling_factor': cost_scaling,
            }),
            self._set_nav2_parameters('/global_costmap/global_costmap', {
                'inflation_layer.inflation_radius': global_inflation_radius,
                'inflation_layer.cost_scaling_factor': cost_scaling,
            }),
            self._set_nav2_parameters('/velocity_smoother', {
                'max_velocity': [max_linear, 0.0, max_angular],
                'min_velocity': [-max_linear, 0.0, -max_angular],
            }),
        )
        if all(updates):
            self._active_avoidance_profile = profile
            self.get_logger().info(
                f'applied avoidance profile={profile}: padding={footprint_padding:.2f} m, '
                f'local/global inflation={inflation_radius:.2f}/{global_inflation_radius:.2f} m, '
                f'obstacle_scale={obstacle_scale:.2f}, path_align={path_align_scale:.1f}, '
                f'linear={min_linear:.2f}-{max_linear:.2f} m/s'
            )
        else:
            self._active_avoidance_profile = ''
            self.get_logger().warning(
                f'avoidance profile {profile} was only partially applied'
            )

    def wait_until_pose_reached(self, target_name: str, x: float, y: float) -> bool:
        if not bool(self.get_parameter('stage_reach_confirmation_enabled').value):
            self.get_logger().info(
                f'[{target_name}] stage_reach confirmation disabled; '
                'accepting Nav2 navigation result directly'
            )
            return True

        tolerance = float(self.get_parameter('stage_reach_xy_tolerance').value)
        fallback_tolerance = float(
            self.get_parameter('stage_reach_fallback_xy_tolerance').value
        )
        stable_sec = float(self.get_parameter('stage_reach_stable_sec').value)
        timeout_sec = float(self.get_parameter('stage_reach_timeout_sec').value)
        deadline = time.time() + timeout_sec
        stable_start = None
        min_dist = math.inf

        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._amcl_pose is None:
                continue

            pose = self._amcl_pose.pose.pose.position
            dist = math.hypot(pose.x - x, pose.y - y)
            min_dist = min(min_dist, dist)
            if dist <= tolerance:
                if stable_start is None:
                    stable_start = time.time()
                    self.get_logger().info(
                        f'[{target_name}] entered reach tolerance: distance={dist:.3f} m'
                    )
                elif time.time() - stable_start >= stable_sec:
                    self.get_logger().info(
                        f'[{target_name}] pose stable within tolerance for {stable_sec:.1f} seconds'
                    )
                    return True
            else:
                stable_start = None

        feedback_dist = self._goal_feedback_distance
        min_dist_text = 'n/a' if math.isinf(min_dist) else f'{min_dist:.3f}'
        feedback_text = 'n/a' if feedback_dist is None else f'{feedback_dist:.3f}'

        if (
            not math.isinf(min_dist)
            and min_dist <= fallback_tolerance
        ) or (
            feedback_dist is not None
            and feedback_dist <= fallback_tolerance
        ):
            self.get_logger().warning(
                f'[{target_name}] pose not stable within {timeout_sec:.1f} seconds, '
                f'but closest distance={min_dist_text} m, '
                f'last feedback distance={feedback_text} m, '
                f'within fallback tolerance {fallback_tolerance:.3f} m; continuing mission'
            )
            return True

        self.get_logger().warning(
            f'[{target_name}] pose did not become stable near target within {timeout_sec:.1f} seconds '
            f'(closest distance={min_dist_text} m, last feedback distance={feedback_text} m); '
            'continuing mission without abort'
        )
        return True

    def _make_pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(self.get_parameter('goal_frame').value)
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = self._yaw_to_quaternion(float(yaw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def send_goal(
        self,
        target_name: str,
        x: float,
        y: float,
        yaw: float,
        near_target_profile: str = '',
        profile_switch_distance: float = 0.0,
        departure_profile: str = '',
        departure_origin: tuple[float, float] | None = None,
        departure_switch_distance: float = 0.0,
    ) -> int:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_pose_stamped(x, y, yaw)

        self.get_logger().info(
            f'sending navigation goal [{target_name}]: '
            f'frame={goal_msg.pose.header.frame_id}, '
            f'x={goal_msg.pose.pose.position.x:.3f}, '
            f'y={goal_msg.pose.pose.position.y:.3f}, '
            f'yaw={float(yaw):.3f}'
        )
        send_goal_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.get_logger().error('failed to send goal to navigate_to_pose')
            return GoalStatus.STATUS_UNKNOWN
        if not goal_handle.accepted:
            self.get_logger().error('navigation goal was rejected')
            return GoalStatus.STATUS_ABORTED

        self.get_logger().info('navigation goal accepted')
        result_future = goal_handle.get_result_async()
        timeout = float(self.get_parameter('nav_timeout_sec').value)
        if near_target_profile or departure_profile:
            deadline = time.time() + timeout
            near_target_profile_attempted = not bool(near_target_profile)
            departure_profile_attempted = not bool(departure_profile)
            while rclpy.ok() and not result_future.done() and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self._amcl_pose is None:
                    continue
                position = self._amcl_pose.pose.pose.position
                if (
                    not departure_profile_attempted
                    and departure_origin is not None
                ):
                    departure_distance = math.hypot(
                        position.x - departure_origin[0],
                        position.y - departure_origin[1],
                    )
                    if departure_distance >= departure_switch_distance:
                        departure_profile_attempted = True
                        self.get_logger().info(
                            f'goal [{target_name}] has departed '
                            f'{departure_distance:.3f} m; switching to avoidance '
                            f'profile={departure_profile}'
                        )
                        self.apply_avoidance_profile(departure_profile)
                distance = math.hypot(position.x - x, position.y - y)
                if (
                    not near_target_profile_attempted
                    and distance <= profile_switch_distance
                ):
                    near_target_profile_attempted = True
                    self.get_logger().info(
                        f'goal [{target_name}] is {distance:.3f} m away; '
                        f'switching to avoidance profile={near_target_profile}'
                    )
                    self.apply_avoidance_profile(near_target_profile)
        else:
            rclpy.spin_until_future_complete(
                self,
                result_future,
                timeout_sec=timeout,
            )
        if not result_future.done():
            self.get_logger().error(
                f'navigation did not finish within {timeout:.1f} seconds, canceling goal'
            )
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)
            return GoalStatus.STATUS_CANCELED

        result = result_future.result()
        if result is None:
            self.get_logger().error('navigation finished without a valid result')
            return GoalStatus.STATUS_UNKNOWN

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('navigation succeeded')
        else:
            self.get_logger().error(f'navigation finished with status={status}')
        return status

    def send_pose_sequence(
        self,
        target_name: str,
        poses: list[tuple[float, float, float]],
        final_profile: str = '',
        final_profile_switch_distance: float = 0.0,
    ) -> int:
        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses = [
            self._make_pose_stamped(x, y, yaw) for x, y, yaw in poses
        ]
        self.get_logger().info(
            f'sending continuous navigation route [{target_name}] with {len(poses)} poses'
        )
        send_goal_future = self._nav_through_poses_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.get_logger().error('failed to send goal to navigate_through_poses')
            return GoalStatus.STATUS_UNKNOWN
        if not goal_handle.accepted:
            self.get_logger().error('continuous navigation route was rejected')
            return GoalStatus.STATUS_ABORTED

        result_future = goal_handle.get_result_async()
        timeout = float(self.get_parameter('nav_timeout_sec').value)
        if final_profile and poses:
            final_x, final_y, _ = poses[-1]
            deadline = time.time() + timeout
            profile_attempted = False
            while rclpy.ok() and not result_future.done() and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if profile_attempted or self._amcl_pose is None:
                    continue
                position = self._amcl_pose.pose.pose.position
                distance = math.hypot(position.x - final_x, position.y - final_y)
                if distance <= final_profile_switch_distance:
                    profile_attempted = True
                    self.get_logger().info(
                        f'route [{target_name}] is {distance:.3f} m from final pose; '
                        f'switching to avoidance profile={final_profile}'
                    )
                    self.apply_avoidance_profile(final_profile)
        else:
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error(
                f'continuous navigation did not finish within {timeout:.1f} seconds'
            )
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)
            return GoalStatus.STATUS_CANCELED
        result = result_future.result()
        if result is None:
            return GoalStatus.STATUS_UNKNOWN
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'continuous navigation route [{target_name}] succeeded')
        else:
            self.get_logger().error(
                f'continuous navigation route [{target_name}] status={result.status}'
            )
        return result.status

    def speak(self, text: str) -> None:
        device = str(self.get_parameter('speech_device').value).strip()
        speak_text(
            text=text,
            voice=str(self.get_parameter('speech_voice').value),
            speed=int(self.get_parameter('speech_speed').value),
            pitch=int(self.get_parameter('speech_pitch').value),
            volume=int(self.get_parameter('speech_volume').value),
            player=str(self.get_parameter('speech_player').value),
            device=device or None,
        )

    def stop_robot(self) -> None:
        stop_command = Twist()
        for _ in range(5):
            self._cmd_vel_pub.publish(stop_command)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.03)
        self.get_logger().info('published zero velocity command; robot stopped')

    def _sign_output_dir(self) -> Path:
        configured = str(self.get_parameter('sign_capture_output_dir').value).strip()
        if configured:
            return Path(configured).expanduser()
        return Path.home() / '.cache' / 'agibot_d1_auto_nav' / 'captures'

    def _qrcode_output_dir(self) -> Path:
        configured = str(self.get_parameter('qrcode_capture_output_dir').value).strip()
        if configured:
            return Path(configured).expanduser()
        return self._sign_output_dir()

    def _traffic_light_output_dir(self) -> Path:
        configured = str(
            self.get_parameter('traffic_light_capture_output_dir').value
        ).strip()
        if configured:
            return Path(configured).expanduser()
        return self._sign_output_dir()

    def wait_for_voice_command(
        self,
        parameter_name: str,
        accept_any_speech: bool = False,
    ) -> None:
        if bool(self.get_parameter('debug_mode').value):
            self._wait_for_debug_space(parameter_name)
            return
        if not bool(self.get_parameter('voice_command_enabled').value):
            return

        expected_phrase = str(self.get_parameter(parameter_name).value).strip()
        if not expected_phrase:
            return

        self._preload_voice_listener()

        last_logged_text = {'value': None}

        def _log_recognized_text(text: str) -> None:
            if text == last_logged_text['value']:
                return
            last_logged_text['value'] = text
            self.get_logger().info(
                f'voice recognition heard: {text}'
            )

        self.get_logger().info(
            f'waiting for recognized speech text ({parameter_name}); '
            f'prompt phrase={expected_phrase}'
        )
        backend = normalize_backend(
            str(self.get_parameter('voice_command_backend').value)
        )
        matched_text = wait_for_command(
            expected_phrase=expected_phrase,
            model_path=self._voice_model_path(backend),
            device_id=int(self.get_parameter('voice_command_device_id').value),
            sample_rate=int(self.get_parameter('voice_command_sample_rate').value),
            channels=int(self.get_parameter('voice_command_channels').value),
            sample_format=str(self.get_parameter('voice_command_sample_format').value),
            chunk_size=int(self.get_parameter('voice_command_chunk_size').value),
            timeout_sec=float(self.get_parameter('voice_command_timeout_sec').value),
            on_text=_log_recognized_text,
            accept_any_speech=accept_any_speech,
            accept_any_recognized_text=bool(
                self.get_parameter('voice_accept_any_recognized_text').value
            ),
            speech_rms_threshold=float(
                self.get_parameter('voice_any_speech_rms_threshold').value
            ),
            speech_consecutive_chunks=int(
                self.get_parameter('voice_any_speech_consecutive_chunks').value
            ),
            noise_calibration_sec=float(
                self.get_parameter('voice_noise_calibration_sec').value
            ),
            threshold_multiplier=float(
                self.get_parameter('voice_threshold_multiplier').value
            ),
            pre_roll_sec=float(self.get_parameter('voice_pre_roll_sec').value),
            end_silence_sec=float(
                self.get_parameter('voice_end_silence_sec').value
            ),
            min_utterance_sec=float(
                self.get_parameter('voice_min_utterance_sec').value
            ),
            max_utterance_sec=float(
                self.get_parameter('voice_max_utterance_sec').value
            ),
            max_digital_gain=float(
                self.get_parameter('voice_max_digital_gain').value
            ),
            text_match_mode=str(
                self.get_parameter('voice_text_match_mode').value
            ),
            text_match_percentage=float(
                self.get_parameter('voice_text_match_percentage').value
            ),
            backend=backend,
            beam_size=int(
                self.get_parameter('voice_command_faster_whisper_beam_size').value
            ),
            cpu_threads=int(
                self.get_parameter('voice_command_faster_whisper_cpu_threads').value
            ),
        )
        self.get_logger().info(
            f'voice command matched for {parameter_name}: {matched_text}'
        )

    def _wait_for_debug_space(self, parameter_name: str) -> None:
        expected_phrase = str(self.get_parameter(parameter_name).value).strip()
        self.get_logger().warning(
            f'debug mode: press SPACE to pass voice command [{parameter_name}] '
            f'(prompt={expected_phrase})'
        )

        fd = None
        close_fd = False
        try:
            try:
                fd = os.open('/dev/tty', os.O_RDONLY)
                close_fd = True
            except OSError:
                if not sys.stdin.isatty():
                    raise RuntimeError(
                        'debug mode requires an interactive terminal to read SPACE'
                    )
                fd = sys.stdin.fileno()

            previous_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while rclpy.ok():
                    readable, _, _ = select.select([fd], [], [], 0.2)
                    if not readable:
                        continue
                    if os.read(fd, 1) == b' ':
                        self.get_logger().info(
                            f'debug SPACE accepted for voice command [{parameter_name}]'
                        )
                        return
                raise RuntimeError('ROS stopped while waiting for debug SPACE')
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)
        finally:
            if close_fd and fd is not None:
                os.close(fd)

    def _extract_front_min_range(self, scan: LaserScan) -> float | None:
        if scan.angle_increment == 0.0:
            return None

        front_angle_deg = float(self.get_parameter('moving_obstacle_front_angle_deg').value)
        front_angle_rad = math.radians(front_angle_deg)
        ranges = []
        for idx, distance in enumerate(scan.ranges):
            angle = scan.angle_min + idx * scan.angle_increment
            if abs(angle) > front_angle_rad:
                continue
            if math.isfinite(distance) and scan.range_min < distance < scan.range_max:
                ranges.append(distance)

        if not ranges:
            return None
        return min(ranges)

    def _check_for_moving_obstacle_speech(self, scan: LaserScan | None = None) -> None:
        if not self._moving_obstacle_monitor_enabled:
            return
        if (
            self._moving_obstacle_speech_max_count >= 0
            and self._moving_obstacle_speech_count
            >= self._moving_obstacle_speech_max_count
        ):
            return
        current_scan = scan if scan is not None else self._latest_scan
        if current_scan is None:
            return

        current_min_range = self._extract_front_min_range(current_scan)
        if current_min_range is None:
            return

        previous_min_range = self._last_moving_obstacle_min_range
        self._last_moving_obstacle_min_range = current_min_range
        if previous_min_range is None:
            return

        distance_threshold = float(self.get_parameter('moving_obstacle_distance_threshold').value)
        delta_threshold = float(self.get_parameter('moving_obstacle_delta_threshold').value)
        cooldown_sec = float(self.get_parameter('moving_obstacle_cooldown_sec').value)
        now = time.time()

        if current_min_range > distance_threshold:
            return
        if abs(previous_min_range - current_min_range) < delta_threshold:
            return
        if now - self._last_moving_obstacle_speech_time < cooldown_sec:
            return

        self._last_moving_obstacle_speech_time = now
        self._moving_obstacle_speech_count += 1
        self.get_logger().info(
            f'moving obstacle detected during {self._moving_obstacle_segment}: '
            f'front_min_range={current_min_range:.3f} m, '
            f'previous_front_min_range={previous_min_range:.3f} m, '
            f'speech_count={self._moving_obstacle_speech_count}/'
            f'{self._moving_obstacle_speech_max_count}'
        )
        try:
            self.speak(
                str(self.get_parameter(self._moving_obstacle_speech_parameter).value)
            )
        except Exception as exc:
            self.get_logger().warning(f'failed to speak moving obstacle message: {exc}')

    def _set_moving_obstacle_monitor(
        self,
        enabled: bool,
        speech_parameter: str = 'moving_obstacle_speech',
        segment: str = '',
        max_speech_count: int = -1,
    ) -> None:
        self._moving_obstacle_monitor_enabled = enabled
        self._last_moving_obstacle_min_range = None
        self._last_moving_obstacle_speech_time = 0.0
        self._moving_obstacle_speech_parameter = speech_parameter
        self._moving_obstacle_segment = segment
        self._moving_obstacle_speech_count = 0
        self._moving_obstacle_speech_max_count = int(max_speech_count)

    def recognize_sign_and_choose_path(self) -> int:
        if not bool(self.get_parameter('sign_recognition_enabled').value):
            current_path_option = int(self.get_parameter('path_option').value)
            self.get_logger().info(
                f'sign recognition disabled, keeping existing path_option={current_path_option}'
            )
            return current_path_option

        rtsp_urls = list(self.get_parameter('sign_camera_rtsp_urls').value)
        if not rtsp_urls:
            rtsp_urls = DEFAULT_RTSP_URLS

        result = capture_and_recognize_sign(
            output_dir=self._sign_output_dir(),
            rtsp_urls=[str(url) for url in rtsp_urls],
            warmup_frames=int(self.get_parameter('sign_camera_warmup_frames').value),
            open_timeout=float(self.get_parameter('sign_camera_open_timeout_sec').value),
            frame_count=int(self.get_parameter('sign_vote_frame_count').value),
            frame_interval_sec=float(
                self.get_parameter('sign_vote_frame_interval_sec').value
            ),
            min_valid_frames=int(
                self.get_parameter('sign_min_valid_frames').value
            ),
            min_char_vote_ratio=float(
                self.get_parameter('sign_min_char_vote_ratio').value
            ),
            min_color_vote_ratio=float(
                self.get_parameter('sign_min_color_vote_ratio').value
            ),
            save_all_frames=bool(
                self.get_parameter('sign_save_all_frames').value
            ),
            preprocess_enabled=bool(
                self.get_parameter('sign_preprocess_enabled').value
            ),
            preprocess_saturation_threshold=int(
                self.get_parameter('sign_preprocess_saturation_threshold').value
            ),
            digit_position=str(
                self.get_parameter('sign_digit_position').value
            ),
            classifier=self._sign_classifier,
        )

        self.get_logger().info(
            'sign recognized: '
            f'number={result.number}, '
            f'number_color={result.number_color}, '
            f'letter={result.letter}, '
            f'letter_color={result.letter_color}, '
            f'image={result.image_path}, '
            f'annotated={result.annotated_path}, '
            f'number_confidence={result.number_confidence:.2f}, '
            f'letter_confidence={result.letter_confidence:.2f}, '
            f'number_color_confidence={result.number_color_confidence:.2f}, '
            f'letter_color_confidence={result.letter_color_confidence:.2f}, '
            f'valid_frames={result.valid_frame_count}/{result.total_frame_count}'
        )

        speeches = [
            f'标识牌上的数字是{result.number}。',
            f'标识牌上的数字颜色是{color_to_chinese(result.number_color)}。',
            f'标识牌上的字母是{result.letter}。',
            f'标识牌上的字母颜色是{color_to_chinese(result.letter_color)}。',
        ]
        for speech in speeches:
            self.speak(speech)

        path_option = int(self.get_parameter('path_option').value)
        if path_option not in (1, 2):
            raise ValueError(
                f'fixed A-to-B path_option must be 1 or 2, got {path_option}'
            )
        self.get_logger().info(
            f'using fixed A-to-B path_option={path_option}; '
            f'recognized number={result.number} does not change the route'
        )
        return path_option

    def recognize_task5_qrcode(self):
        rtsp_urls = list(self.get_parameter('qrcode_camera_rtsp_urls').value)
        if not rtsp_urls:
            rtsp_urls = DEFAULT_RTSP_URLS

        attempts = max(
            int(self.get_parameter('qrcode_recognition_max_attempts').value),
            1,
        )
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = capture_and_recognize_qrcode(
                    output_dir=self._qrcode_output_dir(),
                    rtsp_urls=[str(url) for url in rtsp_urls],
                    warmup_frames=int(
                        self.get_parameter('qrcode_camera_warmup_frames').value
                    ),
                    open_timeout=float(
                        self.get_parameter('qrcode_camera_open_timeout_sec').value
                    ),
                    frame_count=int(
                        self.get_parameter('qrcode_vote_frame_count').value
                    ),
                    frame_interval_sec=float(
                        self.get_parameter('qrcode_vote_frame_interval_sec').value
                    ),
                    min_valid_votes=int(
                        self.get_parameter('qrcode_min_valid_votes').value
                    ),
                    detector=self._qrcode_detector,
                )
                self.get_logger().info(
                    f'QR code recognized on attempt {attempt}/{attempts}: '
                    f'content={result.content!r}, target={result.target}, '
                    f'votes={result.valid_frame_count}/{result.total_frame_count}, '
                    f'image={result.image_path}'
                )
                return result
            except Exception as exc:
                last_error = exc
                self.get_logger().warning(
                    f'QR code recognition attempt {attempt}/{attempts} failed: {exc}'
                )
        raise RuntimeError(
            f'QR code recognition failed after {attempts} attempts: {last_error}'
        )

    def run_task5(self) -> int:
        if not bool(self.get_parameter('task5_enabled').value):
            self.get_logger().info('task 5 is disabled')
            return GoalStatus.STATUS_SUCCEEDED

        self.wait_for_voice_command(
            'voice_command_find_target',
        )
        try:
            result = self.recognize_task5_qrcode()
        except Exception as exc:
            self.get_logger().error(f'task 5 QR code recognition failed: {exc}')
            return GoalStatus.STATUS_ABORTED

        target = result.target
        try:
            self.speak(f'二维码解析成功，准备到{target}点进行目标搜寻')
        except Exception as exc:
            self.get_logger().warning(
                f'failed to speak QR code recognition result: {exc}'
            )
        target_x = float(self.get_parameter(f'{target.lower()}_x').value)
        target_y = float(self.get_parameter(f'{target.lower()}_y').value)
        target_yaw = float(self.get_parameter(f'{target.lower()}_yaw').value)
        self.apply_avoidance_profile('c_departure')
        status = self.send_goal(
            f'task5_{target}',
            target_x,
            target_y,
            target_yaw,
            near_target_profile='endpoint',
            profile_switch_distance=float(
                self.get_parameter('endpoint_profile_switch_distance').value
            ),
            departure_profile='endpoint',
            departure_origin=(
                float(self.get_parameter('c_x').value),
                float(self.get_parameter('c_y').value),
            ),
            departure_switch_distance=float(
                self.get_parameter('c_departure_profile_distance').value
            ),
        )
        if status != GoalStatus.STATUS_SUCCEEDED:
            return status
        self.wait_until_pose_reached(f'task5_{target}', target_x, target_y)
        self.stop_robot()
        try:
            self.speak(f'已到达{target.lower()}点')
        except Exception as exc:
            self.get_logger().warning(
                f'failed to speak task 5 arrival message: {exc}'
            )
        return GoalStatus.STATUS_SUCCEEDED

    def recognize_task6_traffic_light(self):
        rtsp_urls = list(self.get_parameter('traffic_light_camera_rtsp_urls').value)
        if not rtsp_urls:
            rtsp_urls = DEFAULT_RTSP_URLS

        attempts = max(
            int(self.get_parameter('traffic_light_recognition_max_attempts').value),
            1,
        )
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = capture_and_recognize_traffic_light(
                    output_dir=self._traffic_light_output_dir(),
                    rtsp_urls=[str(url) for url in rtsp_urls],
                    warmup_frames=int(
                        self.get_parameter('traffic_light_camera_warmup_frames').value
                    ),
                    open_timeout=float(
                        self.get_parameter('traffic_light_camera_open_timeout_sec').value
                    ),
                    frame_count=int(
                        self.get_parameter('traffic_light_vote_frame_count').value
                    ),
                    frame_interval_sec=float(
                        self.get_parameter('traffic_light_vote_frame_interval_sec').value
                    ),
                    min_valid_votes=int(
                        self.get_parameter('traffic_light_min_valid_votes').value
                    ),
                    min_confidence=float(
                        self.get_parameter('traffic_light_min_confidence').value
                    ),
                    saturation_threshold=int(
                        self.get_parameter('traffic_light_saturation_threshold').value
                    ),
                    value_threshold=int(
                        self.get_parameter('traffic_light_value_threshold').value
                    ),
                    min_component_area=int(
                        self.get_parameter('traffic_light_min_component_area').value
                    ),
                )
                self.get_logger().info(
                    f'traffic light recognized on attempt {attempt}/{attempts}: '
                    f'state={result.state}, confidence={result.confidence:.2f}, '
                    f'votes={result.valid_frame_count}/{result.total_frame_count}, '
                    f'image={result.image_path}'
                )
                return result
            except Exception as exc:
                last_error = exc
                self.get_logger().warning(
                    f'traffic light recognition attempt {attempt}/{attempts} failed: {exc}'
                )
        raise RuntimeError(
            f'traffic light recognition failed after {attempts} attempts: {last_error}'
        )

    def run_task6(self) -> int:
        if not bool(self.get_parameter('task6_enabled').value):
            self.get_logger().info('task 6 is disabled')
            return GoalStatus.STATUS_SUCCEEDED

        self.stop_robot()
        self.wait_for_voice_command(
            'voice_command_can_pass',
        )
        try:
            result = self.recognize_task6_traffic_light()
        except Exception as exc:
            self.get_logger().error(f'task 6 traffic light recognition failed: {exc}')
            return GoalStatus.STATUS_ABORTED

        speech = '可以通行' if result.state == 'green' else '不可通行'
        try:
            self.speak(speech)
        except Exception as exc:
            self.get_logger().warning(f'failed to speak task 6 result: {exc}')
        return GoalStatus.STATUS_SUCCEEDED

    def run_mission(self) -> int:
        a_x = float(self.get_parameter('a_x').value)
        a_y = float(self.get_parameter('a_y').value)
        a_yaw = float(self.get_parameter('a_yaw').value)
        wait_at_a_sec = float(self.get_parameter('wait_at_a_sec').value)
        b_x = float(self.get_parameter('b_x').value)
        b_y = float(self.get_parameter('b_y').value)
        b_yaw = float(self.get_parameter('b_yaw').value)
        wait_at_b_sec = float(self.get_parameter('wait_at_b_sec').value)
        c_x = float(self.get_parameter('c_x').value)
        c_y = float(self.get_parameter('c_y').value)
        c_yaw = float(self.get_parameter('c_yaw').value)

        self.wait_for_voice_command('voice_command_a')
        self.apply_avoidance_profile('endpoint')
        a_status = self.send_goal('A', a_x, a_y, a_yaw)
        if a_status != GoalStatus.STATUS_SUCCEEDED:
            return a_status
        self.wait_until_pose_reached('A', a_x, a_y)
        self.apply_avoidance_profile('normal')

        try:
            self.speak(str(self.get_parameter('a_arrival_speech').value))
        except Exception as exc:
            self.get_logger().warning(f'failed to speak arrival message at A: {exc}')

        if bool(self.get_parameter('sign_recognition_enabled').value):
            self.wait_for_voice_command('voice_command_start_recognition')

        try:
            path_option = self.recognize_sign_and_choose_path()
        except Exception as exc:
            self.get_logger().warning(
                f'failed to recognize sign at A, keeping existing path_option: {exc}'
            )
            path_option = int(self.get_parameter('path_option').value)

        self.get_logger().info(
            f'arrived at A, waiting for {wait_at_a_sec:.1f} seconds before next task'
        )
        time.sleep(wait_at_a_sec)
        self.wait_for_voice_command('voice_command_b')

        if path_option == 1:
            mid_x = float(self.get_parameter('path1_mid_x').value)
            mid_y = float(self.get_parameter('path1_mid_y').value)
            mid_yaw = float(self.get_parameter('path1_mid_yaw').value)
            mid_name = 'path1_mid'
        elif path_option == 2:
            mid_x = float(self.get_parameter('path2_mid_x').value)
            mid_y = float(self.get_parameter('path2_mid_y').value)
            mid_yaw = float(self.get_parameter('path2_mid_yaw').value)
            mid_name = 'path2_mid'
        else:
            self.get_logger().error(f'unsupported path_option={path_option}, expected 1 or 2')
            return GoalStatus.STATUS_ABORTED

        self.apply_avoidance_profile('normal')
        self._set_moving_obstacle_monitor(
            bool(self.get_parameter('moving_obstacle_enable_on_a_to_b').value),
            speech_parameter='mid_obstacle_speech',
            segment='A->B',
            max_speech_count=int(
                self.get_parameter('a_to_b_obstacle_speech_max_count').value
            ),
        )
        mid_status = self.send_goal(mid_name, mid_x, mid_y, mid_yaw)
        if mid_status != GoalStatus.STATUS_SUCCEEDED:
            self._set_moving_obstacle_monitor(False)
            return mid_status
        self.wait_until_pose_reached(mid_name, mid_x, mid_y)
        self.get_logger().info(
            f'confirmed arrival at A->B route midpoint [{mid_name}]; continuing to B'
        )

        status = self.send_goal(
            'B',
            b_x,
            b_y,
            b_yaw,
            near_target_profile='endpoint',
            profile_switch_distance=float(
                self.get_parameter('endpoint_profile_switch_distance').value
            ),
        )
        self._set_moving_obstacle_monitor(False)
        if status != GoalStatus.STATUS_SUCCEEDED:
            return status
        self.wait_until_pose_reached('B', b_x, b_y)
        try:
            self.speak(str(self.get_parameter('b_arrival_speech').value))
        except Exception as exc:
            self.get_logger().warning(f'failed to speak arrival message at B: {exc}')
        self.get_logger().info(
            f'arrived at B, waiting for {wait_at_b_sec:.1f} seconds before next task'
        )
        time.sleep(wait_at_b_sec)
        self.wait_for_voice_command('voice_command_c')

        # Keep the endpoint profile while leaving B so the large B->C inflation
        # radius cannot block the route before the robot clears the endpoint.
        self.apply_avoidance_profile('endpoint')
        self._set_moving_obstacle_monitor(
            bool(self.get_parameter('moving_obstacle_enable_on_b_to_c').value),
            speech_parameter='moving_obstacle_speech',
            segment='B->C',
            max_speech_count=int(
                self.get_parameter('b_to_c_obstacle_speech_max_count').value
            ),
        )
        status = self.send_goal(
            'C',
            c_x,
            c_y,
            c_yaw,
            near_target_profile='endpoint',
            profile_switch_distance=float(
                self.get_parameter('endpoint_profile_switch_distance').value
            ),
            departure_profile='b_to_c',
            departure_origin=(b_x, b_y),
            departure_switch_distance=float(
                self.get_parameter('b_departure_profile_distance').value
            ),
        )
        self._set_moving_obstacle_monitor(False)
        if status != GoalStatus.STATUS_SUCCEEDED:
            return status
        self.wait_until_pose_reached('C', c_x, c_y)
        self.stop_robot()
        try:
            self.speak(str(self.get_parameter('c_arrival_speech').value))
        except Exception as exc:
            self.get_logger().warning(
                f'failed to speak arrival message at C: {exc}'
            )

        task5_status = self.run_task5()
        if task5_status != GoalStatus.STATUS_SUCCEEDED:
            return task5_status
        return self.run_task6()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoNavNode()
    exit_code = 0
    try:
        node.publish_initial_pose()
        if not node.wait_for_amcl_pose():
            exit_code = 1
            return
        if not node.wait_for_navigation_server():
            exit_code = 1
            return
        status = node.run_mission()
        if status != GoalStatus.STATUS_SUCCEEDED:
            exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(exit_code)
