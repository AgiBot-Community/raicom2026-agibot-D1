import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    auto_nav_pkg = get_package_share_directory('agibot_d1_auto_nav')
    localization_pkg = get_package_share_directory('agibot_d1_localization')
    default_map_yaml = os.environ.get('AGIBOT_MAP_YAML', '')
    default_amcl_params = os.path.join(localization_pkg, 'params', 'amcl.yaml')
    default_auto_nav_params = os.path.join(auto_nav_pkg, 'config', 'auto_nav_params.yaml')
    default_waypoints_params = os.path.join(auto_nav_pkg, 'config', 'waypoints.yaml')

    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml',
        default_value=default_map_yaml,
        description=(
            'Absolute path to the map yaml used by nav2_map_server. '
            'Pass map_yaml:=... or set AGIBOT_MAP_YAML.'
        ),
    )
    amcl_params_arg = DeclareLaunchArgument(
        'amcl_params_file',
        default_value=default_amcl_params,
        description='Absolute path to AMCL params yaml file.',
    )
    auto_nav_params_arg = DeclareLaunchArgument(
        'auto_nav_params_file',
        default_value=default_auto_nav_params,
        description='Auto navigation node parameters.',
    )
    waypoints_params_arg = DeclareLaunchArgument(
        'waypoints_params_file',
        default_value=default_waypoints_params,
        description='Initial pose and mission waypoint parameters.',
    )
    bringup_delay_arg = DeclareLaunchArgument(
        'bringup_delay_sec',
        default_value='8.0',
        description='Delay before the auto navigation node starts.',
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'yaml_filename': LaunchConfiguration('map_yaml'),
            'frame_id': 'map',
        }],
    )

    map_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_mapper',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[LaunchConfiguration('amcl_params_file')],
    )

    localization_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['amcl'],
        }],
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(auto_nav_pkg, 'launch', 'nav2_competition.launch.py')
        )
    )

    auto_nav_node = Node(
        package='agibot_d1_auto_nav',
        executable='auto_nav_node',
        name='agibot_d1_auto_nav',
        output='screen',
        parameters=[
            LaunchConfiguration('auto_nav_params_file'),
            LaunchConfiguration('waypoints_params_file'),
        ],
    )

    delayed_auto_nav = TimerAction(
        period=LaunchConfiguration('bringup_delay_sec'),
        actions=[auto_nav_node],
    )

    return LaunchDescription([
        map_yaml_arg,
        amcl_params_arg,
        auto_nav_params_arg,
        waypoints_params_arg,
        bringup_delay_arg,
        map_server_node,
        amcl_node,
        nav2_launch,
        TimerAction(period=2.0, actions=[map_lifecycle_manager]),
        TimerAction(period=3.0, actions=[localization_lifecycle_manager]),
        delayed_auto_nav,
    ])
