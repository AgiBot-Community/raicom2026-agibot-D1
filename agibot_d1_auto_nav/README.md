# agibot_d1_auto_nav

> 本文件介绍包内部细节。作为独立开源仓库构建和运行时，请先阅读仓库根目录的
> [`README.md`](../README.md)。

`agibot_d1_auto_nav` 是比赛任务接入包。  
它在现有 Nav2、AMCL、地图服务之上，补了一层任务编排逻辑，负责：

1. 启动 `map_server`
2. 启动 `amcl`
3. 启动 `competition` 下单独维护的一套 Nav2 参数
4. 发布初始位姿
5. 按比赛流程执行导航、语音播报、视觉识别、语音指令等待

当前任务流程是：

1. 启动后等待语音 `请引领到目标点A`
2. 前往 A 点
3. 到达 A 点后播报 `已到达a点`
4. 拍照识别 A4 纸上的数字、数字颜色、字母、字母颜色
5. 播报识别结果
6. 按 `path_option` 使用固定 A-B 路线
7. 等待语音 `请引领到目标点B`
8. 通过 A-B 路径引导点连续前往 B，中间不停车；检测到动态障碍时播报 `前方发现障碍物，正在绕行，请小心行走`
9. 到达 B 点
10. 等待语音 `请引领到目标点C`
11. 从 B 直接前往 C；检测到动态障碍时播报 `注意前方有移动物体，正在避让`
12. 到达 C 点并停车
13. 完成任务 5 的二维码目标导航
14. 停车并等待语音 `是否可以通行`
15. 多帧识别前方红绿灯；绿灯播报 `可以通行`，红灯播报 `不可通行`

## 目录结构

- `agibot_d1_auto_nav/auto_nav_node.py`
  任务主节点，负责任务状态机、导航下发、语音与视觉流程。
- `agibot_d1_auto_nav/vision_utils.py`
  相机取流与兼容接口。
- `agibot_d1_auto_nav/sign_recognizer.py`
  标识牌候选配对、白纸定位、透视校正和多帧投票。
- `agibot_d1_auto_nav/character_classifier.py`
  HOG + PCA + KNN 轻量字符分类器。
- `agibot_d1_auto_nav/color_classifier.py`
  字符、桌椅 ROI 和红绿灯 ROI 的颜色属性判断。
- `agibot_d1_auto_nav/speech_command_utils.py`
  本地 faster-whisper / Vosk 语音识别及指令匹配，等待 A/B/C 指令。
- `agibot_d1_auto_nav/traffic_light_utils.py`
  高亮红绿灯候选检测和多帧投票。
- `agibot_d1_auto_nav/speaker_utils.py`
  本地 TTS 合成与扬声器播放。
- `launch/auto_nav_bringup.launch.py`
  总启动入口。
- `launch/nav2_competition.launch.py`
  比赛版 Nav2 启动文件。
- `config/auto_nav_params.yaml`
  任务参数、语音参数、视觉参数。
- `config/waypoints.yaml`
  初始位姿、A/B/C 点和 A-B 路径引导点的位置与朝向。所有角度均为弧度。
- `config/nav2_*_competition.yaml`
  Nav2 的控制、规划、行为树、速度平滑参数。

## 依赖总览

这个包要能独立编译运行，依赖分为 4 类：

1. ROS 2 / Nav2 / 业务包依赖
2. Python 依赖
3. 系统命令与音频依赖
4. 运行时外部条件

### 1. ROS 2 / Nav2 / 业务包依赖

`package.xml` 中直接声明了这些依赖：

- `rclpy`
- `geometry_msgs`
- `sensor_msgs`
- `nav2_msgs`
- `action_msgs`
- `ament_index_python`
- `launch`
- `launch_ros`
- `nav2_amcl`
- `nav2_map_server`
- `nav2_lifecycle_manager`
- `agibot_d1_navigation2`
- `agibot_d1_localization`
- `agibot_d1_map_server`

此外，启动文件里还实际拉起了这些 Nav2 组件，所以运行环境还必须存在：

- `nav2_planner`
- `nav2_controller`
- `nav2_behaviors`
- `nav2_waypoint_follower`
- `nav2_velocity_smoother`
- `nav2_smoother`
- `nav2_bt_navigator`

如果你的基础环境已经安装好了 Nav2，一般这些包都会一起存在。

### 2. Python 依赖

代码里实际使用到的 Python 三方库有：

- `faster-whisper`
- `vosk`（切换到 Vosk 后端时需要）
- `pyaudio`
- `cv2`（OpenCV）
- `numpy`

推荐安装方式：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy alsa-utils
python3 -m pip install -U faster-whisper vosk numpy
```

### 3. 系统命令与音频依赖

语音播报不是直接走 ROS 音频节点，而是通过本地命令完成：

- `espeak-ng`：负责 TTS 合成
- `paplay` 或 `aplay`：负责播放音频

建议安装：

```bash
sudo apt install -y espeak-ng pulseaudio-utils alsa-utils
```

说明：

- `paplay` 来自 `pulseaudio-utils`
- `aplay` 来自 `alsa-utils`
- 代码会优先尝试 `paplay`，找不到时退回到 `aplay`

### 4. 运行时外部条件

除了软件依赖，还需要这些运行条件：

- 有可用地图 yaml 文件
- 有 AMCL 参数文件
- 机器人底层已经能正常发布 `/odom`、`/scan`
- Nav2 所需 TF 正常
- 已下载本地 faster-whisper 模型（或配置可选 Vosk 模型）
- 相机 RTSP 流可访问
- 本地麦克风与扬声器设备可用

## 已知默认接口

当前代码默认这些接口存在：

- 初始位姿发布：`/initialpose`
- 定位输出：`/amcl_pose`
- 激光雷达：`/scan`
- 导航 action：`navigate_to_pose`
- 地图坐标系：`map`
- 导航速度链路：
  `controller_server` 输出 `cmd_vel_nav`
  `velocity_smoother` 输出 `cmd_vel`

如果你的机器人实际接口名不同，需要同步改 launch 或参数。

## 相机与视觉识别依赖

视觉识别默认通过 RTSP 拉取机器狗相机画面，代码内默认地址为：

- `rtsp://192.168.168.168:8554/test`
- `rtsp://192.168.234.1:8554/test`

图像使用完全离线、CPU 可运行的本地方案：

- 全画面彩色字符候选提取，不再依赖整张图片中心线
- 根据高度、对齐、间距和白色背景支持度选择“数字 + 字母”候选对
- 白纸定位和可选透视校正；定位不可靠时自动退回原始字符区域
- HOG + PCA + KNN 轻量字符分类器，模型文件约 6 MB
- 仅在字符真实掩码内部判断 HSV 颜色
- 数字、数字颜色、字母、字母颜色分别进行加权多帧投票
- 低置信度或投票不一致时拒绝输出，并保留原有路径作为安全回退

现有回放验证结果见 `VISION_VALIDATION.csv` 和
`validation/vision_replay_summary.csv`。

识别结果图片默认保存在：

- `~/.cache/agibot_d1_auto_nav/captures`

注意：

- 如果 RTSP 地址不通，A 点识别任务会失败
- OpenCV 需要具备视频解码能力，通常系统自带即可

## 语音识别依赖

语音指令等待支持本地 faster-whisper 和原始 Vosk 两种后端，当前默认使用
`faster_whisper`。Vosk 原始实现备份在
`agibot_d1_auto_nav/speech_backends/vosk_original/`。正式任务使用 `command`
匹配模式：只有识别文本与当前提示词语义匹配时才会继续执行；环境声音、任意非空文本、
其他目标点指令以及缺少目标点编号的模糊指令都不会触发导航。系统允许“引领/带我去/前往”
等近义表达，以及少量常见同音识别误差。提示词包括：

- `请引领到目标点A`
- `请引领到目标点B`
- `请引领到目标点C`
- `请寻找指定目标`
- `是否可以通行`

到达 C 点后进入任务 5：机器人停车并播报到达信息，收到上述语音后对前方二维码进行多帧、多预处理投票识别。二维码内容支持 `A`、`QR-A-01`、`TARGET: B`、`目标点C` 等形式；解析后机器人导航到对应点，以该点配置的朝向停车并播报到达信息。

任务 5 完成后进入任务 6：机器人保持停车，检测到语音后采集前方多帧图像，只对高亮、高饱和度的红色和绿色灯面进行投票。识别为绿灯时播报 `可以通行`，识别为红灯时播报 `不可通行`。

导航过程中会动态切换避障参数：靠近 A 和任务 5 目标时使用终点配置；B 到 C 全程扩大局部膨胀、提高障碍权重并降低速度。A 到 B、B 到 C 的障碍播报均由激光动态障碍数据触发，不再按到达固定点播报。

A 到 B 固定使用 `path_option` 指定的路线，默认 `path_option: 1`。A 点标识牌仍会识别和播报，但识别到的数字不再切换路线。

当前参数默认使用：

- `voice_command_backend: faster_whisper`，可改为 `vosk`
- `voice_command_faster_whisper_model_size: base`，可改为 `small`
- faster-whisper 权重默认从比赛包的 `models/speech` 目录读取
- 可用 `AGIBOT_WHISPER_BASE_MODEL_PATH`、`AGIBOT_WHISPER_SMALL_MODEL_PATH`
  或 `AGIBOT_VOSK_MODEL_PATH` 环境变量覆盖模型目录
- `voice_command_device_id: 0`
- `voice_command_sample_rate: 44100`

如果你机器上的录音设备名不同，可以先列设备：

```bash
arecord -l
```

再把 `config/auto_nav_params.yaml` 里的：

- `voice_command_device_id`
- `voice_command_backend`
- `voice_command_faster_whisper_model_size`
- 对应模型路径环境变量（仅在不使用默认目录时设置）
- `voice_command_sample_rate`
- `voice_command_channels`

改成你的实际设备参数。

## 编译前准备

下面假设 D1 基础工作区已经编译，并且你已经进入本比赛仓库根目录：

```bash
cd <competition-repository>
```

比赛包应位于：

```bash
agibot_d1_auto_nav
```

先确认以下目录或包存在：

1. `agibot_d1_auto_nav`
2. `agibot_d1_navigation2`
3. `agibot_d1_localization`
4. `agibot_d1_map_server`
5. 一个可用的外部地图 yaml 文件

## 编译步骤

### 独立比赛仓库（推荐）

```bash
cd <competition-repository>
source /absolute/path/to/d1_base_ws/install/setup.bash
colcon build --base-paths . --packages-select agibot_d1_auto_nav
source install/setup.bash
```

### 比赛目录仍放在完整工作区内

```bash
cd <workspace>
colcon build --base-paths competition --packages-select agibot_d1_auto_nav
source install/setup.bash
```

如果你修改了 Python 代码，建议每次重新 `source install/setup.bash`。

## 启动方式

最常用启动命令：

```bash
ros2 launch agibot_d1_auto_nav auto_nav_bringup.launch.py \
  map_yaml:=/absolute/path/to/map.yaml
```

所有导航点位统一在 `config/waypoints.yaml` 中修改。需要使用另一份点位文件时：

```bash
ros2 launch agibot_d1_auto_nav auto_nav_bringup.launch.py \
  map_yaml:=/absolute/path/to/map.yaml \
  waypoints_params_file:=/absolute/path/to/waypoints.yaml
```

## 重要参数文件

### `config/auto_nav_params.yaml`

这里控制：

- A/B/C 语音命令
- 麦克风设备号
- 视觉模型参数
- 识别图片保存目录
- 到点容差
- 语音播报文案

### `config/waypoints.yaml`

这里是点位数据的唯一配置来源，包含：

- 初始定位点
- A、B、C 三个目标点
- 路径 1、路径 2 的 A-B 引导点

每个点都包含 `x`、`y` 和 `yaw`。修改源码目录中的配置后需重新构建并 `source install/setup.bash`，或通过 `waypoints_params_file` 指定修改后的文件。

### `launch/auto_nav_bringup.launch.py`

这里控制：

- 地图路径
- AMCL 参数文件路径
- 自动导航参数文件路径
- 点位参数文件路径

### `launch/nav2_competition.launch.py`

这里控制比赛版 Nav2 启动，包括：

- `planner_server`
- `controller_server`
- `behavior_server`
- `waypoint_follower`
- `velocity_smoother`
- `smoother_server`
- `bt_navigator`

## 运行前检查清单

正式运行前建议依次确认：

1. `ros2 pkg list | grep agibot_d1_auto_nav` 能看到包
2. `ros2 pkg list | grep nav2_` 能看到 Nav2 相关包
3. 地图 yaml 路径存在
4. `/scan`、`/odom`、`/tf`、`/amcl_pose` 正常
5. RTSP 视频流能打开
6. 麦克风设备号正确
7. 扬声器能正常播放
8. 当前选择的 faster-whisper 或 Vosk 模型目录存在

## 常见问题

### 1. 编译成功但启动时报找不到包

通常是没重新 source：

```bash
source <workspace>/install/setup.bash
```

### 2. 语音识别无反应

重点检查：

- `voice_command_device` 是否正确
- `voice_command_faster_whisper_model_size` 选择的 base/small 模型是否已下载，
  或相应模型路径环境变量是否正确
- `voice_command_sample_rate` 是否和设备匹配
- 运行日志里是否持续出现 `voice recognition heard: ...`

### 3. 到 A 点后视觉识别失败

重点检查：

- RTSP 流地址是否可用
- RTSP 相机网络是否可达
- `~/.cache/agibot_d1_auto_nav/captures` 是否成功生成图片

### 4. 扬声器不发声

重点检查：

- `espeak-ng` 是否安装
- `paplay` 或 `aplay` 是否安装
- 当前系统默认输出设备是否正确
- 必要时设置 `speech_player` 或 `speech_device`

## 最小可运行环境总结

如果你希望别人拿到 `competition/agibot_d1_auto_nav` 后能独立跑起来，至少要满足：

1. 有 ROS 2 + Nav2 环境
2. 有 `agibot_d1_navigation2`、`agibot_d1_localization`、`agibot_d1_map_server`
3. 安装 `python3-opencv`、`faster-whisper`、`vosk`、`numpy`、`alsa-utils`
4. 安装 `espeak-ng`、`pulseaudio-utils`、`alsa-utils`
5. 有地图 yaml
6. 有可用 RTSP 相机
7. 有可用麦克风和扬声器
8. 已准备本地语音模型，且 RTSP 相机网络可达

满足以上条件后，按“编译步骤 + 启动方式”即可独立编译运行。
