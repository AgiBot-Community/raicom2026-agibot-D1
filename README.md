# Agibot D1 睿抗比赛任务包

这是本次比赛实际使用的 ROS 2 overlay 仓库。它不包含机器狗厂商底层 SDK，而是在已有
Agibot D1 基础工作区之上提供比赛任务编排、Nav2 参数、中文语音指令、标识牌识别、
二维码任务和红绿灯识别。

语音导航已经采用安全匹配逻辑：只有识别内容与当前阶段指令匹配时才会执行动作。
例如等待 A 点指令时，可以识别“请引领到目标点A”“带我去目标点A”“前往目标点A”
以及少量同音误识别，但普通声音、闲聊、B/C 点指令、缺少目标点编号或带否定词的句子
均不会触发机器人前进。

> 安全提示：本项目会控制真实机器狗。首次运行请清空测试区域、准备急停、降低速度，
> 并先验证地图、TF、点位、避障和语音拒识逻辑。

## 环境要求

推荐环境：

- Ubuntu 22.04，x86_64 或 aarch64
- ROS 2 Humble、Nav2、Python 3.10
- Agibot D1 基础工作区和厂商驱动
- 激光雷达、麦克风、扬声器、可访问的 RTSP 相机

基础工作区必须提供以下 ROS 包：

```bash
ros2 pkg prefix agibot_d1_navigation2
ros2 pkg prefix agibot_d1_localization
ros2 pkg prefix agibot_d1_map_server
```

同时应已有 `/scan`、`/odom`、`/tf` 和机器人速度控制链路。如果上述命令失败，请先按
机器狗厂商文档安装并编译 D1 基础工作区，再继续本 README。

## 目录结构

```text
.
├── agibot_d1_auto_nav/
│   ├── agibot_d1_auto_nav/   # 任务节点、语音、视觉和工具模块
│   ├── config/                # 比赛、Nav2 和点位参数
│   ├── launch/                # 总启动与比赛版 Nav2 启动文件
│   ├── test/                  # 单元测试和配置检查
│   ├── tools/                 # 麦克风、模型和视觉回放工具
│   ├── package.xml
│   └── requirements.txt
├── scripts/                   # 大模型下载工具
├── .env.example               # 可选的本地路径配置模板
├── OPEN_SOURCE_AUDIT.md       # 不应提交内容及发布检查
└── LICENSE                    # MIT License
```

包内更细的状态机、参数和故障排查说明见
[agibot_d1_auto_nav/README.md](agibot_d1_auto_nav/README.md)。

## 1. 安装系统和 Python 依赖

先打开已经能使用 D1 底层包的终端，然后执行：

```bash
source /opt/ros/humble/setup.bash
source /absolute/path/to/d1_base_ws/install/setup.bash

sudo apt update
sudo apt install -y \
  python3-pip python3-opencv python3-numpy \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  alsa-utils espeak-ng pulseaudio-utils

python3 -m pip install -r agibot_d1_auto_nav/requirements.txt
```

也可以用 rosdep 补齐声明的 ROS 依赖：

```bash
rosdep install --from-paths agibot_d1_auto_nav \
  --ignore-src -r -y --rosdistro humble
```

## 2. 下载离线语音模型

语音权重约数百 MiB，不放入 Git。默认配置使用 `base` 模型：

```bash
python3 scripts/download_speech_model.py --model base
```

需要 `small` 时使用 `--model small`，并把参数
`voice_command_faster_whisper_model_size` 改为 `small`。建议在编译前下载模型，确保普通
colcon 安装也会包含权重。模型脚本使用 Hugging Face 官方支持的 `snapshot_download`
接口，模型目录已被 `.gitignore` 排除。

## 3. 创建本地配置

不要直接改动并提交公共示例，先复制本地配置：

```bash
cp .env.example .env
cp agibot_d1_auto_nav/config/auto_nav_params.yaml \
  agibot_d1_auto_nav/config/auto_nav_params.local.yaml
cp agibot_d1_auto_nav/config/waypoints.yaml \
  agibot_d1_auto_nav/config/waypoints.local.yaml
```

需要环境变量路径覆盖时，编辑 `.env` 并加载：

```bash
set -a
source .env
set +a
```

在两个本地 YAML 中至少检查：

- `voice_command_device_id`、采样率、声道数和语音模型规格；
- 三组 `*_camera_rtsp_urls`；
- 初始点、A/B/C 点和中间引导点坐标；
- 任务开关、到点容差、路径选择和速度/避障参数。

`.env` 与 `*.local.yaml` 都已加入 `.gitignore`。

## 4. 编译 overlay

在本仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
source /absolute/path/to/d1_base_ws/install/setup.bash

colcon build --symlink-install --base-paths . \
  --packages-select agibot_d1_auto_nav
source install/setup.bash
```

## 5. 测试

先运行不需要机器人运动的测试：

```bash
python3 -m pytest agibot_d1_auto_nav/test -q

python3 agibot_d1_auto_nav/tools/test_competition_microphone.py \
  --config agibot_d1_auto_nav/config/auto_nav_params.local.yaml \
  --expected "请引领到目标点A" --timeout 20
```

麦克风测试只有在识别到匹配指令后才显示 `accepted`。说普通聊天、错误点位和否定指令
应继续等待，这是正常行为。

## 6. 启动比赛流程

先在另一个终端按厂商基础工作区说明启动 D1 驱动、雷达、里程计/定位输入，确认
`/scan`、`/odom` 和 TF 正常。然后在本仓库根目录运行：

```bash
source /opt/ros/humble/setup.bash
source /absolute/path/to/d1_base_ws/install/setup.bash
source install/setup.bash

ros2 launch agibot_d1_auto_nav auto_nav_bringup.launch.py \
  map_yaml:=/absolute/path/to/map.yaml \
  auto_nav_params_file:="$(pwd)/agibot_d1_auto_nav/config/auto_nav_params.local.yaml" \
  waypoints_params_file:="$(pwd)/agibot_d1_auto_nav/config/waypoints.local.yaml"
```

地图 YAML 必须使用绝对路径，并且其引用的地图图片存在。也可在 `.env` 中设置
`AGIBOT_MAP_YAML` 后省略 `map_yaml` 参数。

## 常见问题

- 找不到基础包：没有 source D1 基础工作区，或厂商包尚未编译。
- 找不到比赛包：重新执行 `source install/setup.bash`。
- 找不到语音模型：重新运行模型下载脚本，确认下载发生在编译前。
- 麦克风无输入：通过 `arecord -l` 查设备号并更新本地 YAML。
- 相机超时：检查 RTSP 地址、机器人网络和 OpenCV 视频解码能力。
- 导航不动：检查 Nav2 action、地图、AMCL、TF、`/scan` 和 `/odom`，不要绕过急停。
- 普通说话不触发：这是预期的安全行为；查看日志中的 `voice recognition heard`。

## 发布内容与许可证

真正需要提交的是源码、YAML、launch、测试、工具、约 17 MiB 的视觉分类模型和本目录的
开源文档。不要提交 speech 权重、`tmp/`、`build/`、`install/`、`log/` 或缓存。
完整清单见 [OPEN_SOURCE_AUDIT.md](OPEN_SOURCE_AUDIT.md)。

本比赛代码采用 [MIT License](LICENSE)。外部依赖和下载模型的许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

参考资料：[ROS 2 Humble](https://docs.ros.org/en/humble/)、
[rosdep](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html)、
[Hugging Face 下载接口](https://huggingface.co/docs/huggingface_hub/guides/download)、
[Vosk 模型列表](https://alphacephei.com/vosk/models)。

