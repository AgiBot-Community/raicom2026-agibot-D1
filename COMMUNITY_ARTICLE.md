# Agibot D1 比赛自主导航实践：语音、视觉与 Nav2 多模态任务编排

> 推荐标签：开源项目  
> 项目地址：https://github.com/bob142857/raicom2026-agibot-d1

## 前言

在机器人比赛中，让机器狗“能动起来”通常只是第一步。真正困难的是把导航、语音、视觉和安全策略组织成一条稳定、可复现的任务链，同时还要处理真实场地中的环境噪声、相机抖动、动态障碍物和设备差异。

我们将睿抗机器人大赛中使用的 Agibot D1 比赛任务代码整理并开源。项目基于 ROS 2 Humble 和 Nav2，以 overlay 工作区的形式接入机器狗基础驱动，实现了 A/B/C 点语音引导、标识牌识别、二维码目标导航、红绿灯判断和动态避障等功能。

语音控制采用固定指令触发：系统先完成离线语音识别，再确认内容与当前阶段要求的控制用语一致，只有匹配成功才会进入对应的导航或识别状态。为了兼顾比赛现场的可用性，匹配模块可以容忍少量同音字和近义表达，同时会拒绝错误目标点、缺少动作意图或包含否定词的语句。

本文介绍项目的整体设计、语音指令识别方案、部署方法和开源整理经验，希望能为具身智能、移动机器人和 ROS 2 项目提供一份可复用的工程参考。

## 一、项目实现了什么

项目不是完整的机器狗底层 SDK，而是运行在 Agibot D1 基础工作区之上的比赛任务包。基础工作区负责驱动、雷达、里程计、TF 和速度控制，本项目负责上层任务编排。

完整任务流程如下：

1. 等待“请引领到目标点 A”等有效语音指令；
2. 使用 Nav2 导航到 A 点并进行到点播报；
3. 通过 RTSP 相机识别标识牌上的数字、字母及颜色；
4. 等待前往 B 点的指令，沿配置的引导点连续导航；
5. 在 A→B 和 B→C 路段根据激光数据检测动态障碍并播报；
6. 到达 C 点后，等待指令并识别二维码中的目标点；
7. 导航到二维码指定的 A、B 或 C 点；
8. 等待“是否可以通行”，通过多帧图像判断红绿灯并播报结果。

整体数据链路可以概括为：

```text
麦克风 ─→ 离线 ASR ─→ 指令语义门控 ─┐
                                      │
RTSP 相机 ─→ 标识牌/二维码/红绿灯识别 ─┼─→ 任务状态机 ─→ Nav2 Action
                                      │
激光雷达 ─→ 动态障碍判断 ──────────────┘              │
                                                     ├─→ cmd_vel
                                                     └─→ 本地语音播报
```

任务入口、算法模块与参数彼此分离：

- `auto_nav_node.py`：任务状态机和导航调度；
- `speech_command_utils.py`：录音、离线识别和指令匹配；
- `sign_recognizer.py`：标识牌定位及多帧投票；
- `character_classifier.py`：HOG + PCA + KNN 字符分类；
- `qrcode_utils.py`：二维码解析与目标点投票；
- `traffic_light_utils.py`：红绿灯候选检测和状态投票；
- `config/`：任务、Nav2 和点位参数；
- `launch/`：总启动入口和比赛版 Nav2 启动文件。

这种拆分让硬件配置、比赛点位和识别阈值能够通过 YAML 调整，而不必反复修改任务源码。

## 二、固定语音指令的识别与语义约束

比赛现场存在观众交谈、扬声器回声、脚步声和设备噪声，中文 ASR 也可能产生同音字误识别。项目因此采用“语音活动检测 + 离线文本识别 + 当前任务语义匹配”的处理链路。每个任务阶段只监听相应的固定控制意图，例如等待 A 点指令时，仅接受目标为 A 且具有导航动作含义的语句。

### 1. 声音活动检测

节点启动时先采集一段环境音，缓存固定噪声基线。后续录音根据 RMS、连续有效块、前置缓存、尾部静音和最长语句时长切分完整语句。

系统只对整句话应用有上限的数字增益，避免逐块强行拉满音量，把底噪一起放大成“有效语音”。立体声输入也会先正确混合为单声道，再交给识别器。

### 2. 本地离线语音识别

项目默认使用 faster-whisper，也保留 Vosk 后端。识别在本机 CPU 上完成，不依赖云端模型 Key，既减少网络不稳定因素，也降低了敏感信息泄露风险。

语音模型权重体积较大，因此没有直接提交到 Git 仓库。仓库提供下载脚本，并允许通过环境变量覆盖模型路径。

### 3. 面向任务的语义匹配

语音识别有时会把字母 A 识别为“诶”，把“目标点”识别为“目标店”。如果只做字符串全等，现场可用性会很差；如果只做宽松相似度，又容易接受错误命令。

项目采用受约束的模糊匹配：

- 接受“请引领到”“带我去”“前往”“导航到”等限定的近义表达；
- 接受 A/B/C 的少量常见同音识别结果；
- 必须包含正确的目标点，等待 A 点时不会接受 B/C 点；
- “目标点 A”只有地点信息、没有动作意图，不会触发；
- 包含“不要”“停止”“取消”“别去”等否定词时直接拒绝；
- 环境声音、空文本和普通闲聊均不会触发；
- 只有当前状态所期待的命令通过后，状态机才会下发导航目标。

核心判断思路可以简化为：

```python
if contains_negation(text):
    return False
if not target_point_matches(text, expected_point):
    return False
if contains_navigation_keyword(text):
    return True
return constrained_fuzzy_match(text, expected_command)
```

正式配置使用：

```yaml
voice_accept_any_recognized_text: false
voice_text_match_mode: command
```

其中 `first_nonempty` 模式只用于麦克风诊断，不应在正式机器人任务中启用。

我们还为接受与拒绝场景编写了单元测试。例如“请带我去目标店 A”可以接受，而“今天天气很好”“目标点 A”“请引领到目标点 B”“不要引领到目标点 A”必须拒绝。相比单纯调高相似度阈值，这种规则更符合机器人控制的安全边界，也更容易解释和测试。

## 三、导航状态机与动态避障

主节点通过 Nav2 的 `NavigateToPose` action 下发目标。初始位姿、A/B/C 点以及 A→B 路线引导点统一保存在 `waypoints.yaml`，避免同一个坐标散落在代码中。

项目没有给所有路段使用一套固定参数，而是根据阶段动态切换：

- 靠近终点时，减小软膨胀范围并降低速度，便于进入目标区域；
- B→C 路段扩大安全距离、提高障碍代价并限制最大速度；
- 离开 B 点的短距离内继续保留终点配置，避免切换参数后原地反复；
- 从 C 点前往二维码目标时，对短距离起步单独设置转向参数。

动态障碍播报由激光数据触发，不依赖预设时间或固定位置。系统关注前方角度范围和距离变化，并设置冷却时间及每路段播报次数，避免持续重复播报影响语音交互。

需要强调的是，播报检测并不替代 Nav2 避障。它向随行人员提供提示，真正的减速、规划和速度控制仍由代价地图、控制器与速度平滑器完成。

## 四、多帧视觉比“单张出答案”更可靠

比赛现场常见运动模糊、曝光变化、视角倾斜和网络视频流延迟。项目中的三类视觉任务都没有把单帧结果直接作为最终结论，而是使用多帧验证。

### 标识牌识别

标识牌流程先提取彩色字符候选，再根据高度、对齐、间距及白色背景支持度选择数字与字母候选对。必要时进行白纸定位和透视校正，定位不可靠时回退到原始字符区域。

字符分类器采用轻量的 HOG + PCA + KNN 模型，可以在 CPU 上离线运行。颜色判断只统计字符真实掩码内部的 HSV 像素，减少背景颜色干扰。数字、字母和两者颜色分别进行加权投票，低置信度或帧间不一致时拒绝输出。

### 二维码识别

二维码内容支持 `A`、`QR-A-01`、`TARGET: B`、`目标点C` 等常见形式。每帧解析出候选目标后进行投票；票数不足或结果冲突时不执行导航，从而避免一次误码直接改变机器人的目标点。

### 红绿灯识别

红绿灯模块只检测高亮、高饱和度的红色和绿色连通区域，并通过多帧票数和最低置信度输出结果。识别为绿灯时播报“可以通行”，识别为红灯时播报“不可通行”。

## 五、如何运行项目

推荐环境为 Ubuntu 22.04、ROS 2 Humble、Nav2 和 Python 3.10。运行前还需要已经可用的 Agibot D1 基础工作区，并确保 `/scan`、`/odom`、`/tf` 以及速度控制链路正常。

克隆项目后，先安装依赖并下载离线语音模型：

```bash
git clone https://github.com/bob142857/raicom2026-agibot-d1.git
cd raicom2026-agibot-d1

source /opt/ros/humble/setup.bash
source /absolute/path/to/d1_base_ws/install/setup.bash

sudo apt install -y python3-pip python3-opencv python3-numpy \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  alsa-utils espeak-ng pulseaudio-utils

python3 -m pip install -r agibot_d1_auto_nav/requirements.txt
python3 scripts/download_speech_model.py --model base
```

复制本地配置并根据实际设备修改麦克风、RTSP 地址和点位：

```bash
cp .env.example .env
cp agibot_d1_auto_nav/config/auto_nav_params.yaml \
  agibot_d1_auto_nav/config/auto_nav_params.local.yaml
cp agibot_d1_auto_nav/config/waypoints.yaml \
  agibot_d1_auto_nav/config/waypoints.local.yaml
```

编译并执行不驱动机器人的测试：

```bash
colcon build --symlink-install --base-paths . \
  --packages-select agibot_d1_auto_nav
source install/setup.bash

python3 -m pytest agibot_d1_auto_nav/test -q
python3 agibot_d1_auto_nav/tools/test_competition_microphone.py \
  --config agibot_d1_auto_nav/config/auto_nav_params.local.yaml \
  --expected "请引领到目标点A" --timeout 20
```

确认地图、TF、定位和急停均正常后，再启动比赛流程：

```bash
ros2 launch agibot_d1_auto_nav auto_nav_bringup.launch.py \
  map_yaml:=/absolute/path/to/map.yaml \
  auto_nav_params_file:="$(pwd)/agibot_d1_auto_nav/config/auto_nav_params.local.yaml" \
  waypoints_params_file:="$(pwd)/agibot_d1_auto_nav/config/waypoints.local.yaml"
```

真实机器人首次运行时，请清空测试区域、准备急停并降低速度。先验证地图、TF、点位和拒识逻辑，再逐步开启完整任务。

## 六、从比赛代码到开源仓库，我们做了哪些整理

比赛工作空间里通常混有编译输出、日志、缓存、模型权重、现场图片和本机绝对路径，直接上传整个工作空间既不利于复现，也可能泄露信息。

本次开源只保留实际比赛代码目录，并完成了以下工作：

1. 添加 MIT License、根目录 README、第三方依赖说明和发布审计清单；
2. 使用 `.env.example` 和忽略的本地 YAML 管理设备路径与现场参数；
3. 排除 `build/`、`install/`、`log/`、`tmp/`、缓存和语音模型权重；
4. 提供语音模型下载脚本，不把数百 MiB 权重提交到 Git；
5. 清理个人绝对路径和常见 Token 特征；
6. 明确说明本仓库依赖外部 D1 基础工作区，而不是把厂商 SDK 混入项目；
7. 保留运行必需的轻量视觉分类模型、测试和配置文件。

这一步很重要：开源不是简单地“把代码设为 Public”，而是要让使用者知道项目包含什么、不包含什么、如何配置、如何验证，以及哪些操作会让真实机器人运动。

## 七、项目限制与后续方向

当前项目仍有一些明确边界：

- 需要 Agibot D1 基础工作区和真实硬件接口；
- 点位、地图、相机地址与音频设备必须按实际环境配置；
- 视觉方案针对比赛任务设计，不是通用场景感知模型；
- 完整硬件验证需要在 Ubuntu 22.04 / ROS 2 Humble 和真实机器狗上完成；
- 模糊语音匹配强调可解释和可控，不能替代面向开放场景的自然语言理解。

后续可以继续完善仿真启动、录制数据回放、统一诊断面板和更系统的硬件在环测试，也欢迎社区开发者提交 Issue 和 Pull Request。

## 结语

这个项目最大的收获，不是把某个单项识别算法做到复杂，而是把语音、视觉、导航和安全约束组织成一条能够在真实机器人上运行的任务链。

语音交互遵循清晰的控制边界：只有识别内容与当前任务的控制意图、目标点和安全条件都一致，系统才允许机器人执行相应动作。这种约束式设计兼顾了现场识别容错能力与真实机器人运行的可控性。

项目现已采用 MIT License 开源，欢迎参考、复现和改进：

https://github.com/bob142857/raicom2026-agibot-d1
