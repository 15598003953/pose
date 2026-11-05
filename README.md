# Pose 项目

项目包含视频动作分析和旅客与框绑定两套功能。

- 视频分析：人体检测、33 个姿态关键点、放置行李动作基线、区域配置和动作 MQTT 输出。
- 旅客与框绑定：接收稳定旅客 ID、检测框和 RFID，按位置历史生成绑定结果。支持文件回放、MQTT 和 HTTP。

两套功能独立运行。动作识别不检测行李物体，绑定算法也不依赖动作分类结果。
现有标注、视频和历史结果保留原位置。模型统一放入 `models/`。

## 三部分联合调试

统一工作台重新组织了闸机采集、供框台重识别、旅客与行李框绑定。
只打开一个窗口，不打开或嵌入三个旧窗口。原独立入口继续保留。

```powershell
conda run --no-capture-output -n mmpose python D:\code\mediapipe-master\pose\start_integration.py
```

也可执行 `pose\start_integration.cmd`，或运行 `python -m pose_project integrate`。
首次使用先选择两路视频并读取首帧，在“区域”中标定抓拍区、识别区和操作区。
点击“启动联调”后，两路预览、旅客库、消息、绑定结果和终端日志使用同一会话。
底部可以注入明确标记的 RFID 测试事件，也可手动导入上半身图。

默认使用本机直连，不需要 MQTT 服务器，也不会向外发送消息。
选择“MQTT 联调”后，统一填写服务器、认证和四个主题。
程序不会自建 MQTT 服务器。真实 RFID 接入程序需要向配置的 RFID 主题发布消息。
闸机和供框台均支持本地视频或 RTSP，可分别选择，也可同时连接两路 RTSP。
在“视频”页选择来源类型，填写地址后读取首帧，再标定区域。

联调配置单独保存为 `configs/integration_config.json`。
每次运行创建 `result/integration/<时间_随机字段>/`，保留配置、图片、消息和判定日志。
日志中的图片正文只保留长度，界面可以查看原图。MQTT 密码不写入配置文件。
RTSP 地址会保存到配置中，地址内的摄像头账号密码也会保存。
完整说明见 [联合调试工作台](docs/INTEGRATION_WORKBENCH.md)。

### EXE 发布

使用 Python 的 PyInstaller 构建 Windows 64 位便携发布包：

```powershell
conda run --no-capture-output -n mmpose python scripts\build_integration_exe.py
```

发布包包含 EXE、模型、配置和运行库。复制整个文件夹后运行，无需服务器安装 Conda。
支持图形界面和 `PoseIntegration.exe --headless` 终端模式。
部署条件和验证命令见 [EXE 部署说明](docs/EXE_DEPLOYMENT.md)。

## 运行环境

当前使用 Conda 环境 `mmpose`，Python 3.10.20。
`requirements.txt` 记录当前验证的依赖版本，`environment.yml` 用于重建同名环境。
已有可用环境时直接使用，无需重建。

项目已按 `src` 包结构整理。首次运行先安装本地源码：

```powershell
conda run --no-capture-output -n mmpose python -m pip install --no-deps -e D:\code\mediapipe-master\pose
```

当前机器已完成安装。源码修改立即生效，无需反复安装。
配置和模型由本地项目目录读取，不打包进安装文件。
更换项目位置后，重新执行可编辑安装。

以下命令在 `D:\code\mediapipe-master` 下执行：

```powershell
conda run --no-capture-output -n mmpose python -m pose_project video
conda run --no-capture-output -n mmpose python -m pose_project binding
```

在其他机器上安装：

```powershell
conda env create -f .\pose\environment.yml
conda run --no-capture-output -n mmpose python -m pip install --no-deps -e .\pose
```

`paho-mqtt` 是绑定工作台的必要依赖。视频应用关闭 MQTT 时也可运行。
姿态推理默认使用 ONNX Runtime 的 CPU 后端，人体检测使用 OpenCV DNN 的 CPU 后端。
`MPPOSE_ALLOW_OPENCV_DNN=1` 可显式启用姿态模型的 OpenCV 备用实现。
检测演示中的后端选项只控制人体检测，不改变姿态模型的默认执行后端。

## 项目结构

```text
pose/
  src/pose_project/
    apps/          桌面界面和演示入口
    video/         视频来源配置和取流
    vision/        人体检测和姿态推理
    actions/       动作识别和动作消息
    binding/       绑定算法、数据接口、模拟与回放
    training/      标注、特征提取和训练
    common/        项目资源路径
    __main__.py    统一命令入口
  tests/           自动化测试
  configs/         JSON 配置
  models/          ONNX 模型和动作模型
  docs/            使用说明、接口协议和审核记录
  examples/cpp/    上游 C++ 演示
  scripts/         项目验证脚本
  data/            现有视频、标注和实验数据
  result/          运行结果与历史验证资料
  pyproject.toml   包元数据和安装入口
  environment.yml
  requirements.txt
  README.md
```

下表源码路径均相对于 `src/pose_project/`。

| 功能 | 文件 |
| --- | --- |
| 视频界面、区域选择与后台推理 | `apps/pose_app.py` |
| 视频来源配置、最新帧读取 | `video/app_config.py`、`video/latest_frame_capture.py` |
| 本地资源路径 | `common/project_paths.py` |
| 人体检测与姿态推理 | `vision/mp_persondet.py`、`vision/mp_pose.py` |
| 公共上半身特征、模型分类和规则识别 | `actions/pose_action_recognizer.py` |
| 动作 MQTT 输出 | `actions/mqtt_publisher.py` |
| 标注、特征提取、训练 | `training/` |
| 离线视频规则识别 | `apps/run_put_luggage_offline.py` |
| 绑定算法、离线回放和评估 | `binding/passenger_tray_binding.py`、`binding/replay_binding.py` |
| 绑定工作台及实时接口 | `apps/binding_app.py`、`binding/binding_runtime.py` |
| 模拟数据及生成窗口 | `binding/generate_binding_simulation.py`、`apps/binding_simulation_dialog.py` |

原来直接执行根目录脚本的命令，统一改为 `python -m pose_project <命令>`。
运行 `python -m pose_project --help` 可查看全部入口。
可编辑安装也提供 `conda run -n mmpose pose <命令>`。
训练和实时分类共用一套特征计算；特征提取不再导入视频界面。

## 视频分析

程序使用目录内的两个模型：

```text
models/person_detection_mediapipe_2023mar.onnx
models/pose_estimation_mediapipe_2023mar.onnx
```

设置环境变量 `PERSON_DET_MODEL` 可指定其他人体检测模型。
显式指定的路径不存在时，程序报告错误。程序不会自动下载模型。

视频应用默认加载 `models/put_luggage_centroid_model.json`。
模型缺失或无效时，应用回退到规则识别。
模型按视频时间采样，跳帧不会再按帧号取余来延长采样时间。
拖动播放进度或修改区域后，程序清除旧动作历史。

配置文件：

- `configs/source_config.json`：本地视频、RTSP 或 RTMP，以及显示与推理间隔。
- `configs/roi_config.json`：多个区域及其对应画面尺寸。
- `configs/mqtt_config.json`：动作消息服务器、主题和认证信息。

网络流启用低延迟读取后，只保存最新画面，推理在后台运行。
动作消息与事件日志含区域编号、置信度、视频时间和来源。
事件日志追加写入 `result/<视频名>_put_luggage_events.jsonl`。
没有配置区域时，区域编号为 `FULL`。

### 标注和训练

```powershell
conda run --no-capture-output -n mmpose python -m pose_project label --video .\pose\data\2026-08-12_102107_553.mp4 --resume
conda run --no-capture-output -n mmpose python -m pose_project extract --stride_frames 10
conda run --no-capture-output -n mmpose python -m pose_project train
```

默认片段长 4 秒，步长 2 秒。标注按键为 `p` 放置行李、`n` 非放置行李、
`s` 跳过、`b` 上一段、`r` 重播、`q` 保存退出。
继续编辑已有标注时使用 `--resume`。默认标注文件位于 `data/clip_labels/labels.csv`。

特征输出到 `data/clip_features/`，训练结果输出到 `data/clip_classifier_baseline/`。
训练命令不会替换当前运行模型。确认训练结果后，再将模型及报告更新到 `models/`。
使用 `--output_dir` 可以为实验指定独立目录，保留旧结果。

当前模型是 NumPy 最近质心分类基线。报告中的准确率来自训练集，
数据量较小且片段重叠，不能当作独立测试或现场准确率。

### 离线视频与演示

```powershell
conda run --no-capture-output -n mmpose python -m pose_project offline --video .\pose\data\2026-08-12_102107_553.mp4 --output_dir .\pose\result\offline_run --max_frames 40
conda run --no-capture-output -n mmpose python -m pose_project demo --help
```

离线脚本使用规则识别，输出事件、逐帧调试记录和预览 MP4。
同名结果会覆盖，重复实验应指定不同的输出目录。
`demo.py` 支持图片和本机摄像头，导入模块或查看帮助不会连接网络。

## 旅客与框绑定

详细说明见 [工作台与接口](docs/BINDING_GUI.md) 和 [模拟数据与算法](docs/BINDING_SIMULATION.md)。
检测端接入见 [MQTT 协议](docs/MQTT_DETECTION_PROTOCOL.md)。

```powershell
conda run --no-capture-output -n mmpose python -m pose_project simulate --output .\pose\data\binding_new
conda run --no-capture-output -n mmpose python -m pose_project replay --input .\pose\data\binding_new\replay.jsonl --expected .\pose\data\binding_new\expected_bindings.jsonl --output .\pose\result\binding_new
```

生成和回放目录须为新目录或空目录。默认模拟数据已经存在，可直接从工作台回放。
`configs/binding_config.json` 保存实际算法参数；`configs/binding_simulation_config.json` 保存模拟场景。
独立预期答案仅参与事后评估，不参与算法判定。

每条成功绑定只包含：

```json
{
  "roi_id": "A",
  "passenger_id": "0123456789abcdef0123456789abcdef-2026-10-22",
  "rfid": "RFID-A-0001"
}
```

多人同时符合条件时保存候选依据，不任意选择旅客。
真实应用仍需验证稳定 ID、时钟同步、坐标标定和网络延迟。

## 验证

从仓库根目录发现并运行全部测试：

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:QT_QPA_PLATFORM = "offscreen"
conda run --no-capture-output -n mmpose python -m unittest discover -s .\pose\tests -p "test_*.py" -v
```

`--no-capture-output` 配合 UTF-8 可避免 Windows 下 Conda 转发中文输出失败。
测试覆盖绑定回放、界面控件、本机 HTTP/MQTT、特征、配置和线程状态。
真实模型与视频检查见 [审核报告](docs/AUDIT_2026-10-22.md)。
本次目录整理的验证见 [项目化记录](docs/PROJECT_LAYOUT_2026-10-22.md)。

完整运行检查会使用实际模型和现有视频，结果写入新的目录：

```powershell
conda run --no-capture-output -n mmpose python .\pose\scripts\verify_project.py --output .\pose\result\verify_new
```

## C++ 演示

`examples/cpp/` 保留上游独立演示：

```powershell
cmake -S .\pose\examples\cpp -B .\pose\build\cpp
cmake --build .\pose\build\cpp --config Release
```

需要 CMake >= 3.24、C++ 编译器及 OpenCV >= 4.10 的开发文件。
Python 的 OpenCV 安装包不能替代 C++ 开发环境。
从 `pose` 目录启动生成的程序，默认读取 `models/` 中的检测模型和姿态模型。
可通过 `--person_model` 和 `--model` 另行指定。
本次环境没有 CMake，未完成 C++ 编译及运行验证。

## 许可证与来源

本目录采用 [Apache 2.0](LICENSE) 许可证。
姿态模型源于 MediaPipe BlazePose，ONNX 模型及演示沿用 OpenCV Zoo 版本。
