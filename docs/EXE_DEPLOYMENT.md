# 联调程序 EXE 服务器部署

适用日期：2026-11-05。

## 运行条件

本发布包面向 Windows 64 位。Linux 服务器不能直接运行此 EXE。
目标机器无需安装 Python、Conda、PyTorch 或单独的 CUDA Toolkit。
使用 NVIDIA GPU 时，服务器必须安装与包内 CUDA 12.8 兼容的 NVIDIA 驱动。
无可用 GPU 时，现有算法自动使用 CPU，处理速度需现场检查。
打包电脑验证通过不代表目标服务器已经验证通过。

完整复制或解压 `PoseIntegration` 文件夹。保留其中所有文件及 `_internal` 目录。
这是 PyInstaller 目录发布包，单独复制 EXE 无法运行。
请放在有写权限的目录，例如 `D:\apps\PoseIntegration`。
配置、缓存和结果保存在发布目录内，不依赖开发电脑的盘符。
构建脚本统一 Qt 和 Torch 使用的 VC++ 运行库，避免旧 DLL 冲突。

## 首次配置

双击 `PoseIntegration.exe`。在“视频”页配置两路 RTSP 或本地视频。
读取首帧，检查旋转方向，再标定抓拍区、识别区和操作区。
检查消息连接和绑定参数，保存联调配置。
默认使用本机消息连接；选择 MQTT 后需提供现有服务器。
程序不包含或启动 MQTT 服务器。

配置文件：`configs\integration_config.json`。
相对文件路径以配置文件所在目录为起点。
程序把发布目录内的文件路径保存为相对路径，方便整体迁移。
MQTT 密码不保存，可在界面填写，或使用 `MQTT_PASSWORD` 环境变量。
RTSP 地址中的摄像头账号密码会保存在配置中。

## 终端运行

在发布目录打开 PowerShell：

```powershell
.\PoseIntegration.exe --check
.\PoseIntegration.exe --check-models
.\PoseIntegration.exe --self-test
.\PoseIntegration.exe --headless --config .\configs\integration_config.json
```

`--check` 检查内置依赖和模型文件。
`--check-models` 实际加载闸机模型、YOLO、ByteTrack 和 FastReID，检查推理结果。
`--self-test` 只检查模拟消息绑定，不连接摄像头或真实 RFID。
`--headless` 根据已保存的配置运行两路视频和绑定任务。
按 Ctrl+C 请求正常停止。可加 `--duration 60` 在 60 秒后请求停止。
直接执行 `start_server.cmd` 也可启动终端模式。
图形界面可加 `--duration 3`，在 3 秒后关闭，用于检查界面是否能启动。

需要指定另一份配置时使用 `--config <JSON文件路径>`。
默认保留控制台，方便检查启动错误。
自动启动、Windows 服务注册和远程服务器部署需单独配置，本包不修改服务器设置。

## 文件与结果

```text
PoseIntegration.exe           主程序
_internal/                    Python、Qt、Torch、OpenCV、模型等运行文件
configs/                      可编辑的配置
runtime/                      库缓存，运行时生成
result/integration/           每次会话的日志、图片和绑定结果
start_server.cmd              终端启动入口
check_environment.cmd         环境及模型检查入口
build_info.json               构建环境和依赖版本
manifest.json                 发布文件大小与 SHA-256
licenses/                     收集到的第三方许可证
```

发布包包含本项目使用的四个主模型，以及闸机现有的人脸模型和级联文件。
未打包原始旅客视频、历史抓拍、旧会话日志和训练数据。
MediaPipe 五官网格是原程序的可选功能，本包沿用 YuNet 五点描线。
联调使用 OpenCV DNN 和 PyTorch，不包含其他动作识别入口或 ONNX Runtime。

## 重新构建

在现有 `mmpose` 环境中：

```powershell
conda run --no-capture-output -n mmpose python scripts\build_integration_exe.py
```

构建入口为 `packaging\integration.spec`。
额外构建依赖记录在 `requirements-build.txt` 中。
默认生成新的发布目录、ZIP 和 SHA-256 文件，不覆盖已有发布包。
