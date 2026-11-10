# 联调程序 EXE 服务器部署

适用日期：2026-11-10。

当前源码目录见 [项目交付文档](PROJECT_HANDOFF.md)。
姿态动作分析已独立到 `D:\code\pose_action`，不参与主项目打包。
本次目录整理未重建 EXE，历史发布包仍对应原来的版本。

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
读取首帧，检查旋转方向，再标定抓拍区和供框台区域。
检查消息连接和绑定参数，保存联调配置。
默认使用本机消息连接；选择 MQTT 后需提供现有服务器。
程序不包含或启动 MQTT 服务器。

配置文件：`configs\integration_config.json`。
相对文件路径以配置文件所在目录为起点。
程序把发布目录内的文件路径保存为相对路径，方便整体迁移。
MQTT 密码不保存，可在界面填写，或使用 `MQTT_PASSWORD` 环境变量。
RTSP 地址中的摄像头账号密码会保存在配置中。

## 区域编辑

在“区域”页新增供框台区域，每个供框位只画一个区域。
框旁的删除按钮删除单个框；每组“启用”选项旁的删除按钮删除整组区域。
闸机抓拍区可以独立删除和重画。
未完成的配置可以保存并重新打开。启动前检查已启用区域，缺少区域时提示具体区域编号。
未启用区域不参与识别和绑定。

每个供框台可独立选择“多人排队”的队首方向：下、上、左、右或最大目标，
也可以关闭队首筛选。方向以旋转后的预览画面为准。更改后自动保存，重启联调后生效。
启用队首筛选后，每个区域只识别一个目标。现任队首仍在场时，
其他目标须领先区域边长的 8%，或面积达到现任队首的 125%，才会切换。
队首缺失达到第 8 个处理帧时，允许选择新队首；等待时不输出旅客身份。
新轨迹仍需完成原有三帧确认，并达到匹配阈值。

## 识别结果与性能

“重识别结果”显示当前检测目标、候选身份、相似度、区域、轨迹及匹配状态。
低分候选只用于调试，不会作为绑定身份发送。过期结果会清除预览中的检测框。
供框台 RTSP 预览独立于推理刷新。启动时先预热跟踪和特征模型，
处理旅客入库后重新读取最新帧，识别速度较慢时跳过旧帧。

“性能调试”显示实际推理设备、识别帧率、各阶段耗时、位置延迟和等待数量。
勾选“记录性能日志”后，会话目录生成 `performance.jsonl`，每秒至多追加一条。
“现场测试”页可以填写测试编号、地点、场景和人工核对结论。启用后会话目录生成
`field_test.json` 和 `field_test_events.jsonl`，记录每帧识别结果、绑定判定、
MQTT 状态、后台错误和人工观察。启用现场测试记录会自动记录性能日志。
“算法”页可设置识别间隔、检测输入尺寸、预览帧率及翻转增强。
默认 200 ms、960、15 fps、翻转增强开启。修改计算参数后需复核现场识别效果。
位置有效期和绑定规则保持原值，超时结果不会通过修改时间戳变成新位置。

同一轨迹每 10 个处理帧刷新一次重识别特征，无轨迹编号时逐帧提取。
新增闸机旅客时保留当前队首、跟踪记录和已确认身份，避免因人物库更新而重新选人。
旅客首次完成匹配后，任务锁定到原区域和轨迹，所有供框台共享锁定记录。
原目标离开、轨迹过期或更新旅客图片都不会把该旅客重新分配给其他目标。
多个目标同时符合条件时暂不锁定；没有轨迹编号时等待稳定轨迹。
结束并重新启动联调会创建新会话，清空锁定记录。
首次误匹配或跟踪器把同一有效轨迹编号交给另一人，仍需现场验证和排查。
CPU 检测阶段使用 8 线程，特征提取阶段使用 20 线程，OpenCV 使用 4 线程。
8 帧等待和 10 帧刷新按实际处理帧计数；识别间隔为 200 ms 时，
分别约为 1.6 秒和 2 秒。推理较慢时会更长，现场效果和速度需要实测。

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
field_test.json               现场测试元数据和会话汇总
field_test_events.jsonl       现场测试结构化事件
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

指定程序名时添加 `--name text_v2`。主程序、目录及启动脚本会使用同一个名称。

构建入口为 `packaging\integration.spec`。
额外构建依赖记录在 `requirements-build.txt` 中。
默认生成新的发布目录、ZIP 和 SHA-256 文件，不覆盖已有发布包。
