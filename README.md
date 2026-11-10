# 旅客识别与行李框绑定

本项目只包含闸机抓拍、供框台重识别、旅客与行李框绑定。
日常使用联合工作台，也可以单独打开三块业务界面。

姿态识别、放置行李动作分析和标注训练已迁入独立项目：
`D:\code\pose_action`。两个项目分别维护源码、配置、模型和测试。

详细目录、源码阅读顺序和验证结果见
[项目交付与源码导航](docs/PROJECT_HANDOFF.md)。

## 启动

双击 `start_integration.cmd`，或执行：

```powershell
conda run --no-capture-output -n mmpose python D:\code\mediapipe-master\pose\start_integration.py
```

首次使用时选择两路视频或 RTSP，读取首帧，再标定闸机抓拍区和供框位区域。
每个供框位只画一个区域。启用区域后才参与识别。
配置保存在 `configs/integration_config.json`。
运行结果写入 `result/integration/<时间_编号>/`。

默认本机直连。选择 MQTT 时，统一配置服务器、认证和主题。
真实 RFID 接入应向配置的 RFID 主题发布消息。
完整使用说明见 [联合工作台](docs/INTEGRATION_WORKBENCH.md)。

## 源码

```text
src/pose_project/
  gate/          闸机检测、跟踪、优选照片和旅客消息
  reid/          供框台检测、外观特征、匹配和旅客锁定
  binding/       旅客位置与行李框 RFID 绑定、实时接口和回放
  integration/   联合配置、会话控制和后台调度
  apps/          联合与独立桌面界面
  video/         本地文件与网络流取帧
  common/        公共资源路径
  __main__.py    命令入口
```

| 目录 | 内容 |
| --- | --- |
| `configs/` | 联合、闸机、重识别、绑定和模拟配置 |
| `models/gate/` | 闸机人体、人脸和身份特征模型 |
| `models/reid/` | 行人检测和重识别权重 |
| `tests/` | 当前主项目的回归测试 |
| `scripts/` | 主项目验证与 EXE 构建 |
| `packaging/` | PyInstaller 配置及发布启动脚本 |
| `docs/` | 使用说明、接口协议、交付文档及历史记录 |
| `data/` | 主业务视频、绑定模拟数据和验证素材 |
| `result/` | 运行结果、日志和历史验证资料 |
| `dist/` | 历史 EXE 与 ZIP |
| `archive/` | 整理前备份、旧版本和临时诊断 |

## 环境与独立入口

沿用现有 Conda 环境 `mmpose`，Python 3.10.20。
本次没有重建环境或升级依赖。
新机器的联合依赖见 `requirements-integration.txt`。

安装本地源码后，可以从任意目录调用各入口：

```powershell
conda run --no-capture-output -n mmpose python -m pip install --no-deps -e D:\code\mediapipe-master\pose
conda run --no-capture-output -n mmpose python -m pose_project integrate
conda run --no-capture-output -n mmpose python -m pose_project gate
conda run --no-capture-output -n mmpose python -m pose_project reid
conda run --no-capture-output -n mmpose python -m pose_project binding
```

`simulate` 生成绑定模拟数据，`replay` 回放并评估绑定结果。
对应说明见 [绑定工作台](docs/BINDING_GUI.md)、
[模拟与回放](docs/BINDING_SIMULATION.md) 和
[检测消息协议](docs/MQTT_DETECTION_PROTOCOL.md)。

## 验证与发布

在本项目根目录执行：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONIOENCODING = "utf-8"
conda run --no-capture-output -n mmpose python -m unittest discover -s tests
conda run --no-capture-output -n mmpose python scripts/verify_project.py --output result/new_project_check
conda run --no-capture-output -n mmpose python scripts/verify_integration.py --output result/new_integration_check
```

验证使用新输出目录，保留旧结果。
服务器发布说明见 [EXE 部署](docs/EXE_DEPLOYMENT.md)。
本次整理没有重新构建 EXE，历史发布包仍对应原来的版本。

本项目保留原 [Apache 2.0 许可证](LICENSE)。
