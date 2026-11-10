# Pose 项目目录整理记录

日期：2026-10-22。项目：`D:\code\mediapipe-master\pose`。

这是历史整理记录。当前源码位置与验证结果见
[2026-11-10 项目交付与源码导航](PROJECT_HANDOFF.md)。

## 完成内容

已将 49 个平铺文件迁入分类目录。
19 个原有 Python 模块进入 `src/pose_project/`。
16 个原有测试进入 `tests/`，另增加 3 项资源路径检查。
配置、模型、文档和 C++ 示例分别归入独立目录。
原有数据、标注和历史结果仍在 `data/` 和 `result/`。

| 目录 | 内容 |
| --- | --- |
| `src/pose_project/apps/` | 视频界面、绑定工作台、模拟生成窗口、离线视频和演示入口 |
| `src/pose_project/video/` | 视频来源配置和取流 |
| `src/pose_project/vision/` | 人体检测和姿态推理 |
| `src/pose_project/actions/` | 动作识别和动作消息输出 |
| `src/pose_project/binding/` | 绑定算法、实时接口、模拟生成和回放评估 |
| `src/pose_project/training/` | 片段标注、特征提取和分类器训练 |
| `src/pose_project/common/` | 项目根目录、配置、模型、数据和结果路径 |
| `tests/` | 17 个测试文件 |
| `configs/` | 5 份原有 JSON 配置 |
| `models/` | 4 个原有 ONNX 模型和 2 份动作模型文件 |
| `docs/` | 使用说明、MQTT 协议、审核及目录整理记录 |
| `examples/cpp/` | 原有 C++ 演示和 CMake 配置 |
| `scripts/` | 实际运行验证脚本 |

内部导入和测试替换路径均已改为 `pose_project` 包路径。
所有默认资源路径统一由 `common/project_paths.py` 提供。
配置不再按源码所在子目录查找，结果也不会误写入源码目录。
新增 `pyproject.toml`，采用本地可编辑安装。
安装文件包含源码；模型、配置和数据保存在本地项目目录。

## 启动方式

已在原有 `mmpose` 环境完成安装，原有依赖版本保持不变。
首次安装或移动整个项目后，执行：

```powershell
conda run --no-capture-output -n mmpose python -m pip install --no-deps -e D:\code\mediapipe-master\pose
```

常用入口：

```powershell
conda run --no-capture-output -n mmpose python -m pose_project video
conda run --no-capture-output -n mmpose python -m pose_project binding
conda run --no-capture-output -n mmpose python -m pose_project --help
```

其余子命令为 `simulate`、`replay`、`label`、`extract`、`train`、`offline` 和 `demo`。
原来直接执行根目录 `.py` 文件的命令已替换。
项目根目录 README 和现行使用指南均已同步。

## 验证结果

| 检查 | 实际结果 |
| --- | --- |
| 自动化测试 | 109 项全部通过；包含原有 106 项和新增 3 项 |
| 包内模块 | 28 个 Python 文件完成语法解析和导入 |
| 统一入口 | 9 个子命令的帮助均正常退出 |
| 安装后的 `pose` 命令 | 正常运行 |
| 其他工作目录启动 | 在临时目录验证全部子命令帮助及实际运行命令 |
| 模拟生成 | 3600 条检测消息、9 条 RFID 消息，共 3609 条回放消息 |
| 绑定回放 | 9/9 匹配独立答案；错误、未绑定、多人封存、缺少及额外结果均为 0 |
| 真实片段特征 | 2 个片段成功，分别提取 5 帧和 6 帧有效姿态 |
| 分类器训练 | 使用原有 102 个片段重新训练，保存后可加载 |
| 实际视频输出 | 40 帧预览全部成功解码，生成 1 条动作事件 |
| 整理前后输出比较 | 预览 MP4、逐帧记录和动作事件的 SHA-256 分别一致 |
| 视频界面 | 实际模型加载、播放、跳转至第 45 帧、播放至第 66 帧和线程停止通过 |
| 绑定界面及接口 | 控件、区域图、本机 HTTP/MQTT、QoS 确认等原有测试通过 |
| 原有 JSON/ONNX 文件 | 12 个文件与迁移前 SHA-256 一致，含编辑器设置 |
| 原有数据与历史结果 | 137 个文件仍在原位置，大小及修改时间保持不变 |
| 核心实现 | 7 个配置、取流、推理、动作和绑定模块与迁移前逐字节一致 |

第一次完整检查在界面跳转阶段超过了共用的 20 秒时限。
独立复测约 18 秒完成，最终完整检查中的界面步骤约 5 秒完成。
验证脚本现为播放和跳转分别计时，并保存实际帧进度。
未修改应用的播放或推理逻辑。
首次失败日志保留在证据目录中。

训练准确率仍为 0.7255，是原有小样本训练集上的结果。
该数值不能代表独立测试或现场准确率。
本次训练结果仅保存在验证目录，未替换原有运行模型。

## 备份与证据

所有证据位于 `result/project_layout_20261022_145533/`：

- `before_restructure.zip`：迁移前 59 个源码、配置、模型和文档文件。
- `files_before.json`：备份文件的 SHA-256。
- `moves.json`：49 个文件的原路径和新路径。
- `assets_before.json`：137 个原有数据及历史结果文件的位置、大小和修改时间。
- `preserved_configs_models.json`：配置、模型及编辑器设置的内容核对。
- `git_status_before.txt`：本轮开始前的 Git 状态。
- `editable_install.log`：本地包安装日志。
- `tests.log`：109 项测试记录。
- `runtime.log`、`gui_diagnostic.log`：首次检查及独立排查记录。
- `runtime_final/verification.json`：最终完整运行检查结果。
- `runtime_final/`：各命令日志、生成数据、绑定结果、训练结果和预览视频。
- `runtime_final/video_progress.json`：界面播放和跳转的实际帧进度。
- `ui/`：绑定工作台和模拟生成界面的测试截图。
- `delivery_verification.json`：最终保存和内容一致性核对。

原有 Git 仓库和已有修改均保留，未创建提交。
上次审核报告中关于没有 Git 仓库的判断已补充更正。

## 验证范围

本次使用现有视频和模型，界面检查采用 Qt 离屏模式。
真实桌面显示、摄像头、现场 RFID、现场时钟与网络仍需实际设备验证。
未编译 C++ 示例，本机未找到 CMake。
多人身份稳定性和现场绑定效果不能由当前模拟数据证明。
