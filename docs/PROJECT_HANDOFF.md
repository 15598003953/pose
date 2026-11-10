# 旅客主项目交付与源码导航

整理日期：2026-11-10。
项目位置：`D:\code\mediapipe-master\pose`。

主项目只保留闸机抓拍、供框台重识别、旅客与行李框绑定。
姿态动作分析已迁入 `D:\code\pose_action`，单独交付与验证。

## 1. 三块源码位置

| 业务 | 源码目录 | 职责 |
| --- | --- | --- |
| 闸机口识别 | `src/pose_project/gate/` | 人体与人脸检测、轨迹维护、照片优选、稳定旅客编号与图片消息 |
| 供框台重识别 | `src/pose_project/reid/` | 行人跟踪、外观特征、旅客库匹配、队首选择和任务锁定 |
| 人包绑定 | `src/pose_project/binding/` | 根据旅客位置历史与行李框 RFID 建立关联 |

“人包绑定”目前实际关联旅客和带 RFID 的行李框。
算法不直接检测包，也不判断包是否已经放入框内。

## 2. 每个目录放什么

```text
pose/
  start_integration.py         联合工作台源码启动入口
  start_integration.cmd        Windows 启动脚本
  src/pose_project/
    __main__.py               六个命令的统一入口
    apps/                     联合界面与三块业务的独立界面
    gate/                     闸机抓拍算法和旅客消息
    reid/                     重识别模型、匹配与独立检索
    binding/                  绑定算法、消息接口和模拟回放
    integration/              联合配置、会话和后台任务
    video/                    文件兼容读取与实时流最新帧
    common/                   统一资源路径
  configs/                    JSON 配置
  models/gate/                闸机人体、人脸和身份特征模型
  models/reid/                供框台检测与重识别权重
  tests/                      主项目自动化回归
  scripts/                    主项目验证和 EXE 构建
  packaging/                  发布配置、运行设置与服务器启动脚本
  docs/                       说明、协议、交付文档和历史记录
  data/                       主业务视频、模拟和验证素材
  result/                     会话结果、图片、日志及历史验证
  dist/                       历史发布目录与 ZIP
  archive/                    整理前备份和旧资料
```

`.vscode/` 是本地编辑器配置，`.git/` 保存原版本历史。
`build/` 和 `__pycache__/` 是自动生成目录，不属于源码。
原构建资料已归档，后续构建会重新生成 `build/`。

## 3. 闸机源码怎么读

| 文件 | 主要对象 | 用途 |
| --- | --- | --- |
| `gate/core.py` | `FaceEngine`、`YoloPoseDetector`、`PersonDetector` | 人脸、人体与头肩关键点检测 |
| `gate/core.py` | `Track`、`SimpleTracker` | 跨帧轨迹与接近状态 |
| `gate/core.py` | `candidate_score`、`PersonRegistry` | 照片质量评分和同一旅客候选合并 |
| `gate/core.py` | `save_person`、`save_csv` | 图片和人员汇总保存 |
| `gate/processor.py` | `ProcessorThread` | 视频读取、旋转、抓拍筛选、保存与消息发布 |
| `gate/mqtt_publisher.py` | `MqttPublisher` | 稳定旅客编号映射、图片编码与 MQTT 发送 |
| `apps/gate_app.py` | `VideoLabel`、`MainWindow` | 独立闸机界面的预览、区域编辑和参数 |

先读 `ProcessorThread._run` 和 `_process_video`。
再按调用顺序阅读检测、轨迹、抓拍评分和消息生成。
闸机的头肩关键点用于抓拍裁剪，属于闸机识别逻辑。

联合工作台通过 `integration/workers.py` 的 `GateWorker` 调用处理线程。
本机直连时注入内部消息接口，不另建闸机 MQTT 连接。

## 4. 重识别源码怎么读

| 文件 | 主要对象 | 用途 |
| --- | --- | --- |
| `reid/core.py` | `PersonDetector` | YOLO 行人检测与 ByteTrack 跟踪 |
| `reid/core.py` | `ReidExtractor`、`FastReidSBS` | FastReID SBS R50 与 2048 维归一化特征 |
| `reid/matcher.py` | `TrackMatcher` | 外观得分、轨迹历史、队首筛选、特征缓存和旅客锁定 |
| `reid/matcher.py` | `confirm_frame` | 汇总所有区域候选后确认唯一匹配 |
| `reid/runtime.py` | `set_phase_threads` | 检测与特征提取的 CPU 线程设置 |
| `reid/search_worker.py` | `SearchWorker` | 独立界面的文件和实时流检索 |
| `reid/messaging.py` | MQTT 类和图片解码函数 | 独立界面通信及旅客图片解码 |
| `apps/reid_app.py` | `MainWindow`、`VideoPlayer` | 独立界面的旅客库、区域、阈值与预览 |

联合重识别调度位于 `integration/workers.py` 的 `RecognitionWorker`。
独立界面使用 `SearchWorker`。两者复用同一个匹配器。

建议阅读顺序：

1. `RecognitionWorker._drain_gallery`：按完整旅客编号建立特征库。
2. `RecognitionWorker._process`：裁剪各供框位区域，调用匹配器。
3. `TrackMatcher.annotate`：计算候选与跨帧得分，判断轨迹成熟状态。
4. `TrackMatcher.confirm_frame`：各区域一起确认匹配，登记旅客锁定。

原有规则继续保留：

- 每个供框位一个区域，启用后才处理。
- 同一轨迹每 10 个处理帧刷新外观特征。
- 新增旅客图片时保留轨迹、队首和已确认身份。
- 首次完成匹配后，旅客锁定到原区域中的原轨迹实例。
- 所有供框位共享旅客锁定记录。
- 原目标消失、轨迹过期或编号复用，都不会解锁旅客。
- 多个目标同时符合条件时暂不确认，无轨迹编号时等待稳定轨迹。
- 新建会话重新建立任务与锁定记录。

锁定记录保存在内存中。重启程序或新建会话会重新开始。
本次没有新增跨重启保存锁定记录的功能。

## 5. 绑定源码怎么读

| 文件 | 主要对象 | 用途 |
| --- | --- | --- |
| `binding/passenger_tray_binding.py` | `PassengerTrayBinder` | 绑定算法本体 |
| 同上 | `ingest` | 按接收时间处理完整位置快照与 RFID |
| 同上 | `advance`、`_finalize` | 处理等待到期的 RFID 任务 |
| 同上 | `_decide` | 位置新鲜度、停留时长和候选唯一性判断 |
| `binding/binding_runtime.py` | `LiveInbox`、`normalize_message` | 接收时间、消息顺序与字段映射 |
| 同上 | `MqttLink`、`HttpInput`、`BindingSession` | 独立界面的 MQTT、HTTP 和会话 |
| `binding/generate_binding_simulation.py` | 模拟生成函数 | 离线消息及独立评估标签 |
| `binding/replay_binding.py` | `replay`、`evaluate` | 回放及结果评估 |
| `apps/binding_app.py` | `BindingWindow` | 独立绑定界面 |
| `apps/binding_simulation_dialog.py` | 对话框和生成线程 | 编辑模拟场景并生成数据 |

位置消息需要完整描述当前区域，空列表也有意义。
算法用相邻检测都看到同一旅客的时间段累计停留。
位置过期或没有合格旅客时不绑定。
只有一个合格旅客时绑定，多人合格时保留为待确认。

## 6. 调度、界面和公共代码

| 文件 | 功能 |
| --- | --- |
| `integration/configuration.py` | 默认配置、校验、路径、备份和原子保存 |
| `integration/controller.py` | 会话启动停止、消息处理、界面更新和日志 |
| `integration/workers.py` | 闸机、重识别和首帧预览的后台任务 |
| `integration/widgets.py` | 视频画面与区域坐标换算 |
| `integration/diagnostics.py` | 真实模型加载与推理检查 |
| `apps/integration_app.py` | 日常联合界面及命令参数 |
| `video/stream_capture.py` | 最新帧取流、时间戳、中断和中文路径文件读取 |
| `common/project_paths.py` | 模型、配置、数据、结果和发布资源的路径 |
| `__main__.py` | 联合、闸机、重识别、绑定、模拟、回放六个命令 |

## 7. 配置和资源

| 位置 | 用途 |
| --- | --- |
| `configs/integration_config.json` | 联合来源、区域、重识别、绑定及消息参数 |
| `configs/gate_config.json` | 独立闸机配置及联合默认抓拍参数 |
| `configs/reid_roi_config.json` | 独立重识别界面的区域 |
| `configs/binding_config.json` | 独立绑定和回放参数 |
| `configs/binding_simulation_config.json` | 模拟场景 |
| `models/gate/` | YOLO pose、YuNet、SFace、Haar |
| `models/reid/yolov8s.pt` | 供框台行人检测 |
| `models/reid/msmt_sbs_R50.pth` | 旅客外观特征 |
| `data/verification/reid_sample.mp4` | 重识别本地验证素材 |
| `result/integration/<时间_编号>/` | 每次联合会话的配置、图片、消息、判定及日志 |

联合配置只改了两项模型路径。原来源、区域和业务参数保留。
独立闸机仍把结果写入所选视频目录的“人像提取结果”。
发布包模型来自 `_internal/models/`，可写配置位于 EXE 旁的 `configs/`。

## 8. 启动与验证

沿用 Conda `mmpose`，Python 3.10.20。本次未升级依赖。

```powershell
conda run --no-capture-output -n mmpose python D:\code\mediapipe-master\pose\start_integration.py
```

可编辑安装后，使用 `python -m pose_project` 的六个命令：
`integrate`、`gate`、`reid`、`binding`、`simulate`、`replay`。
完整示例见根目录 `README.md`。

在主项目根目录验证：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
conda run --no-capture-output -n mmpose python -m unittest discover -s tests
conda run --no-capture-output -n mmpose python scripts/verify_project.py --output result/new_project_check
conda run --no-capture-output -n mmpose python scripts/verify_integration.py --output result/new_integration_check
```

`verify_project.py` 检查当前主项目的模块、入口、模型和绑定回放。
`verify_integration.py` 用两路本地视频验证后台任务与联合界面。
`verify_integration_latency.py` 单独测量本机 RTSP 与识别延迟。
构建及 EXE 验证脚本仍位于 `scripts/`，发布配置位于 `packaging/`。

## 9. 备份和清理

备份目录：`archive/20261110_structure/`。

| 文件或目录 | 内容 |
| --- | --- |
| `source_before.zip`、`backup_manifest.json` | 第一次整理前的 143 个文件及 SHA-256 |
| `git_diff_before.patch`、`git_status_before.txt` | 整理前未提交修改 |
| `legacy/` | 旧闸机、重识别版本、旧临时测试、截图和重复资源 |
| `previous_source_backups/` | 原 `source_backup/` |
| `previous_build/` | 原构建目录 |
| `config_backups/` | 原配置备份 |
| `migration_map.json` | 第一阶段资源迁移位置 |
| `before_pose_separation.zip` | 姿态分离前的 136 个源码、配置和数据文本文件 |
| `pose_separation_backup_manifest.json` | 姿态分离前文件校验清单 |
| `pose_separation_map.json` | 迁入独立姿态项目的文件及校验值 |
| `organize_sources.py`、`separate_pose.py`、`verify_separation.py` | 本次机械迁移与核对记录，不参与运行 |

原闸机抓拍和诊断分别迁入 `result/gate_legacy_capture/`、
`result/gate_legacy_diagnostics/`。
姿态专用源码、模型、标注、训练数据、测试、C++ 演示和明确属于姿态的结果，
已经迁入独立项目。混合历史验证报告继续保留，供追溯。

旧日期目录不再参与加载。无调用的重识别辅助函数和旧动态加载代码已移除。
恢复备份时应先解压到独立目录核对，再选择恢复，避免覆盖后续修改。

## 10. 验证记录和范围

本次记录集中在 `result/project_cleanup_20261110/`。

| 检查 | 分离后的实际结果 | 记录 |
| --- | --- | --- |
| 主项目自动化测试 | 131 项全部通过 | `main_unit_tests.log` |
| 主项目模块 | 33 个模块解析和导入成功 | `main_final/verification.json` |
| 六个主命令 | 从无关目录调用帮助全部成功 | `main_final/help_*.log` |
| 绑定模拟回放 | 9 个案例均符合预设标签 | `main_final/replay/evaluation.json` |
| 本地真实模型 | 闸机人体、人脸模型可用；重识别输出 2048 维特征，归一化长度为 1 | `main_final/verification.json` |
| 两路本地视频 | 闸机预览 9 帧，重识别预览 7 帧，输出 4 条有效位置消息 | `integration_final/verification.json` |
| 分离核对 | 两份备份共 279 项通过校验；迁入独立项目的 76 个文件均已核对 | `separation_audit.json` |
| 源码保持 | 主项目 30 个模块除说明文字外与分离前一致；其余三个仅修改包说明、命令和资源路径 | `separation_audit.json` |
| 中文说明 | 主项目 33 个、独立姿态项目 21 个源码模块均有中文职责说明，关键判断另有中文注释 | `separation_audit.json` |

本机重识别使用 RTX 5060 Laptop GPU 和 CUDA 12.8。
闸机仍使用原 OpenCV DNN 处理方式，不据此声称闸机已经使用 GPU。

整理前 181 项测试通过。第一次迁移后 185 项测试通过。
第一阶段核对确认：闸机、匹配器、独立检索、取流等 14 个主要定义的算法内容一致。
中文说明覆盖当时的 48 个源码模块；最终两个项目共 54 个模块。
业务配置除两项模型路径外保持一致。

两次本地视频检查中，两路预览和重识别位置消息正常。
验证通过手动注入一张旅客参考图建立人物库。
短片段没有触发闸机自动抓拍，注入的 RFID 没有形成绑定。
绑定正确性另由独立模拟回放验证，不能把这次视频检查称为自动抓拍闭环。

姿态项目单独通过 56 项测试、六个命令、实际视频特征提取、训练诊断、
40 帧离线视频和界面播放控制验证。
完整记录在 `D:\code\pose_action\result\separation_20261110/`，
详见独立项目的 `docs/PROJECT_HANDOFF.md`。

现场旅客准确率、真实 RFID、生产消息服务器和目标服务器仍需现场验证。
本次没有重新构建 EXE，`dist/` 历史发布包仍对应原来的版本。
