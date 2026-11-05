# 人包绑定模拟数据与离线算法

生成器提供模拟日志，绑定算法按接收顺序回放消息，再与独立答案比较。
生成器和离线回放脚本不连接真实 MQTT，也不修改现有视频应用。
输入只包含检测框、稳定旅客 ID 和 RFID 消息，不包含动作识别结果。
需要界面、MQTT 或 HTTP 接入时，运行 `binding_app.py`，详见 `BINDING_GUI.md`。

## 运行

首次运行先按项目根目录的 `README.md` 安装本地源码。

在 `D:\code\mediapipe-master` 下执行：

```powershell
conda run -n mmpose python -m pose_project simulate
```

默认输出到 `pose\data\binding_simulation`。
已有非空目录会拒绝写入，保留原文件。
需要重新生成时，可以指定另一个输出目录：

```powershell
conda run -n mmpose python -m pose_project simulate --output .\pose\data\binding_simulation_run2
```

## 场景和参数

修改 `configs/binding_simulation_config.json`，或用 `--config` 指定其他配置文件。
相同配置和随机种子生成相同数据。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `update_interval_ms` | 200 | 每个 ROI 的人员快照间隔 |
| `duration_ms` | 360000 | 模拟六分钟 |
| `binding_parameters.lookback_ms` | 10000 | 保存已讨论的回看范围，生成器不据此计算答案 |
| `binding_parameters.min_dwell_ms` | 5000 | 保存已讨论的停留门槛，生成器不据此计算答案 |
| `detection_delay_ms` | 20～450 | 模拟坐标消息从采集到接收的延迟范围 |
| `photoelectric_delay_ms` | 5～40 | 模拟光电触发信号到达上位系统的延迟范围 |
| `bbox_jitter_px` | 6 | 检测框中心的随机像素变化范围 |

以上延迟是人为设置的测试条件，并非现场测量结果。
绑定参数仅随数据保存，生成器不使用这些参数决定正确答案。
所有时间段使用毫秒，采用左闭右开的区间。
`start_time` 必须附带时区。

默认包含：

- A、B 两个供框台，每台一次仅有一个框。每次推框对应一个独立 RFID。
- 上半身检测框和稳定 ID，ID 格式为 `A-随机字段-YYYY-MM-DD`。
- 一名旅客连续使用多个框。
- 一名旅客离开操作位置三秒，返回后继续用框。
- 新旅客接替后，上一名仍停留在较大的 A 区内。
- 操作区外长期停留的陪同者，以及短暂穿过操作区的行人。
- 第一框分别等待一分钟、五分钟。
- A、B 同时推框，以及检测消息延迟、到达乱序。

`roi_xyxy` 和 `operation_roi_xyxy` 均为 `[左, 上, 右, 下]`。
人员框使用相同坐标格式；本例用框中心描述上半身位置。
各轨迹段支持 `center` 到 `end_center` 的匀速移动；
省略 `end_center` 时，中心固定，仅加入像素变化。
本例的操作区和像素坐标是虚构数据，不代表真实斜拍画面的标定结果。

## 输出

| 文件 | 用途 |
| --- | --- |
| `detections.jsonl` | 每个 ROI 的人员快照，默认 3600 条 |
| `rfid.jsonl` | RFID 接收记录，默认 9 条 |
| `replay.jsonl` | 两路消息按接收时间合并，默认 3609 条 |
| `expected_bindings.jsonl` | 独立编写的预期绑定，默认 9 条 |
| `manifest.json` | 数据标记、完整配置和记录数量 |

每一行是一个 JSON 对象。`topic` 与 `payload` 表示可供以后回放的 MQTT 消息，
`received_at_ms` 是日志记录的上位系统接收时间，不是读码器提供的字段。
人员消息另含 `captured_at_ms`，表示产生这组坐标的画面采集时间。
即使出现乱序，采集时间仍保持原值。
每个 ROI 每次发送完整人员列表，无人时发送空列表。

RFID 的 `payload` 只有 `roi_id` 和 `rfid`，不包含硬件时间戳或旅客答案。
预期绑定中的 `simulated_push_at_ms` 是模拟器已知的触发时刻，
只供评估延迟，实际接口没有这个字段。

绑定程序仅把 `replay.jsonl` 作为消息输入。
预期绑定只在判定完成后用于比较，场景轨迹不进入绑定算法。
生成器的默认场景暂未加入多人同时符合操作条件的情况。
算法测试单独检查此时封存结果，不选择其中一人。

## 离线绑定

在 `D:\code\mediapipe-master` 下执行：

```powershell
conda run -n mmpose python -m pose_project replay --expected .\pose\data\binding_simulation\expected_bindings.jsonl
```

默认结果目录是 `pose\result\binding_replay`。
程序拒绝写入已有非空目录。再次回放时，指定新的结果目录：

```powershell
conda run -n mmpose python -m pose_project replay --expected .\pose\data\binding_simulation\expected_bindings.jsonl --output .\pose\result\binding_replay_run2
```

`--input` 可指定其他接收日志，`--config` 可指定其他算法配置。
省略 `--expected` 时，只生成绑定结果，不比较正确答案。
日志必须按 `received_at_ms` 升序排列，允许同一毫秒收到多条消息。
程序不会自动排序，以免掩盖接收顺序错误。
有预期答案时，全部匹配返回退出码 0；存在差异返回 1；输入或写入失败返回 2。

`passenger_tray_binding.py` 负责判定，`replay_binding.py` 负责回放和评估。
算法单独使用 `configs/binding_config.json`，不读取生成器配置中的轨迹或推框安排。
两份配置中的区域和消息主题必须与实际输入相符。
调整算法参数时，修改 `configs/binding_config.json`。

| 算法配置 `binding_parameters` | 默认值 | 含义 |
| --- | --- | --- |
| `lookback_ms` | 10000 | 每次推框向前查看十秒 |
| `min_dwell_ms` | 5000 | 最近十秒内的有效停留至少五秒 |
| `binding_wait_ms` | 600 | 收到 RFID 后等待坐标消息的时长 |
| `max_sample_gap_ms` | 600 | 两次采集间隔超过此值时，整段间隔不累计 |
| `max_position_age_ms` | 600 | 参考时刻与最近一次采集的最大间隔 |
| `rfid_time_offset_ms` | 0 | 从 RFID 接收时间中减去的延迟补偿 |

600 毫秒是本次实验起点，相当于三个坐标更新周期。
模拟消息的最大延迟是 450 毫秒，600 毫秒等待可以覆盖本例。
现场仍需测量延迟、丢帧和时钟误差，再调整这些参数。
补偿默认设为零，不把模拟器已知的触发时刻当成硬件提供的数据。

每次判定依次检查：

1. 使用 RFID 接收时间减去补偿，得到判定参考时刻。
2. 等待结束后，只采用期限内收到的快照。采集时间还须不晚于参考时刻。
3. 用上半身框中心判断是否进入操作区。较大供框台区域用于限定操作区。
4. 按采集时间排序。同一旅客须在相邻两个快照中都位于操作区，才能累计两者间隔。
5. 把间隔裁剪到最近十秒。离开操作区、空快照和长时间数据中断会停止累计。
6. 旅客在最近快照中仍须位于操作区。快照采集时间须不晚于参考时刻，且尚未过期。
7. 只有一名合格旅客时才绑定。每次绑定后保留有效历史，支持同一旅客连续用框。

操作区的左、上边界计入区域，右、下边界不计入。
同一采集时间的重复快照不会增加停留时间；操作区人员名单冲突时，程序报错。
最后一次快照之后的时间不累计，因此停留估计偏保守。
短暂离开后，同一 ID 可以继续使用十秒范围内仍有效的历史。
等待不会向后移动参考时刻，也不会把推框之后到达操作区的人作为当次候选。

输出文件：

| 文件 | 用途 |
| --- | --- |
| `bindings.jsonl` | 仅包含成功绑定的供框台区域、旅客 ID 和 RFID，可供上位系统读取 |
| `decisions.jsonl` | 每次推框的结果、参考时间、快照新鲜程度及候选人的停留依据 |
| `evaluation.json` | 与预期答案逐条比较；仅指定 `--expected` 时生成 |
| `run.json` | 输入路径、消息数量和本次实际算法配置 |

每次成功绑定的返回值位于 `decision["binding"]`。
`bindings.jsonl` 每行保存一条相同的返回值，格式如下：

```json
{
  "roi_id": "A",
  "passenger_id": "0123456789abcdef0123456789abcdef-2026-10-20",
  "rfid": "RFID-A-0001"
}
```

`roi_id` 是供框台区域，采用大写字母。
`passenger_id` 由已有的随机唯一字段和首次出现日期组成。
输出时去掉旧检测 ID 中的区域前缀，不重新生成随机字段或日期。
同一旅客使用多个框时，旅客 ID 保持不变，每个框返回自身 RFID。
已采用“随机字段-YYYY-MM-DD”的输入 ID 会原样保留。
判定日志中的旅客 ID 保留原值，便于追查对应检测消息。
未绑定或多人封存时，`decision["binding"]` 为 `null`，不写入 `bindings.jsonl`。

结果中的 `status` 含义：

- `bound`：唯一候选满足条件，写入旅客 ID。
- `unbound`：数据缺失、坐标过期或没有合格候选，旅客 ID 为空。
- `deferred`：多人同时合格，保存候选和依据，旅客 ID 为空。后续处理暂未实现。

系统不会提示人工确认，也不会在多名合格候选中任选一人。
`decided_at_ms` 是离线回放采用的等待期限，不是脚本实际执行的电脑时间。

## 当前验证结果

2026-10-20 在 `mmpose` 环境完成回放：

- 输入 3609 条消息，生成 9 条绑定，9 条全部匹配独立预期答案。
- 错误绑定、未绑定、封存、缺少结果和额外结果均为 0。
- 数据生成和绑定算法共 37 项测试通过。

这只证明当前模拟场景通过，不能据此推算现场准确率。
桌面程序已支持实时 MQTT 和 HTTP，见 `BINDING_GUI.md`。
桌面程序的“数据接口 → 本地文件回放”也提供“生成模拟数据”入口。
可选择模板、输出目录并调整随机种子、起始时间、采集间隔、坐标扰动和传输延迟。
生成成功后自动选择新日志，点击“开始”即可回放。
真实设备时钟同步和现场网络仍需联调。
当前算法假定稳定 ID 和完整区域快照，也假定采集时间与接收时间可直接比较。
采集时间晚于接收时间时，程序会报错。
操作区需要使用真实画面标定，尤其要检查斜拍时的上半身框中心是否可靠。
旅客在操作区停留只是一项位置证据，仍不能直接证明谁向框中放了物品。

## 验证

在 `D:\code\mediapipe-master\pose` 下执行：

```powershell
conda run -n mmpose python -m unittest discover -s tests -p "test_*.py" -v
```
