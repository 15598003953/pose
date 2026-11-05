"""
mqtt_publisher.py
================================================================
将 person_extractor.py 采集到的最优人员照片（人脸 + 上半身），通过 MQTT
推送给项目组的「人员重识别模块」。

发送内容（MQTT JSON 消息）：
  - person_id   ：采集端本地人员ID（如 P001）
  - passenger_id：稳定旅客ID = UUID4 前 N 位 + 采集日期(YYYY-MM-DD)
  - capture_time：采集时间戳（采集端解析出的真实时间，
                  如 2026-09-23 18:04:26.600）
  - timestamp   ：发送时刻（ISO8601，毫秒）
  - face        ：人脸图（JPEG base64，可直接解码为图片）
  - upper       ：上半身图（头+上半身，JPEG base64，可选）
  - source_video / score / has_face 等元信息

三种用法（都不需要改动 person_extractor.py）：
  1) 实时监视（推荐，配合正在运行的提取程序）：
         python mqtt_publisher.py <人像提取结果目录> --watch
     person_extractor 每保存/覆盖一张照片，就立即把该人员的人脸+ID+时间发出去；
     启动时目录里已有的照片不会重发，只发之后新增的。
  2) 批量重发（提取完成后一次性发送目录里全部记录）：
         python mqtt_publisher.py <人像提取结果目录>
  3) 实时接口：采集端保存成功后直接调用
         from mqtt_publisher import MqttPublisher
         publisher = MqttPublisher(out_dir=out_dir)
         publisher.publish_person(local_id=p['id'], capture_time=p['ts_str'],
                                 face_img=p.get('face'), upper_img=p.get('upper'),
                                 score=p['score'], has_face=p.get('face') is not None)
         publisher.close()

MqttPublisher 维护 本地ID(P001) -> 旅客ID 的稳定映射，跨“更新”与多次运行
保持同一旅客ID，并持久化到 out_dir/mqtt_id_map.json（幂等）。

依赖： paho-mqtt  (pip install paho-mqtt)
连接配置： 见下方常量；账号密码走环境变量 MQTT_USERNAME / MQTT_PASSWORD。
"""

import os
import csv
import time
import json
import base64
import uuid
import cv2
from datetime import datetime

# ---------------- MQTT 连接配置（按实际环境修改） ----------------
BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "192.168.1.60")
BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
TOPIC = os.environ.get("MQTT_TOPIC", "check_gate/person/collected")
CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "check_gate_publisher")
KEEPALIVE = 60
QOS = 1

# 对外人员ID 中“随机不重复字段”的长度（UUID4 的 hex 前 N 位）。
# 32 位 = 完整 UUID4 hex（128 bit）：碰撞概率极低，作为对外唯一标识。
# 需要更短 ID 时可下调（如 12），但 8 位以下在单日大客流时有碰撞风险。
ID_RANDOM_LEN = 32

# paho 缺失 / broker 不可达时是否静默降级（不阻断主提取流程）
FAIL_SOFT = True


def _b64_jpg(path):
    """读取图片文件，返回其 JPEG 字节的 base64 字符串；不存在返回 None。"""
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return base64.b64encode(data).decode("ascii")


def _b64_jpg_img(img_bgr):
    """把 BGR(numpy) 图像编码为 JPEG 的 base64 字符串；无效返回 None。"""
    if img_bgr is None or not hasattr(img_bgr, "size") or img_bgr.size == 0:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", img_bgr)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


class MqttPublisher:
    def __init__(self, out_dir=None, enabled=True, host=None, port=None, topic=None):
        self.enabled = enabled
        self.out_dir = out_dir
        self._host = host or BROKER_HOST
        self._port = port or BROKER_PORT
        self._topic = topic or TOPIC
        self._client = None
        self._id_map = {}                 # 本地ID -> 旅客ID
        self._map_path = None
        if out_dir:
            self._map_path = os.path.join(out_dir, "mqtt_id_map.json")
            self._load_map()
        if self.enabled:
            self._connect()

    # ---- ID 映射（稳定 + 幂等） ----
    def _load_map(self):
        if self._map_path and os.path.exists(self._map_path):
            try:
                with open(self._map_path, encoding="utf-8") as f:
                    self._id_map = json.load(f)
            except Exception:
                self._id_map = {}

    def _save_map(self):
        if self._map_path:
            try:
                with open(self._map_path, "w", encoding="utf-8") as f:
                    json.dump(self._id_map, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def passenger_id(self, local_id, date_str):
        """为本地ID(P001)分配稳定旅客ID：<uuid前N位>-<YYYY-MM-DD>。

        若缓存里是旧版本生成的 ID（随机段长度与当前 ID_RANDOM_LEN 不符，
        或不是当天日期），自动重新生成并覆盖，避免继续发送旧格式。
        """
        if local_id in self._id_map:
            exist = self._id_map[local_id]
            # ID 形如 <hex>-YYYY-MM-DD，日期含两个 '-'，故 rsplit 取随机段
            old_rand = exist.rsplit("-", 3)[0] if "-" in exist else exist
            if len(old_rand) == ID_RANDOM_LEN and exist.endswith(date_str):
                return exist
            # 长度不符（旧的 8 位）或跨天：落到下面重新生成并覆盖
        rand = uuid.uuid4().hex[:ID_RANDOM_LEN]
        pid = "%s-%s" % (rand, date_str)
        self._id_map[local_id] = pid
        self._save_map()
        return pid

    # ---- 连接 ----
    def _connect(self):
        try:
            from paho.mqtt.client import Client, CallbackAPIVersion
        except Exception as e:
            if FAIL_SOFT:
                self.enabled = False
                print("[mqtt] paho-mqtt 未安装，已禁用 MQTT 推送：", e)
                return
            raise
        try:
            self._client = Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=CLIENT_ID,
            )
            user = os.environ.get("MQTT_USERNAME")
            pw = os.environ.get("MQTT_PASSWORD")
            if user:
                self._client.username_pw_set(user, pw)
            self._client.connect(self._host, self._port, keepalive=KEEPALIVE)
            self._client.loop_start()
        except Exception as e:
            if FAIL_SOFT:
                self.enabled = False
                print("[mqtt] 连接 broker 失败，已禁用 MQTT 推送：", e)
            else:
                raise

    # ---- 发布 ----
    def build_payload(self, local_id, capture_time,
                       face_img=None, face_path=None,
                       upper_img=None, upper_path=None,
                       source_video="", score=0.0, has_face=None):
        """构造一条人员采集记录，不连接服务器。

        三要素一定包含：
          - 人脸图像（JPEG base64）：优先用内存图像 face_img(BGR numpy)，
            否则读 face_path；二者皆无则这条不推人脸。
          - 人员ID：local_id（如 P001），并附稳定旅客ID passenger_id。
          - 时间戳：采集时刻 capture_time（采集端解析出的真实时间）。
        另含发送时刻 timestamp(ISO8601)、来源视频、质量分、上半身图。
        """
        # 采集日期：用图片采集时刻的日期（同人同日同ID）
        try:
            dt = datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            try:
                dt = datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        pid = self.passenger_id(local_id, date_str)

        face_b64 = (_b64_jpg_img(face_img) if face_img is not None
                    else _b64_jpg(face_path))
        upper_b64 = (_b64_jpg_img(upper_img) if upper_img is not None
                     else _b64_jpg(upper_path))
        if has_face is None:
            has_face = bool(face_b64)

        payload = {
            # 对外发送的人员ID：<32位随机hex>-<YYYY-MM-DD>
            # 例：a6deca95bec249a4b5b0124ec6348ff6-2026-10-22
            "person_id": pid,
            "local_id": local_id,      # 内部编号（P001），便于回溯排查
            "passenger_id": pid,       # 稳定旅客ID，与 person_id 同值
            "capture_time": capture_time,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "source_video": source_video,
            "score": round(float(score), 2),
            "has_face": bool(has_face),
            "images": {
                "face": face_b64,
                "upper": upper_b64,
            },
        }
        return payload

    def publish_person(self, local_id, capture_time,
                       face_img=None, face_path=None,
                       upper_img=None, upper_path=None,
                       source_video="", score=0.0, has_face=None):
        """Build and send one collected-person message."""
        if not self.enabled or self._client is None:
            return False
        payload = self.build_payload(
            local_id=local_id, capture_time=capture_time,
            face_img=face_img, face_path=face_path,
            upper_img=upper_img, upper_path=upper_path,
            source_video=source_video, score=score, has_face=has_face)
        try:
            self._client.publish(
                self._topic,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                qos=QOS, retain=False)
            return True
        except Exception as e:
            if FAIL_SOFT:
                print("[mqtt] 发布失败：", e)
                return False
            raise

    # ---- 批量：从 人员汇总.csv 发送 ----
    def publish_from_output_dir(self, out_dir):
        """读取 out_dir/人员汇总.csv 及对应图片，把每条采集记录发到 MQTT。
        返回成功发送条数；broker 不可达（FAIL_SOFT 禁用）时返回 0。"""
        csv_path = os.path.join(out_dir, "人员汇总.csv")
        if not os.path.exists(csv_path):
            print("[mqtt] 找不到汇总文件：", csv_path)
            return 0
        sent = 0
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row or row[0] == "人员ID":
                    continue
                person_id = row[0]
                capture_time = row[1]
                try:
                    score = float(row[2])
                except Exception:
                    score = 0.0
                has_face = (len(row) > 3 and row[3] == "是")
                source_video = row[4] if len(row) > 4 else ""
                upper_path = row[5] if len(row) > 5 else ""
                face_path = row[6] if len(row) > 6 else ""
                if self.publish_person(
                        local_id=person_id, capture_time=capture_time,
                        face_path=face_path or None,
                        upper_path=upper_path or None,
                        source_video=source_video, score=score,
                        has_face=has_face):
                    sent += 1
        print("[mqtt] 已发送 %d 条记录" % sent)
        return sent

    # ---- 实时：监视输出目录，新增照片立即发送 ----
    def _current_bases(self, out_dir):
        """扫描目录，返回当前已有照片的“基名”集合（不含 _上半身/_脸部/_全景 后缀）。"""
        bases = set()
        try:
            names = os.listdir(out_dir)
        except OSError:
            return bases
        for n in names:
            if not n.lower().endswith(".jpg"):
                continue
            for suf in ("_上半身.jpg", "_脸部.jpg", "_全景.jpg"):
                if n.endswith(suf):
                    bases.add(n[:-len(suf)])
                    break
        return bases

    def _publish_base(self, out_dir, base):
        """根据一组文件名（不含后缀）解析出人员ID/采集时间并发送。"""
        parts = base.split("_")
        try:
            d, t, ms = parts[1], parts[2], parts[3]   # YYYYmmdd / HHMMSS / mmm
            dt = datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]),
                          int(t[0:2]), int(t[2:4]), int(t[4:6]),
                          int(ms) * 1000)
            capture_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except Exception:
            capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pid_local = parts[0]
        face_path = os.path.join(out_dir, base + "_脸部.jpg")
        upper_path = os.path.join(out_dir, base + "_上半身.jpg")
        has_face = os.path.exists(face_path)
        ok = self.publish_person(
            local_id=pid_local, capture_time=capture_time,
            face_path=face_path if has_face else None,
            upper_path=upper_path if os.path.exists(upper_path) else None,
            has_face=has_face)
        if ok:
            print("[mqtt] 发送 %s %s" % (pid_local, capture_time))
        return ok

    def watch(self, out_dir, interval=1.0, stop_event=None):
        """实时监视 out_dir：person_extractor 每保存/覆盖一张照片，
        就立刻把该人员的人脸 + 人员ID + 采集时间戳发到 MQTT。
        启动时目录里已有的照片不会重发，只发之后新增的。按 Ctrl+C 停止。"""
        seen = self._current_bases(out_dir)   # 启动前已有的不重发
        print("[mqtt] 开始实时监视目录：%s（按 Ctrl+C 停止）" % out_dir)
        try:
            while stop_event is None or not stop_event.is_set():
                bases = self._current_bases(out_dir)
                for b in sorted(bases - seen):
                    seen.add(b)
                    self._publish_base(out_dir, b)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("[mqtt] 已停止监视")

    def close(self):
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="把 人像提取结果 目录里的采集照片通过 MQTT 发送")
    ap.add_argument("out_dir", help="人像提取结果目录（含照片与 人员汇总.csv）")
    ap.add_argument("--host", default=BROKER_HOST, help="MQTT broker 地址")
    ap.add_argument("--port", type=int, default=BROKER_PORT, help="MQTT broker 端口")
    ap.add_argument("--topic", default=TOPIC, help="MQTT 主题")
    ap.add_argument("--watch", action="store_true",
                    help="实时监视目录：新增照片立即发送（配合正在运行的提取程序）")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="--watch 时的轮询间隔（秒）")
    args = ap.parse_args()
    pub = MqttPublisher(out_dir=args.out_dir, host=args.host,
                        port=args.port, topic=args.topic)
    if not pub.enabled:
        print("[mqtt] 推送未启用（缺少 paho-mqtt 或连接失败），未发送任何数据。")
        return
    if args.watch:
        pub.watch(args.out_dir, interval=args.interval)
    else:
        pub.publish_from_output_dir(args.out_dir)
    pub.close()


if __name__ == "__main__":
    _main()
