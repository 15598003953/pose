"""本地文件兼容读取、网络流最新帧与 FFmpeg 打开参数隔离。"""

import os
import re
import threading
import time
import uuid

import cv2


STREAM_SCHEME_RE = re.compile(r"^(rtsp|rtsps|rtmp|rtmps|http|https|udp|tcp)://", re.I)

# FFmpeg 打开 RTSP 的参数（OpenCV 通过该环境变量透传）：
# - rtsp_transport=tcp：默认 UDP 一旦丢包，H.264 参考帧缺失就会整片花屏，
#   强制 TCP（interleaved）可彻底避免丢包花屏，这是花屏问题的根因；
# - buffer_size / max_delay / reorder_queue_size：加大接收缓冲并允许按序重排；
# - stimeout / timeout：socket 超时（微秒），避免读阻塞把线程卡死。
#   旧版 FFmpeg 只认 stimeout，新版改名为 timeout，两个都写，
#   不认识的那个会被忽略（不影响打开）。
# - flags=discardcorrupt：解码器层面直接丢弃损坏帧（不输出带马赛克的画面，
#   本程序只取最新帧，丢一帧不影响观感，比显示花屏好）。
# - fflags=discardcorrupt：封装层面的同类开关，两者同时开启更保险。
FFMPEG_RTSP_OPTS = (
    "rtsp_transport;{transport}|buffer_size;1048576|max_delay;500000|"
    "reorder_queue_size;0|timeout;5000000|stimeout;5000000|"
    "fflags;discardcorrupt|flags;discardcorrupt"
)


def is_stream_source(src):
    """判断来源是网络流（RTSP/HTTP…）还是本地文件。"""
    return bool(STREAM_SCHEME_RE.match((src or "").strip()))


def probe_real_fps(cap, allow_seek=True):
    """返回视频的真实 FPS。

    优先用元数据 FPS；若与“总帧数/实际时长”推算值偏差超过 10%
    （如可变帧率或元数据错误的视频），则采用推算值，
    保证检测节奏与实际播放时间同步。

    allow_seek=False 时不定位到尾帧校验（TS/M2TS 这类无索引容器 seek
    代价高且常失败，直接用元数据 FPS，缺失时按 25 处理）。
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    real = None
    if allow_seek and total > 1:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ok, _ = cap.read()
            dur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ok and dur > 0:
                real = total / dur
        except Exception:
            real = None
    if real and (not fps or fps <= 0 or abs(real - fps) > 0.1 * max(real, fps)):
        return real
    return fps if fps and fps > 0 else 25.0


def open_file_capture(path):
    """打开本地视频文件（TS/MP4 等），返回 (cap, tmp_path)。

    Windows 下 OpenCV 对非 ASCII（中文）路径支持不佳：直接打开失败时，
    在源文件同目录建一个 ASCII 文件名的硬链接（同盘零拷贝）再打开，
    tmp_path 供调用方结束后删除，避免复制大文件。
    """
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            return cap, None
        cap.release()

    if all(ord(ch) < 128 for ch in path):
        return cv2.VideoCapture(path), None

    tmp = None
    try:
        folder, name = os.path.split(path)
        ext = os.path.splitext(name)[1]
        # 文件名唯一，避免同一文件被同时打开多次时互相覆盖/删除
        tmp = os.path.join(folder or ".",
                           f".reid_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}{ext}")
        if os.path.exists(tmp):
            os.remove(tmp)
        try:
            os.link(path, tmp)  # NTFS 硬链接，不占额外空间、瞬间完成
        except OSError:
            import shutil
            shutil.copy2(path, tmp)
        for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
            cap = cv2.VideoCapture(tmp, backend)
            if cap.isOpened():
                return cap, tmp
            cap.release()
    except Exception:
        pass
    if tmp and os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return cv2.VideoCapture(path), None


# ----------------------------- RTSP 抓帧线程 -----------------------------
class StreamGrabber:
    """RTSP 后台抓帧器：独立线程持续 grab，只保留最新一帧。

    为什么需要它（花屏 + 延迟增长的根因）：
    1) OpenCV 默认用 UDP 传输 RTSP，网络抖动导致丢包后，H.264 的参考帧不完整，
       解码出来就是花屏/绿块 —— 这里强制 TCP 传输。
    2) 主循环要做检测 + 特征提取，通常慢于视频帧率；直接在主循环里 cap.read()
       会让解码缓冲持续堆积，既越播越延迟，也更容易因缓冲溢出丢包而花屏 ——
       这里让抓帧线程始终把缓冲“抽空”，主循环永远只消费最新一帧。
    """

    def __init__(self, source, log=None, clock=None):
        self.source = source
        self._log = log or (lambda m: None)
        self._clock = clock or (lambda: time.time_ns() // 1000000)
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0          # 累计抓到的帧数（也作为帧序号）
        self._captured_at_ms = 0
        self._stopped = False
        self._thread = None
        self.fps = 25.0
        self.transport = "tcp"
        self.reconnects = 0
        self.failed = False    # 彻底连不上（重连次数超限）
        self.size = (0, 0)

    # ----- 对外接口 -----
    def start(self):
        self._thread = threading.Thread(target=self._loop, name="rtsp-grabber",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def interrupt(self):
        """只请求停止、不 join（供 UI 线程调用，立即打断阻塞中的 wait_frame）。"""
        self._stopped = True
        with self._cond:
            self._cond.notify_all()

    def wait_frame(self, last_seq=-1, timeout=5.0, with_timestamp=False):
        """等待一帧序号大于 last_seq 的新帧；超时/已停止返回 (None, last_seq)。

        last_seq 传 -1 表示“首帧”：一直等到抓到第一帧为止（此时 seq 从 0 开始，
        不能只比较序号，否则会立刻返回 None）。
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._frame is not None and self._seq > last_seq:
                    if with_timestamp:
                        return self._frame, self._seq, self._captured_at_ms
                    return self._frame, self._seq
                if self._stopped:
                    return (None, last_seq, 0) if with_timestamp else (None, last_seq)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return (None, last_seq, 0) if with_timestamp else (None, last_seq)
                self._cond.wait(remaining)

    @property
    def grabbed(self):
        return self._seq

    # ----- 抓帧线程 -----
    def _create_capture(self, transport):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000]
        # 两路摄像头可能同时打开，锁内设置各自参数，防止传输方式互相覆盖。
        with _FFMPEG_OPEN_LOCK:
            if self._stopped:
                return None
            try:
                _enable_ffmpeg_rtsp_opts(transport)
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
            finally:
                _restore_ffmpeg_env()
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _loop(self):
        transport = "tcp"
        first_attempt = True
        while not self._stopped:
            try:
                cap = self._create_capture(transport)
            except Exception as exc:
                self._log("视频流连接异常：" + str(exc))
                cap = None
            if self._stopped:
                if cap is not None:
                    cap.release()
                return
            if cap is None and first_attempt and transport == "tcp":
                # 少数设备/中间件不支持 RTSP over TCP，回退 UDP 再试一次
                self._log("RTSP TCP 方式打开失败，回退 UDP 重试…")
                transport = "udp"
                first_attempt = False
                continue
            first_attempt = False
            if cap is None:
                self._log(f"无法打开视频流：{self.source}")
                self.failed = True
                with self._cond:
                    self._cond.notify_all()
                return

            self.transport = transport
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.size = (w, h)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and 1 <= fps <= 120:
                self.fps = float(fps)
            kind = "RTSP" if (self.source or "").lower().startswith("rtsp") else "视频流"
            self._log(f"{kind} 已连接（{transport.upper()} 传输，{w}x{h} @ "
                      f"{self.fps:.1f}fps）")

            fails = 0
            while not self._stopped:
                if not cap.grab():
                    fails += 1
                    if fails > 50:  # ≈2.5s 完全读不到数据 -> 认为断流，重连
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                ok, frame = cap.retrieve()
                if not ok or frame is None or frame.size == 0:
                    continue
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue
                with self._cond:
                    self._frame = frame
                    self._captured_at_ms = self._clock()
                    self._seq += 1
                    self._cond.notify_all()

            cap.release()
            if self._stopped:
                break
            self.reconnects += 1
            if self.reconnects > 20:
                self._log("RTSP 连续重连失败，停止抓帧。")
                self.failed = True
                with self._cond:
                    self._cond.notify_all()
                return
            self._log(f"RTSP 连接中断，2 秒后第 {self.reconnects} 次重连…")
            for _ in range(20):
                if self._stopped:
                    return
                time.sleep(0.1)


_FFMPEG_ENV_KEY = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_FFMPEG_ENV_OLD = None
_FFMPEG_OPEN_LOCK = threading.Lock()


def _enable_ffmpeg_rtsp_opts(transport):
    """设置 FFmpeg 抓流参数（全局环境变量，OpenCV 创建 capture 时读取）。"""
    global _FFMPEG_ENV_OLD
    if _FFMPEG_ENV_OLD is None:
        _FFMPEG_ENV_OLD = os.environ.get(_FFMPEG_ENV_KEY, "")
    os.environ[_FFMPEG_ENV_KEY] = FFMPEG_RTSP_OPTS.format(transport=transport)


def _restore_ffmpeg_env():
    """恢复原有环境变量，避免影响后续打开本地文件。"""
    global _FFMPEG_ENV_OLD
    if _FFMPEG_ENV_OLD is None:
        return
    if _FFMPEG_ENV_OLD:
        os.environ[_FFMPEG_ENV_KEY] = _FFMPEG_ENV_OLD
    else:
        os.environ.pop(_FFMPEG_ENV_KEY, None)
    _FFMPEG_ENV_OLD = None
