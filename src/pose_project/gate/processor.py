"""闸机视频处理线程：采集、筛选、保存和消息发布。"""

import os
import time
import traceback
from datetime import datetime, timedelta

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from pose_project.gate.core import (
    FRESH_MISSED, HAS_MP, KP_LSHOULDER, KP_NOSE, KP_RSHOULDER,
    PersonDetector, PersonRegistry, SimpleTracker, candidate_score,
    face_landmark_overlay, iou, parse_start_time, pose_upper, save_csv, save_person,
)
from pose_project.gate.mqtt_publisher import MqttPublisher


class ProcessorThread(QThread):
    """后台批量处理线程：读视频 → 旋转 → 检测 → 跟踪 →
    只抓拍 ROI 内最前面（目标框最大）且正朝镜头靠近
    （人脸逐渐放大）的人 → 打分 → 保存最优。"""
    sig_frame = pyqtSignal(object, object)      # bgr 帧, 检测列表
    sig_person = pyqtSignal(object)             # 新增/更新的人员信息
    sig_progress = pyqtSignal(float, str)       # 总进度 0~1, 当前视频名
    sig_pos = pyqtSignal(int, int, float)       # 当前帧号, 总帧数, fps
    sig_log = pyqtSignal(str)
    sig_finished = pyqtSignal()

    def __init__(self, files, roi, opts, out_dir, start_frames=None,
                 publisher=None, mqtt_enabled=True, stream_factory=None):
        super().__init__()
        self.files = list(files)
        self.roi = dict(roi)
        self.opts = dict(opts)
        self.out_dir = out_dir
        self.stream_factory = stream_factory
        # 各视频的起始帧（界面进度条指定的快进起点）
        self.start_frames = list(start_frames) if start_frames \
            else [0] * len(self.files)
        self._stop = False
        self._seek_target = -1   # 处理中拖动进度条 → 快进到该帧
        self.reg = None
        self._frame_h = 0
        self._diag_last = {}
        self._no_face_hinted = False
        # MQTT 实时推送：broker 不可达会自动禁用，不影响主提取流程
        self.mqtt = publisher
        if self.mqtt is None and mqtt_enabled and MqttPublisher is not None:
            try:
                self.mqtt = MqttPublisher(out_dir=self.out_dir)
            except Exception as e:
                self.sig_log.emit("MQTT 初始化失败（将不推送）：%s" % e)

    def stop(self):
        self._stop = True

    def request_seek(self, frame):
        """处理中把当前视频快进到指定帧（线程安全：仅写一个 int）。"""
        self._seek_target = max(0, int(frame))

    # ---------- 工具 ----------

    def _roi_rect(self, shape):
        """把 ROI 归一化矩形映射为像素矩形 (x, y, w, h)。"""
        H, W = shape[:2]
        r = self.roi
        return (r['x'] * W, r['y'] * H, r['w'] * W, r['h'] * H)

    def _in_roi(self, box, rect):
        """人员框中心是否落在 ROI 矩形内。"""
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        rx, ry, rw, rh = rect
        return rx <= cx <= rx + rw and ry <= cy <= ry + rh

    def _make_candidate(self, frame, track, ts, video_name,
                        detector=None):
        """构造"头 + 上半身"整体人像候选。

        以检测到的人脸为锚点统一裁剪（头顶上方留头发、下方裁到
        腰线、两侧含双肩），保证不出现单独的手部/腰部等局部；
        检测不到人脸、人脸太小、人像被画面裁掉太多或框内没有
        对应运动能量时返回 None（不抓拍）。"""
        H, W = frame.shape[:2]
        min_sharp = float(self.opts.get('min_sharp', 30) or 0)
        prefer_static = bool(self.opts.get('prefer_static', True))
        face = track.face
        require_face = self.opts.get('require_face', True)
        min_face_w = self.opts.get('min_face_w', 30)
        if face is not None and face[2] < min_face_w:
            return None                    # 人脸太小：人还远，不拍
        if face is None:
            if require_face:
                return None                # 必须看到脸，否则不算完整人像
            # 降级模式：按运动/人体框上部裁剪（不保证含头部）
            x, y, w, h = track.box
            ux0 = max(0, int(x - 0.05 * w))
            uy0 = max(0, int(y - 0.03 * h))
            ux1 = min(W, int(x + 1.05 * w))
            uy1 = min(H, int(y + 0.68 * h))
            upper_box = (ux0, uy0, ux1 - ux0, uy1 - uy0)
        else:
            fx, fy, fw, fh = face[:4]
            # 人贴镜头导致脸被画面边缘切掉时，无法得到完整人脸，
            # 这一帧不拍（其余帧会更完整）。判定从严：脸框必须
            # 完整落在画面内，且五官关键点（若有）也不得越界，
            # 否则保存的会是"半张脸"。
            if (fx < 0 or fy < 0
                    or fx + fw > W or fy + fh > H):
                return None
            lm5 = face[4] if len(face) > 4 else None
            if lm5 and any(px < 0 or py < 0 or px >= W or py >= H
                           for px, py in lm5):
                return None
            cx = fx + fw / 2.0
            ux0 = int(cx - 1.25 * fw)       # 两侧含双肩（约 2.5 倍脸宽）
            ux1 = int(cx + 1.25 * fw)
            uy0 = int(fy - 0.55 * fh)       # 头顶（含头发）
            uy1 = int(fy + fh + 3.4 * fh)   # 上半身至腰线
            # 头部姿态锚点（修正裁剪宽度/高度，保证完整上半身入镜）：
            # 首选 YOLO 姿态关键点（鼻尖 + 双肩），MediaPipe 为备用。
            # 鼻尖必须落在本脸框内，避免用错旁边人的关键点。
            anchors = []
            kps = getattr(track, 'kps', None)
            if kps is not None:

                def _kp(i):
                    kx, ky, kc = kps[i]
                    return (kx, ky) if kc > 0.3 else None

                anchors.append((_kp(KP_NOSE), _kp(KP_LSHOULDER),
                                _kp(KP_RSHOULDER)))
            if HAS_MP:
                pinfo = pose_upper(frame)
                if pinfo:
                    anchors.append((pinfo['nose'], pinfo['shoulder_l'],
                                    pinfo['shoulder_r']))
            for nose, shl, shr in anchors:
                if nose is None:
                    continue
                if not ((fx - 0.5 * fw) <= nose[0] <= (fx + 1.5 * fw)
                        and (fy - 0.5 * fh) <= nose[1]
                        <= (fy + 1.5 * fh)):
                    continue
                m = 0.30 * fh
                if shl:
                    ux0 = min(ux0, int(shl[0] - m))
                if shr:
                    ux1 = max(ux1, int(shr[0] + m))
                break
            # YOLO 人体框约束上下界：人很近、框只到腰部/画面边时，
            # 不再向下裁出下半身。上界钳制不允许切到脸——人体框顶
            # 偏低（低头/头部被遮挡导致关键点框从颈部开始）时，
            # 放弃钳制，保住头顶
            pbx, pby, pbw, pbh = track.box
            uy0 = max(uy0, min(int(pby - 0.10 * fh), int(fy - 0.20 * fh)))
            if pby + pbh < uy1:
                uy1 = int(pby + pbh + 0.10 * fh)
            vx0, vy0 = max(0, ux0), max(0, uy0)
            vx1, vy1 = min(W, ux1), min(H, uy1)
            if vx1 - vx0 < 16 or vy1 - vy0 < 16:
                return None
            # 裁剪框被画面边界/人体框钳制后不得切到脸，
            # 确保保存的上半身照里脸是完整的
            if (vx0 > fx or vy0 > fy
                    or vx1 < fx + fw or vy1 < fy + fh):
                return None
            exp = max(1, (ux1 - ux0) * (uy1 - uy0))
            vis = (vx1 - vx0) * (vy1 - vy0)
            if vis < 0.72 * exp:
                return None                 # 人被画面裁掉太多，不完整
            # 运动能量应主要来自这个人（排除只框到局部肢体的情形）
            bx, by, bw, bh = track.box
            ix0, iy0 = max(vx0, bx), max(vy0, by)
            ix1, iy1 = min(vx1, bx + bw), min(vy1, by + bh)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter < 0.25 * vis:
                return None
            upper_box = (vx0, vy0, vx1 - vx0, vy1 - vy0)
        ux0, uy0, uw, uh = upper_box
        if uw < 16 or uh < 16:
            return None
        upper = frame[uy0:uy0 + uh, ux0:ux0 + uw].copy()
        # 清晰度（Laplacian 方差）：低于下限直接不抓拍，避免保存模糊图
        _gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(_gray, cv2.CV_64F).var())
        if min_sharp > 0 and sharp < min_sharp:
            return None
        # 一图两脸：裁剪出的上半身照里若还有第二张不重叠的人脸
        # （旁边/身后的旅客），保存出来会带上别人的脸，这张不要
        eng = getattr(detector, 'face_engine', None) \
            if detector is not None else None
        if eng is not None and eng.available:
            kept = []
            for f in eng.find_in_roi(upper, min_neighbors=2):
                if all(iou(f[:4], k[:4]) < 0.3 for k in kept):
                    kept.append(f)
            if len(kept) >= 2:
                return None
        # 静止优先：用人员框高度近期变化率近似运动强度，越静止加分越多
        static_bonus = 0.0
        if prefer_static:
            hs = [hh[1] for hh in list(track.size_hist)[-8:]]
            if len(hs) >= 4:
                m = max(1, len(hs) // 3)
                early = float(np.mean(hs[:m]))
                late = float(np.mean(hs[-m:]))
                if early > 1.0:
                    motion = abs(late - early) / early
                    static_bonus = 1.0 * max(0.0, 1.0 - motion / 0.10)
        face_img = None
        face_lm5 = None      # 脸部图坐标系下的五官关键点（描线图用）
        face_roi = None
        face_box = None      # 脸框在脸部裁剪图坐标系（SFace 特征用）
        if face is not None:
            fx, fy, fw, fh = face[:4]
            fx0 = max(0, int(fx - 0.35 * fw))
            fy0 = max(0, int(fy - 0.35 * fh))
            fx1 = min(W, int(fx + 1.35 * fw))
            fy1 = min(H, int(fy + 1.35 * fh))
            if fx1 - fx0 >= 16 and fy1 - fy0 >= 16:
                face_img = frame[fy0:fy1, fx0:fx1].copy()
                lm5 = face[4] if len(face) > 4 else None
                if lm5:
                    face_lm5 = [(px - fx0, py - fy0) for px, py in lm5]
                    face_roi = (fx0, fy0)
                    # 人脸框在裁剪图坐标系内（SFace 生成特征用）
                    face_box = (fx - fx0, fy - fy0, fw, fh)
        roi_rect = self._roi_rect(frame.shape)
        # 脸部裁剪图单独测清晰度：整图 sharp 会被衣服纹理抬高，
        # 脸糊不糊只有看脸本身才知道；低于 min_sharp 直接放弃
        face_sharp = None
        if face_img is not None:
            _fgray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_sharp = float(cv2.Laplacian(_fgray, cv2.CV_64F).var())
            if min_sharp > 0 and face_sharp < min_sharp:
                return None
        score = candidate_score(upper, upper_box, face, roi_rect,
                                frame.shape, static_bonus, face_sharp)
        full = frame.copy()
        cv2.rectangle(full, (ux0, uy0), (ux0 + uw, uy0 + uh),
                      (0, 220, 0), 2)
        if face is not None:
            fx, fy, fw, fh = face[:4]
            cv2.rectangle(full, (fx, fy), (fx + fw, fy + fh),
                          (255, 180, 0), 2)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(full, ts_str, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return {'upper': upper, 'face': face_img, 'full': full,
                'score': score, 'sharp': sharp, 'ts': ts, 'ts_str': ts_str,
                'video': video_name, 'face_lm5': face_lm5,
                'face_roi': face_roi, 'face_box': face_box,
                'face_sharp': face_sharp}

    def _face_overlay(self, p):
        """为保存的人脸图生成五官描线图。
        有 MediaPipe：468 点网格 + 五官轮廓（本身也是二次校验——
        画不出网格的"人脸"多半是误检）。
        无 MediaPipe：用检测时的五官关键点（双眼/鼻尖/双嘴角）
        描点连线，同样能核对五官是否齐全。返回 None 表示无法生成。"""
        img = p.get('face')
        if img is None:
            return None
        if HAS_MP:
            ov = face_landmark_overlay(img)
            if ov is not None:
                return ov
        lm5 = p.get('face_lm5')
        if not lm5:
            return None
        ann = img.copy()
        pts = [(int(round(px)), int(round(py))) for px, py in lm5]
        # 双眼连线 / 双嘴角连线，五个关键点画圈标记
        cv2.line(ann, pts[0], pts[1], (255, 200, 0), 2)
        cv2.line(ann, pts[3], pts[4], (255, 255, 0), 2)
        for (px, py) in pts:
            cv2.circle(ann, (px, py), 6, (0, 255, 255), 2)
            cv2.circle(ann, (px, py), 2, (0, 0, 255), -1)
        return ann

    def _passed_gate(self, t):
        """是否实际进入安检闸机（轨迹定稿提交时把关）。

        唯一判定条件——闸机口虚线：以"距画面底部的百分比"表示
        （界面画面上会画出这条橙色虚线，可拖动调整，默认 10%）。
        闸机口在画面中位置固定，真正进入闸机的人，脚部（人体框
        底部）必须到达过该线以下；折返回去的旅客、以及在闸机旁
        维持秩序却没进去的安检员，脚部都到不了这里。
        设为 0 表示不启用该判定。"""
        gate_bottom = float(self.opts.get('gate_line_bottom_pct', 0) or 0)
        if gate_bottom <= 0:
            return True
        fh = getattr(self, '_frame_h', 0) or 0
        if fh <= 0:
            return True
        line_y = (100.0 - gate_bottom) / 100.0 * fh
        return t.max_depth >= line_y

    def _submit(self, track):
        if track.best is None or self.reg is None:
            return
        # 未实际进入闸机（折返旅客、驻留安检员等）一律不入库
        if not self._passed_gate(track):
            return
        p, changed, old_paths = self.reg.submit(track.best)
        if changed:
            for op in old_paths:      # 更优照片覆盖时删除旧文件
                if op and os.path.exists(op):
                    try:
                        os.remove(op)
                    except OSError:
                        pass
            save_person(self.out_dir, p, face_lm=self._face_overlay(p))
            # 零延迟实时推送：保存那一刻立即把人脸 + 人员ID + 时间戳发出去
            if self.mqtt is not None:
                try:
                    self.mqtt.publish_person(
                        local_id=p['id'], capture_time=p['ts_str'],
                        face_img=p.get('face'),        # 内存 numpy 图，免落盘再读
                        upper_img=p.get('upper'),
                        source_video=p.get('video', ''),
                        score=p['score'],
                        has_face=p.get('face') is not None)
                except Exception as e:
                    self.sig_log.emit("MQTT 发布失败：%s" % e)
            thumb_src = p.get('face')
            if thumb_src is None:
                thumb_src = p['upper']
            th, tw = thumb_src.shape[:2]
            sc = 96.0 / max(tw, 1)
            thumb = cv2.resize(thumb_src, (96, max(1, int(th * sc))))
            self.sig_person.emit(dict(
                id=p['id'], ts_str=p['ts_str'], score=p['score'],
                face=p.get('face') is not None, thumb=thumb,
                upper_path=p.get('upper_path'),
                face_path=p.get('face_path'),
                face_lm_path=p.get('face_lm_path'),
                full_path=p.get('full_path')))
            self.sig_log.emit("%s 已保存/更新：%s（分数 %.1f）"
                              % (p['id'], p['ts_str'], p['score']))

    # ---------- 主流程 ----------

    def run(self):
        try:
            self._run()
        except Exception:
            self.sig_log.emit("处理异常：\n" + traceback.format_exc())
        finally:
            if self.reg is not None:
                save_csv(self.out_dir, self.reg.persons)
            if getattr(self, 'mqtt', None) is not None:
                try:
                    self.mqtt.close()
                except Exception:
                    pass
            self.sig_finished.emit()

    def _run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        detector = PersonDetector(self.opts)
        for wmsg in detector.warnings:
            self.sig_log.emit("提示：" + wmsg)
        self.reg = PersonRegistry(
            self.opts.get('merge_thr', 0.5),
            time_window_min=float(
                self.opts.get('merge_time_window', 2) or 0))
        if self.opts.get('require_approach', True):
            self.sig_log.emit(
                "抓拍条件：ROI 内站在最前面的人，且人脸宽度增长 ≥ "
                "%.0f%%（朝镜头移动）；判定为接近后持续抓拍，"
                "走到最近处仍会保存最清晰的一张"
                % (self.opts.get('face_growth', 0.08) * 100))
        else:
            self.sig_log.emit(
                "抓拍条件：ROI 内站在最前面的人（未启用接近判定）")
        starts = ([datetime.now() for _ in self.files] if self.stream_factory
                  else [parse_start_time(f) for f in self.files])
        # 手动时间覆盖：以第一个视频的偏移量平移所有片段
        ov = (self.opts.get('time_override') or "").strip()
        if ov and self.stream_factory:
            self.sig_log.emit("实时流使用收到画面的时间，忽略文件时间覆盖")
        elif ov:
            try:
                base = datetime.strptime(ov, "%Y%m%dT%H%M%S")
                delta = base - starts[0]
                starts = [s + delta for s in starts]
                self.sig_log.emit("时间已手动覆盖为 %s 起" % ov)
            except ValueError:
                self.sig_log.emit("时间覆盖格式无效（应为 20260923T180406），"
                                  "已改用文件名解析")
        n_files = len(self.files)
        for i, path in enumerate(self.files):
            if self._stop:
                break
            try:
                self._process_video(
                    i, path, starts[i], detector,
                    self.start_frames[i] if i < len(self.start_frames)
                    else 0)
            except Exception:
                if self.stream_factory:
                    raise
                # 单个视频出错不再中断整批任务：记录完整堆栈后继续下一个
                self.sig_log.emit(
                    "处理 %s 时出错（已跳过该视频，继续后面的）：\n%s"
                    % (os.path.basename(path), traceback.format_exc()))
        # 收尾：把还在跟踪中的轨迹定稿
        if self._stop and self.reg is not None:
            save_csv(self.out_dir, self.reg.persons)
        self.sig_log.emit("处理结束，共登记 %d 人。输出目录：%s"
                          % (len(self.reg.persons), self.out_dir))

    def _recover_face(self, detector, frame, t):
        """上一轮检到过脸、这一轮丢失时，按人员框变化推算人脸位置，
        在预测点周围降低阈值重新锁定。真实画面里人脸检测是断续的，
        这一步能大幅提高人脸命中率与"接近"判定的样本连续性。"""
        eng = getattr(detector, 'face_engine', None)
        if eng is None or not eng.available:
            return None
        if t.last_face is None or t.last_box is None:
            return None
        H, W = frame.shape[:2]
        obx, oby, obw, obh = t.last_box
        nbx, nby, nbw, nbh = t.box
        if obw <= 0 or obh <= 0 or nbw <= 0 or nbh <= 0:
            return None
        lfx, lfy, lfw, lfh = t.last_face[:4]
        rx = (lfx + lfw / 2.0 - obx) / float(obw)     # 脸心相对位置
        ry = (lfy + lfh / 2.0 - oby) / float(obh)
        rw = lfw / float(obw)
        rh = lfh / float(obh)
        cx = nbx + rx * nbw
        cy = nby + ry * nbh
        fw = rw * nbw
        fh = rh * nbh
        if fw < 8 or fh < 8:
            return None
        x0 = max(0, int(cx - 1.15 * fw))
        x1 = min(W, int(cx + 1.15 * fw))
        y0 = max(0, int(cy - 1.15 * fh))
        y1 = min(H, int(cy + 1.15 * fh))
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        # 同 find_face：YuNet 忽略 min_neighbors，只跑一次
        tries = (2, 1) if eng.mode == 'haar' else (2,)
        for nb in tries:
            faces = eng.find_in_roi(roi, min_neighbors=nb)
            for det in faces:
                fx, fy, fw2, fh2 = det[:4]
                if fw2 < 0.55 * fw or fw2 > 1.9 * fw:
                    continue               # 尺寸应接近预测值
                lms = det[4] if len(det) > 4 else None
                return (int(fx) + x0, int(fy) + y0, int(fw2), int(fh2),
                        [(px + x0, py + y0) for px, py in lms]
                        if lms else None)
        return None

    def _diag(self, t, name, ts, growth_thr):
        """限频输出"未接近"原因，便于调参（每个轨迹约 3 秒一次）。"""
        key = (name, t.tid)
        now = time.monotonic()
        if now - self._diag_last.get(key, -9.9) < 3.0:
            return
        self._diag_last[key] = now
        nf = len(t.face_hist)
        nb = len(t.size_hist)
        fw = t.face_hist[-1][1] if nf else 0
        bh0 = t.size_hist[0][1] if nb else 0
        bh1 = t.size_hist[-1][1] if nb else 0
        self.sig_log.emit(
            "%s 未判定接近：人脸样本=%d 当前脸宽=%dpx 增长率=%.0f%%"
            "（需≥%.0f%%）；人体框高 %d→%dpx%s"
            % (ts.strftime("%H:%M:%S"), nf, fw, t.growth * 100,
               growth_thr * 100, bh0, bh1,
               "（无人脸时按人体框判定，阈值×1.5）" if nf < 4 else ""))
        if nf == 0 and nb >= 12 and not self._no_face_hinted:
            self._no_face_hinted = True
            self.sig_log.emit(
                "提示：一直检不到人脸（人脸样本=0），可能是拍摄角度/"
                "光线/分辨率导致人脸检测器漏检。可尝试：① 取消勾选"
                "「必须检测到人脸」改为按人体框抓拍；② 缩小「最小"
                "有效脸宽」；③ 改用 YuNet 模型（把 onnx 放到脚本目录）")

    def _process_video(self, idx, path, start, detector, start_frame=0):
        cap = (self.stream_factory(path) if self.stream_factory
               else cv2.VideoCapture(path))
        try:
            self._process_capture(idx, path, start, detector, cap, start_frame)
        finally:
            cap.release()

    def _process_capture(self, idx, path, start, detector, cap, start_frame=0):
        name = os.path.basename(path)
        if self.stream_factory:
            name = "闸机实时流"
        if not cap.isOpened():
            self.sig_log.emit("无法打开视频：%s（跳过）" % name)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 1:
            fps = 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.sig_log.emit("[%d/%d] %s  起始时间 %s  fps=%.1f"
                          % (idx + 1, len(self.files), name,
                             start.strftime("%Y-%m-%d %H:%M:%S"), fps))
        tracker = SimpleTracker()
        rotate180 = (self.opts.get('rotate', 180) == 180)
        det_iv = max(1, self.opts.get('det_interval', 1))
        fi = 0
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fi = start_frame
            self.sig_log.emit(
                "快进：从第 %d 帧（约 %s）开始处理"
                % (fi, (start + timedelta(seconds=fi / fps))
                   .strftime("%H:%M:%S")))
        last_emit = 0.0
        last_progress = 0.0
        last_frame_at = time.monotonic()
        n_files = len(self.files)
        while not self._stop:
            # 处理中拖动进度条：跳到指定帧继续（时间戳按帧号换算，
            # 不受快进影响）
            if self._seek_target >= 0 and self._seek_target != fi:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_target)
                fi = self._seek_target
            self._seek_target = -1
            ok, frame = cap.read()
            read_wait = time.monotonic() - last_frame_at
            last_frame_at = time.monotonic()
            if read_wait >= 5.0:
                self.sig_log.emit("读取视频帧耗时 %.1f 秒，正在继续…" % read_wait)
            if not ok:
                break
            if rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)  # 顺时针 180°
            if self._frame_h != frame.shape[0]:
                self._frame_h = frame.shape[0]   # 闸机口虚线换算用
            ts = (datetime.fromtimestamp(cap.captured_at_ms / 1000)
                  if self.stream_factory else start + timedelta(seconds=fi / fps))
            do_det = (fi % det_iv == 0)
            if do_det:
                detect_started = time.monotonic()
                dets = detector.detect(frame)
                detect_wait = time.monotonic() - detect_started
                if detect_wait >= 2.0:
                    self.sig_log.emit("第 %d 帧检测耗时 %.1f 秒，正在继续…"
                                      % (fi, detect_wait))
                for t in tracker.update(dets, fi):
                    self._submit(t)
                # 人脸断续丢失时在预测位置重新锁定
                for t in tracker.tracks:
                    if t.face is None and t.missed == 0:
                        f = self._recover_face(detector, frame, t)
                        if f is not None:
                            t.face = f
                            t.observe_face(fi, f, t.box)
            # 抓拍判定：ROI 内所有人中只取"最前面"（目标框最高、最近）的
            # 一个，且必须呈"朝镜头靠近"趋势（人脸逐渐放大）才抓拍
            rect = self._roi_rect(frame.shape)
            in_roi_tracks = [t for t in tracker.tracks
                             if t.missed <= FRESH_MISSED
                             and self._in_roi(t.box, rect)]
            front = None
            if in_roi_tracks:
                front = max(in_roi_tracks,
                            key=lambda t: (t.box[3], t.box[2] * t.box[3]))
            require_approach = self.opts.get('require_approach', True)
            growth_thr = self.opts.get('face_growth', 0.08)
            min_samples = int(self.opts.get('approach_min_frames', 4))
            min_hits = int(self.opts.get('min_hits', 2))
            view = []
            for t in tracker.tracks:
                in_roi = self._in_roi(t.box, rect)
                is_front = (front is not None and t is front)
                appr, growth = t.approaching(growth_thr, min_samples)
                if is_front and not appr and require_approach:
                    # 诊断输出：让用户知道差多少，便于调阈值
                    nf = len(t.face_hist)
                    nb = len(t.size_hist)
                    if nf or nb:
                        self._diag(t, name, ts, growth_thr)
                if t.missed <= FRESH_MISSED:
                    view.append((t.box, t.face, in_roi, t.tid, is_front,
                                 appr, growth))
                # missed<=FRESH_MISSED：框仍新鲜（人员可能静止被背景吸收，
                # 容忍一小段时间后停止抓拍，避免拍空场景）。
                # 不再要求"只抓最前方"：多人紧跟通过时，前方的人
                # 会一直占据最前方，后面的人将完全无法积累候选而
                # 被整条漏掉；未进闸机的人由 _passed_gate 统一过滤。
                ok = (do_det and in_roi and t.hits >= min_hits
                      and t.missed <= FRESH_MISSED)
                if ok and require_approach and not appr:
                    ok = False    # 没有"朝镜头移动"的动作，不拍
                if ok:
                    cand = self._make_candidate(frame, t, ts, name,
                                                detector)
                    if cand is not None:
                        if t.best is None:
                            t.best = cand
                        elif cand['score'] > t.best.get('score', -1):
                            t.best = cand
                        else:
                            # 分数接近时保留"脸更清楚"的那张：
                            # 优先比脸部清晰度，无人脸再比整图
                            cs = cand.get('face_sharp') \
                                or cand.get('sharp', 0)
                            bs = t.best.get('face_sharp') \
                                or t.best.get('sharp', 0)
                            if (cs > bs and cand['score'] >
                                    t.best.get('score', -1) - 0.8):
                                t.best = cand
            # 限频推送界面；低频更新避免 GUI 事件队列积压而显示旧画面。
            now = time.monotonic()
            if now - last_emit > 0.15:
                self.sig_frame.emit(frame, view)
                last_emit = now
            # 不依赖每 20 帧：检测较慢时至少每 0.5 秒反馈一次进度。
            if fi % 20 == 0 or now - last_progress >= 0.5:
                if n_total > 0:
                    prog = (idx + fi / float(n_total)) / n_files
                    status = name
                else:
                    prog = (idx + 0.9) / n_files
                    status = "%s（已处理 %d 帧）" % (name, fi)
                self.sig_progress.emit(prog, status)
                self.sig_pos.emit(fi, n_total, fps)
                last_progress = now
            fi += 1
        # 停止时不再把尚未处理完的轨迹写出为最终结果。
        if not self._stop:
            for t in tracker.finalize_all():
                self._submit(t)
        if not self._stop:
            self.sig_progress.emit((idx + 1) / float(n_files), name)
