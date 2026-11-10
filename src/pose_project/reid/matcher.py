"""轨迹特征匹配、队首筛选和旅客会话锁定；界面不参与判定。"""

from collections import Counter, deque
import time

import cv2
import numpy as np

from pose_project.reid.runtime import DET_THREADS, EXT_THREADS, set_phase_threads
from pose_project.reid.core import crop_from_frame, valid_crop


FRONT_MODES = ("bottom", "top", "left", "right", "area")
def draw_text(frame, text, org, color=(0, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class TrackMatcher:
    """逐帧检测 + 特征比对 + 跨帧轨迹 top-k 投票（文件 / 实时流共用）。

    人物库按标签分组（同一人物的多个角度/表情照片共用一个标签），
    匹配时对同一标签的所有图片取最大相似度（多镜头匹配），
    可显著提升对姿态、视角、表情变化的鲁棒性。
    """
    TRACK_HIST = 20    # 每条轨迹保留最近 20 帧的相似度历史
    TRACK_TOPK = 5     # 轨迹得分取历史中最高的 5 帧平均（抗遮挡/姿态突变）
    MIN_FRAMES = 3     # 轨迹至少出现 3 帧才显示匹配标签（抑制偶发误检）
    TRACK_TTL = 60     # 轨迹连续 60 帧未出现则清除
    FEAT_REFRESH = 10  # 同一轨迹每 10 帧刷新一次外观特征
    LEADER_MISS_LIMIT = 8
    LEADER_MARGIN = 0.08
    IDENTITY_MARGIN = 0.05

    def __init__(self, persons, detector, extractor, threshold,
                 display_labels=None, front_mode=None, passenger_locks=None):
        self.detector = detector
        self.extractor = extractor
        self.threshold = threshold
        self.display_labels = display_labels or {}
        self.front_mode = front_mode if front_mode in FRONT_MODES else None
        self._leader_tid = None
        self._leader_miss = 0
        self.last_queue_count = 0
        self.last_skipped = 0
        self.passenger_locks = passenger_locks if passenger_locks is not None else {}

        self.tracks = {}
        self._feat_cache = {}
        self.idx = 0
        self.matched = 0
        self.last_detections = []
        self.last_candidates = []
        self.last_timing = {}
        self.update_gallery(persons, display_labels)

    def update_gallery(self, persons, display_labels=None):
        locked_labels = {
            tid: self.label_names[state["label_index"]]
            for tid, state in self.tracks.items()
            if state.get("label_index") is not None
        }
        label_names, label_feats = [], []
        for p in persons:
            lab = p["label"].strip() or "未命名"
            if lab in label_names:
                label_feats[label_names.index(lab)].append(p["feature"])
            else:
                label_names.append(lab)
                label_feats.append([p["feature"]])
        self.label_names = label_names
        self.gallery_all = np.stack([f for fs in label_feats for f in fs])  # (F, D)
        label_cols, start = [], 0
        for fs in label_feats:
            label_cols.append(list(range(start, start + len(fs))))
            start += len(fs)
        self.label_cols = label_cols
        self.n_labels = len(label_names)
        self.display_labels = display_labels or {}
        # 图库列可能变化，只清理分数历史，保留轨迹、队首和已确认身份。
        label_indices = {label: index for index, label in enumerate(label_names)}
        for tid, state in self.tracks.items():
            state["hist"].clear()
            state["votes"].clear()
            state["label_index"] = label_indices.get(locked_labels.get(tid))
        self.last_detections = []
        self.last_candidates = []

    def reset_leader(self):
        self._leader_tid = None
        self._leader_miss = 0

    def _front_score(self, box):
        x1, y1, x2, y2 = box
        if self.front_mode == "bottom":
            return y2
        if self.front_mode == "top":
            return -y1
        if self.front_mode == "left":
            return -x1
        if self.front_mode == "right":
            return x2
        return (x2 - x1) * (y2 - y1)

    def _leader_better(self, candidate, current, shape):
        if self.front_mode == "area":
            return candidate >= current * 1.25
        height, width = shape[:2]
        margin = (self.LEADER_MARGIN * height
                  if self.front_mode in ("bottom", "top")
                  else self.LEADER_MARGIN * width)
        return candidate - current >= margin

    def _select_leader(self, boxes, valid_indices, tids, shape):
        def keep_locked(indices):
            locked = [
                index for index, tid in enumerate(tids)
                if (self.tracks.get(tid) or {}).get("label_index") is not None
            ]
            return sorted(set(indices) | set(locked))

        if not valid_indices:
            if self._leader_tid is not None:
                self._leader_miss += 1
                if self._leader_miss >= self.LEADER_MISS_LIMIT:
                    self._leader_tid = None
                    self._leader_miss = 0
            return []

        scores = [self._front_score(boxes[valid_indices[j]])
                  for j in range(len(valid_indices))]
        best = int(np.argmax(scores))
        if self._leader_tid is None:
            self._leader_tid = tids[best]
            self._leader_miss = 0
            return keep_locked([best])

        try:
            current = tids.index(self._leader_tid)
        except ValueError:
            current = None
        if current is None:
            self._leader_miss += 1
            if self._leader_miss >= self.LEADER_MISS_LIMIT:
                self._leader_tid = tids[best]
                self._leader_miss = 0
                return keep_locked([best])
            return keep_locked([])

        self._leader_miss = 0
        if best != current and self._leader_better(scores[best], scores[current], shape):
            self._leader_tid = tids[best]
            return keep_locked([best])
        return keep_locked([current])

    def _extract_feats(self, crops, tids, tta=True):
        if not crops:
            feat_dim = getattr(self.extractor, "feat_dim", self.gallery_all.shape[1])
            return np.zeros((0, feat_dim), dtype=np.float32)
        features = [None] * len(crops)
        pending, pending_indices = [], []
        for index, tid in enumerate(tids):
            cached = self._feat_cache.get(tid) if tid is not None else None
            if (cached is not None and self.FEAT_REFRESH
                    and self.idx - cached["age"] < self.FEAT_REFRESH):
                features[index] = cached["feat"]
            else:
                pending.append(crops[index])
                pending_indices.append(index)
        if pending:
            fresh = self.extractor.extract(pending, tta=tta)
            for fresh_index, crop_index in enumerate(pending_indices):
                features[crop_index] = fresh[fresh_index]
                tid = tids[crop_index]
                if tid is not None:
                    self._feat_cache[tid] = {
                        "feat": fresh[fresh_index], "age": self.idx}
        return np.ascontiguousarray(np.stack(features))

    def annotate(self, frame, canvas=None, offset=(0, 0), imgsz=960, tta=True,
                 defer_confirmation=False):
        """对一帧做检测+匹配，返回标注后的新图（不修改原图）。

        canvas / offset（ROI 模式）：frame 是 ROI 裁剪图，检测在裁剪图上做，
        框与文字画在整帧画布 canvas 上（坐标平移 offset=(x, y)），
        匹配结果写入 self.last_detections（画布坐标系）供 ROI 上报使用。
        """
        started = time.perf_counter()
        annotated = frame.copy() if canvas is None else canvas.copy()
        ox, oy = int(offset[0]), int(offset[1])
        set_phase_threads(DET_THREADS)
        xyxy, confs, track_ids = self.detector.detect_track(frame, imgsz=imgsz)
        detected_at = time.perf_counter()

        # 无人或等待队首的帧也要清理；先清理，避免复用过期身份。
        stale = [tid for tid, state in self.tracks.items()
                 if self.idx - state["seen"] > self.TRACK_TTL]
        for tid in stale:
            del self.tracks[tid]
            self._feat_cache.pop(tid, None)

        crops, valid_idx, valid_tids = [], [], []
        for i, b in enumerate(xyxy):
            c = crop_from_frame(frame, b)
            if valid_crop(c):
                crops.append(c)
                valid_idx.append(i)
                valid_tids.append(track_ids[i] if i < len(track_ids) else None)

        for tid in valid_tids:
            if tid in self.tracks:
                self.tracks[tid]["seen"] = self.idx
        self.last_queue_count = len(crops)
        keep = list(range(len(crops)))
        if self.front_mode:
            keep = self._select_leader(xyxy, valid_idx, valid_tids, frame.shape)
        self.last_skipped = len(crops) - len(keep)
        if len(keep) != len(crops):
            crops = [crops[index] for index in keep]
            valid_idx = [valid_idx[index] for index in keep]
            valid_tids = [valid_tids[index] for index in keep]

        # 每个检测框对每个标签的相似度（同标签多张图取最大），
        # 并写入轨迹历史，用 top-k 平滑后的轨迹得分做最终判定，
        # 避免逐帧匹配抖动与偶发误配（如外观相似的路人）。
        lab_sims = [None] * len(crops)
        extracted_at = detected_at
        unlocked_indices = [
            index for index, tid in enumerate(valid_tids)
            if (self.tracks.get(tid) or {}).get("label_index") is None
        ]
        if unlocked_indices:
            set_phase_threads(EXT_THREADS)
            feats = self._extract_feats(
                [crops[index] for index in unlocked_indices],
                [valid_tids[index] for index in unlocked_indices],
                tta=tta)
            extracted_at = time.perf_counter()
            s = feats @ self.gallery_all.T                    # (N, F)
            unlocked_sims = np.zeros((len(unlocked_indices), self.n_labels))
            for li, cols in enumerate(self.label_cols):
                unlocked_sims[:, li] = s[:, cols].max(axis=1)
            for local_index, crop_index in enumerate(unlocked_indices):
                lab_sims[crop_index] = unlocked_sims[local_index]
                tid = valid_tids[crop_index]
                if tid is None:
                    continue
                st = self.tracks.get(tid)
                if st is None:
                    st = self.tracks[tid] = {
                        "hist": deque(maxlen=self.TRACK_HIST),
                        "seen": self.idx,
                        "label_index": None,
                        "votes": deque(maxlen=self.MIN_FRAMES),
                        "token": object(),
                    }
                st["hist"].append(unlocked_sims[local_index])
                st["votes"].append(int(unlocked_sims[local_index].argmax()))
                st["seen"] = self.idx
        candidates = []
        for j, i in enumerate(valid_idx):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            tid = valid_tids[j]
            st = self.tracks.get(tid) if tid is not None else None
            locked_index = st.get("label_index") if st is not None else None
            if locked_index is not None:
                label_index = locked_index
                sim = float(st.get("confirmed_score", self.threshold))
                second_score = None
                margin = None
                stable = True
                status = "matched"
            elif st is not None and len(st["hist"]) > 0:
                h = np.stack(st["hist"])                      # (T, L)
                k = min(self.TRACK_TOPK, len(h))
                vec = np.sort(h, axis=0)[-k:].mean(axis=0)    # (L,) 轨迹得分
                mature = len(h) >= self.MIN_FRAMES
                ranked_index = int(vec.argmax())
                second_score = (
                    float(np.partition(vec, -2)[-2])
                    if self.n_labels > 1 else None)
                label_index = ranked_index
                sim = float(vec[label_index])
                margin = (sim - second_score
                          if second_score is not None else None)
                votes = list(st.get("votes", ()))
                stable = (
                    len(votes) >= self.MIN_FRAMES
                    and len(set(votes[-self.MIN_FRAMES:])) == 1
                    and votes[-1] == label_index)
                status = ("below_threshold" if sim < self.threshold else
                          "warming_track" if not mature or not stable else
                          "ambiguous_match"
                          if margin is not None and margin < self.IDENTITY_MARGIN
                          else "matched")
            else:
                vec = lab_sims[j] if lab_sims[j] is not None else np.zeros(self.n_labels)
                mature = True
                ranked_index = int(vec.argmax())
                second_score = (
                    float(np.partition(vec, -2)[-2])
                    if self.n_labels > 1 else None)
                label_index = ranked_index
                sim = float(vec[label_index])
                margin = (sim - second_score
                          if second_score is not None else None)
                stable = False
                status = ("below_threshold" if sim < self.threshold else
                          "waiting_track")
            label = self.label_names[label_index]
            owner = self.passenger_locks.get(label)
            if owner is not None and (st is None or owner is not st["token"]):
                status = "passenger_locked"
            candidates.append({"label": label, "score": round(sim, 4),
                               "second_score": (
                                   round(second_score, 4)
                                   if second_score is not None else None),
                               "match_margin": (
                                   round(margin, 4)
                                   if margin is not None else None),
                               "detector_score": round(float(confs[i]), 4),
                               "bbox_xyxy": [x1, y1, x2, y2], "tid": tid,
                               "status": status,
                               "identity_locked": locked_index is not None,
                               "identity_frozen": locked_index is not None,
                               "stable_candidate": stable})
        self.last_candidates = candidates
        self.last_detections = []
        if not defer_confirmation:
            self.confirm_frame([self], annotated)
        if canvas is None:
            draw_text(annotated, f"frame {self.idx}  matched={self.matched}", (10, 25),
                      (255, 255, 255), 0.6, 2)
        self.idx += 1
        self.last_timing = {
            "detection_ms": round((detected_at - started) * 1000, 2),
            "reid_ms": round((extracted_at - detected_at) * 1000, 2),
            "matching_ms": round((time.perf_counter() - extracted_at) * 1000, 2)}
        return annotated

    @staticmethod
    def confirm_frame(matchers, canvas):
        """整帧所有区域完成候选计算后，确认唯一对象并锁定旅客。"""
        counts = Counter(
            c["label"] for m in matchers for c in m.last_candidates
            if c["status"] == "matched" and not c.get("identity_locked"))
        for matcher in matchers:
            matcher.last_detections = []
            for candidate in matcher.last_candidates:
                if candidate.get("identity_locked"):
                    state = matcher.tracks.get(candidate["tid"])
                    if state is None:
                        continue
                    matcher.last_detections.append({
                        key: candidate[key]
                        for key in ("label", "score", "bbox_xyxy", "tid")})
                    x1, y1, x2, y2 = candidate["bbox_xyxy"]
                    color = (0, 255, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
                    shown = matcher.display_labels.get(
                        candidate["label"], candidate["label"])
                    draw_text(canvas, f"{shown} {candidate['score']:.2f}",
                              (x1, max(0, y1 - 12)), color, 0.8, 2)
                    continue
                if candidate["status"] != "matched":
                    continue
                label, tid = candidate["label"], candidate["tid"]
                state = matcher.tracks[tid]
                owner = matcher.passenger_locks.get(label)
                if owner is not None and owner is not state["token"]:
                    candidate["status"] = "passenger_locked"
                    continue
                if counts[label] > 1:
                    candidate["status"] = "duplicate"
                    continue
                # 会话级记录独立于轨迹缓存，轨迹过期或编号复用都不会解锁。
                matcher.passenger_locks[label] = state["token"]
                state["label_index"] = matcher.label_names.index(label)
                state["confirmed_score"] = candidate["score"]
                candidate["identity_locked"] = True
                candidate["identity_frozen"] = True
                matcher.last_detections.append({
                    key: candidate[key] for key in ("label", "score", "bbox_xyxy", "tid")})
                x1, y1, x2, y2 = candidate["bbox_xyxy"]
                color = (0, 255, 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
                shown = matcher.display_labels.get(label, label)
                draw_text(canvas, f"{shown} {candidate['score']:.2f}",
                          (x1, max(0, y1 - 12)), color, 0.8, 2)
                matcher.matched += 1
        # 本帧另一区域刚完成锁定时，未成熟的候选也不能覆盖旅客证据。
        for matcher in matchers:
            for candidate in matcher.last_candidates:
                owner = matcher.passenger_locks.get(candidate["label"])
                state = matcher.tracks.get(candidate["tid"])
                if owner is not None and (state is None or owner is not state["token"]):
                    candidate["status"] = "passenger_locked"
