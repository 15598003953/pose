"""用旅客位置历史匹配行李框 RFID；只使用检测与收报时间，不读取评估标签。"""

from collections import defaultdict, deque
from copy import deepcopy
import math
import re


def require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string".format(name))
    return value


def binding_passenger_id(passenger_id):
    match = re.fullmatch(
        r"[A-Z]-([0-9a-fA-F]{32}-[0-9]{4}-[0-9]{2}-[0-9]{2})", passenger_id
    )
    return match.group(1) if match else passenger_id


def require_box(value, name):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("{} must be [x1, y1, x2, y2]".format(name))
    if any(
        type(number) not in (int, float) or not math.isfinite(number)
        for number in value
    ):
        raise ValueError("{} must contain finite coordinates".format(name))
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("{} must have positive width and height".format(name))
    return value


def validate_config(config, *, allow_empty_regions=False):
    parameters = config["binding_parameters"]
    for key in ("lookback_ms", "min_dwell_ms", "max_sample_gap_ms", "max_position_age_ms"):
        require_int(parameters[key], key, 1)
    for key in ("binding_wait_ms", "rfid_time_offset_ms"):
        require_int(parameters[key], key)
    if parameters["min_dwell_ms"] > parameters["lookback_ms"]:
        raise ValueError("min_dwell_ms must not exceed lookback_ms")
    topics = config["topics"]
    for key in ("detections", "rfid"):
        require_text(topics[key], "topics.{}".format(key))
    if topics["detections"] == topics["rfid"]:
        raise ValueError("detection and RFID topics must differ")
    if not isinstance(config["regions"], dict) or (
        not config["regions"] and not allow_empty_regions
    ):
        raise ValueError("regions must be a nonempty object")
    for roi_id, region in config["regions"].items():
        if not re.fullmatch("[A-Z]", roi_id):
            raise ValueError("region IDs must be single uppercase letters")
        outer = require_box(region["roi_xyxy"], "roi_xyxy")
        operation = require_box(region["operation_roi_xyxy"], "operation_roi_xyxy")
        if not (
            outer[0] <= operation[0] < operation[2] <= outer[2]
            and outer[1] <= operation[1] < operation[3] <= outer[3]
        ):
            raise ValueError("operation ROI must be inside its station ROI")


class PassengerTrayBinder:
    """按接收时间处理整区快照；advance 用于到期判定，finish 用于结束回放。"""

    def __init__(self, config):
        validate_config(config)
        self.config = deepcopy(config)
        self.parameters = self.config["binding_parameters"]
        self.history = {roi_id: {} for roi_id in self.config["regions"]}
        self.pending = deque()
        self.last_received_at_ms = -1

    def ingest(self, message):
        received = require_int(message["received_at_ms"], "received_at_ms")
        if received < self.last_received_at_ms:
            raise ValueError("messages must be ordered by received_at_ms")
        payload = message["payload"]
        roi_id = payload["roi_id"]
        if roi_id not in self.history:
            raise ValueError("unknown ROI: {}".format(roi_id))
        topic = message["topic"]
        if topic == self.config["topics"]["detections"]:
            captured = require_int(payload["captured_at_ms"], "captured_at_ms")
            if captured > received:
                raise ValueError("capture time is ahead of the receiver clock")
            if not isinstance(payload["detections"], list):
                raise ValueError("detections must be a full list, including empty lists")
            inside, seen = set(), set()
            operation = self.config["regions"][roi_id]["operation_roi_xyxy"]
            for person in payload["detections"]:
                passenger_id = require_text(person["passenger_id"], "passenger_id")
                if passenger_id in seen:
                    raise ValueError("duplicate passenger ID in one snapshot")
                seen.add(passenger_id)
                box = require_box(person["bbox_xyxy"], "bbox_xyxy")
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if operation[0] <= cx < operation[2] and operation[1] <= cy < operation[3]:
                    inside.add(passenger_id)
            inside = frozenset(inside)
            old = self.history[roi_id].get(captured)
            if old is not None and old != inside:
                raise ValueError("conflicting snapshots for the same ROI and capture time")
        elif topic == self.config["topics"]["rfid"]:
            require_text(payload["rfid"], "rfid")
            if received < self.parameters["rfid_time_offset_ms"]:
                raise ValueError("RFID reference time must be nonnegative")
        else:
            raise ValueError("unknown topic: {}".format(topic))

        # 截止时刻收到的消息也参与本次判定，先处理严格早于当前时刻的任务。
        decisions = self._finalize(received, inclusive=False)
        self.last_received_at_ms = received
        if topic == self.config["topics"]["detections"]:
            self.history[roi_id][captured] = inside
        else:
            self.pending.append({
                "roi_id": roi_id,
                "rfid": payload["rfid"],
                "rfid_received_at_ms": received,
                "reference_at_ms": received - self.parameters["rfid_time_offset_ms"],
                "decided_at_ms": received + self.parameters["binding_wait_ms"],
            })
        self._prune(received)
        return decisions

    def advance(self, received_at_ms):
        require_int(received_at_ms, "received_at_ms")
        if received_at_ms < self.last_received_at_ms:
            raise ValueError("timer must not move backwards")
        decisions = self._finalize(received_at_ms, inclusive=True)
        self.last_received_at_ms = received_at_ms
        self._prune(received_at_ms)
        return decisions

    def finish(self):
        if not self.pending:
            return []
        return self.advance(self.pending[-1]["decided_at_ms"])

    def _finalize(self, now, inclusive):
        decisions = []
        while self.pending:
            deadline = self.pending[0]["decided_at_ms"]
            if deadline > now or (deadline == now and not inclusive):
                break
            decisions.append(self._decide(self.pending.popleft()))
        return decisions

    def _prune(self, now):
        retention = sum(self.parameters[key] for key in (
            "lookback_ms", "binding_wait_ms", "rfid_time_offset_ms", "max_sample_gap_ms"
        ))
        for history in self.history.values():
            for captured in list(history):
                if captured < now - retention:
                    del history[captured]

    def _decide(self, event):
        # RFID 参考时刻之前的最后位置决定人员是否仍在区内，过期位置不可使用。
        reference = event["reference_at_ms"]
        samples = sorted(
            (captured, people)
            for captured, people in self.history[event["roi_id"]].items()
            if captured <= reference
        )
        result = dict(event, status="unbound", reason="no_position_data", passenger_id=None,
                      binding=None,
                      latest_snapshot_at_ms=None, position_age_ms=None, candidates=[])
        if not samples:
            return result
        latest_at, current = samples[-1]
        age = reference - latest_at
        fresh = age <= self.parameters["max_position_age_ms"]
        result.update(latest_snapshot_at_ms=latest_at, position_age_ms=age)
        window_start = reference - self.parameters["lookback_ms"]
        dwell = defaultdict(int)
        for (begin, before), (end, after) in zip(samples, samples[1:]):
            duration = end - max(begin, window_start)
            if duration > 0 and end - begin <= self.parameters["max_sample_gap_ms"]:
                # 相邻两次检测都看到同一旅客，才累计中间的停留时间。
                for passenger_id in before & after:
                    dwell[passenger_id] += duration
        eligible = []
        for passenger_id in sorted(set(dwell) | current):
            qualifies = (
                fresh and passenger_id in current
                and dwell[passenger_id] >= self.parameters["min_dwell_ms"]
            )
            result["candidates"].append({
                "passenger_id": passenger_id,
                "dwell_ms": dwell[passenger_id],
                "present_in_latest_snapshot": passenger_id in current,
                "eligible": qualifies,
            })
            if qualifies:
                eligible.append(passenger_id)
        if not fresh:
            result["reason"] = "stale_position_data"
        elif not eligible:
            result["reason"] = "no_eligible_candidate"
        elif len(eligible) == 1:
            # 只有一个合格旅客才绑定；多人同时满足条件时保留为待确认。
            result.update(status="bound", reason="unique_candidate", passenger_id=eligible[0])
            result["binding"] = {
                "roi_id": event["roi_id"],
                "passenger_id": binding_passenger_id(eligible[0]),
                "rfid": event["rfid"],
            }
        else:
            result.update(status="deferred", reason="multiple_candidates")
        return result
