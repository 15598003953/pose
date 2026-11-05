"""Generate offline MQTT-shaped input logs and independent binding labels."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import re
import uuid


from pose_project.common.project_paths import CONFIG_DIR, DATA_DIR


DEFAULT_CONFIG = CONFIG_DIR / "binding_simulation_config.json"
DEFAULT_OUTPUT = DATA_DIR / "binding_simulation"


def require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def require_pair(value, name, minimum=0):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("{} must contain two integers".format(name))
    for item in value:
        require_int(item, name, minimum)
    return value


def require_rect(value, name):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("{} must be [x1, y1, x2, y2]".format(name))
    for item in value:
        require_int(item, name)
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("{} must have positive width and height".format(name))
    return value


def inside_rect(inner, outer):
    return (
        outer[0] <= inner[0] <= inner[2] <= outer[2]
        and outer[1] <= inner[1] <= inner[3] <= outer[3]
    )


def position_at(track, offset_ms):
    for segment in track["segments"]:
        if segment["start_ms"] <= offset_ms < segment["end_ms"]:
            start = segment["center"]
            end = segment.get("end_center", start)
            fraction = (
                (offset_ms - segment["start_ms"])
                / (segment["end_ms"] - segment["start_ms"])
            )
            return [start[i] + (end[i] - start[i]) * fraction for i in (0, 1)]
    return None


def validate_config(config):
    start = datetime.fromisoformat(config["start_time"])
    if start.utcoffset() is None:
        raise ValueError("start_time must include a timezone, for example +08:00")
    require_int(config["seed"], "seed")
    duration = require_int(config["duration_ms"], "duration_ms", 1)
    interval = require_int(config["update_interval_ms"], "update_interval_ms", 1)
    if duration % interval:
        raise ValueError("duration_ms must be divisible by update_interval_ms")
    width, height = require_pair(config["frame_size"], "frame_size", 1)
    jitter = require_int(config["bbox_jitter_px"], "bbox_jitter_px")
    for name in ("detection_delay_ms", "photoelectric_delay_ms"):
        low, high = require_pair(config[name], name)
        if low > high:
            raise ValueError("{} must be [minimum, maximum]".format(name))
    parameters = config["binding_parameters"]
    window = require_int(parameters["lookback_ms"], "lookback_ms", 1)
    dwell = require_int(parameters["min_dwell_ms"], "min_dwell_ms", 1)
    if dwell > window:
        raise ValueError("min_dwell_ms must not exceed lookback_ms")
    topics = config["topics"]
    for name in ("detections", "rfid"):
        if not isinstance(topics[name], str) or not topics[name]:
            raise ValueError("topics.{} must be a nonempty string".format(name))
    if topics["detections"] == topics["rfid"]:
        raise ValueError("detection and RFID topics must differ")

    regions = config["regions"]
    if not isinstance(regions, dict) or not regions:
        raise ValueError("regions must be a nonempty object")
    for region_id, region in regions.items():
        if not re.fullmatch("[A-Z]", region_id):
            raise ValueError("region IDs must be single uppercase letters")
        roi = require_rect(region["roi_xyxy"], "roi_xyxy")
        operation = require_rect(region["operation_roi_xyxy"], "operation_roi_xyxy")
        if not inside_rect(roi, [0, 0, width, height]):
            raise ValueError("ROI {} is outside the frame".format(region_id))
        if not inside_rect(operation, roi):
            raise ValueError("operation ROI {} is outside its ROI".format(region_id))

    tracks = {}
    for track in config["tracks"]:
        name = track["name"]
        if not isinstance(name, str) or not name or name in tracks:
            raise ValueError("track names must be nonempty and unique")
        if track["roi_id"] not in regions:
            raise ValueError("unknown track ROI: {}".format(track["roi_id"]))
        box_width, box_height = require_pair(track["bbox_size"], "bbox_size", 1)
        if not track["segments"]:
            raise ValueError("each track needs at least one segment")
        previous_end = 0
        for segment in track["segments"]:
            begin = require_int(segment["start_ms"], "segment.start_ms")
            end = require_int(segment["end_ms"], "segment.end_ms", 1)
            if not previous_end <= begin < end <= duration:
                raise ValueError("track segments must be ordered and nonoverlapping")
            for key in ("center", "end_center"):
                if key not in segment and key == "end_center":
                    continue
                cx, cy = require_pair(segment[key], key)
                bounds = [
                    cx - box_width / 2 - jitter,
                    cy - box_height / 2 - jitter,
                    cx + box_width / 2 + jitter,
                    cy + box_height / 2 + jitter,
                ]
                if not inside_rect(bounds, regions[track["roi_id"]]["roi_xyxy"]):
                    raise ValueError("track {} bbox can leave its ROI".format(name))
            previous_end = end
        tracks[name] = track

    seen_rfids = set()
    for event in config["push_events"]:
        at_ms = require_int(event["at_ms"], "push_event.at_ms")
        if at_ms >= duration:
            raise ValueError("push events must occur before duration_ms")
        track = tracks.get(event["passenger"])
        if track is None or track["roi_id"] != event["roi_id"]:
            raise ValueError("push event passenger must belong to the event ROI")
        if position_at(track, at_ms) is None:
            raise ValueError("expected passenger must be visible at the push time")
        rfid = event["rfid"]
        if not isinstance(rfid, str) or not rfid or rfid in seen_rfids:
            raise ValueError("RFIDs must be nonempty and unique in this simulation")
        seen_rfids.add(rfid)
    return start


def generate_dataset(config):
    start = validate_config(config)
    start_ms = int(start.timestamp() * 1000)
    rng = random.Random(config["seed"])
    identities = {}
    used_ids = set()
    for track in config["tracks"]:
        first_seen = start + timedelta(milliseconds=track["segments"][0]["start_ms"])
        while True:
            token = uuid.UUID(int=rng.getrandbits(128), version=4).hex
            passenger_id = "{}-{}-{}".format(
                track["roi_id"], token, first_seen.date().isoformat()
            )
            if passenger_id not in used_ids:
                break
        identities[track["name"]] = passenger_id
        used_ids.add(passenger_id)

    messages = []
    for offset in range(0, config["duration_ms"], config["update_interval_ms"]):
        people = {region_id: [] for region_id in config["regions"]}
        for track in config["tracks"]:
            center = position_at(track, offset)
            if center is None:
                continue
            jitter = config["bbox_jitter_px"]
            cx, cy = [value + rng.randint(-jitter, jitter) for value in center]
            width, height = track["bbox_size"]
            x1, y1 = round(cx - width / 2), round(cy - height / 2)
            people[track["roi_id"]].append({
                "passenger_id": identities[track["name"]],
                "bbox_xyxy": [x1, y1, x1 + width, y1 + height],
            })
        for region_id in sorted(people):
            captured_at = start_ms + offset
            messages.append({
                "topic": config["topics"]["detections"],
                "received_at_ms": captured_at + rng.randint(
                    *config["detection_delay_ms"]
                ),
                "payload": {
                    "roi_id": region_id,
                    "captured_at_ms": captured_at,
                    "detections": people[region_id],
                },
            })

    expected = []
    for event in config["push_events"]:
        pushed_at = start_ms + event["at_ms"]
        received_at = pushed_at + rng.randint(*config["photoelectric_delay_ms"])
        messages.append({
            "topic": config["topics"]["rfid"],
            "received_at_ms": received_at,
            "payload": {"roi_id": event["roi_id"], "rfid": event["rfid"]},
        })
        # Labels come from the authored scenario, never from a candidate heuristic.
        expected.append({
            "roi_id": event["roi_id"],
            "rfid": event["rfid"],
            "passenger_id": identities[event["passenger"]],
            "rfid_received_at_ms": received_at,
            "simulated_push_at_ms": pushed_at,
            "scenario": event["scenario"],
        })
    messages.sort(key=lambda item: item["received_at_ms"])
    expected.sort(key=lambda item: item["rfid_received_at_ms"])
    return messages, expected


def write_dataset(output, config, messages, expected):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    detection_topic = config["topics"]["detections"]
    rfid_topic = config["topics"]["rfid"]
    datasets = {
        "detections.jsonl": [m for m in messages if m["topic"] == detection_topic],
        "rfid.jsonl": [m for m in messages if m["topic"] == rfid_topic],
        "replay.jsonl": messages,
        "expected_bindings.jsonl": expected,
    }
    for filename, rows in datasets.items():
        with (output / filename).open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "synthetic": True,
        "description": "Offline input logs; no live MQTT publishing or binding engine.",
        "counts": {filename: len(rows) for filename, rows in datasets.items()},
        "config": config,
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output, manifest["counts"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        with args.config.open(encoding="utf-8") as handle:
            config = json.load(handle)
        messages, expected = generate_dataset(config)
        output, counts = write_dataset(args.output, config, messages, expected)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    print("Generated synthetic data: {}".format(output))
    for filename, count in counts.items():
        print("{}: {} records".format(filename, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
