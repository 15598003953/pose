"""联合工作台配置：默认值、区域校验、路径解析、备份和原子保存。"""

from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import string
import sys
from urllib.parse import urlsplit
import uuid

from pose_project.binding.binding_runtime import validate_broker, validate_topic
from pose_project.binding.passenger_tray_binding import require_box, validate_config
from pose_project.common.project_paths import (
    APP_DIR, CONFIG_DIR, GATE_MODEL_DIR, REID_MODEL_DIR, RESOURCE_CONFIG_DIR, RESULT_DIR,
)


DEFAULT_CONFIG = CONFIG_DIR / "integration_config.json"
REID_RUNTIME_DEFAULTS = {
    "sample_interval_ms": 200, "detector_image_size": 960,
    "preview_fps": 15, "flip_tta": True,
}
FIELD_TEST_DEFAULTS = {
    "enabled": False, "test_id": "", "operator": "", "site": "",
    "scenario": "", "notes": "",
}
FRONT_MODE_VALUES = (None, "bottom", "top", "left", "right", "area")
FRONT_MODE_ITEMS = ("队首:关", "队首:下", "队首:上", "队首:左",
                    "队首:右", "队首:最大")
DEFAULT_FRONT_MODE = "bottom"


def is_stream_source(source):
    return "://" in source


def validate_video_source(source, name="视频"):
    if not source:
        raise ValueError("请选择{}文件或填写 RTSP 地址".format(name))
    if is_stream_source(source):
        try:
            url = urlsplit(source)
            if (url.scheme.lower() not in
                    ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "udp", "tcp")
                    or not url.hostname or any(char.isspace() for char in source)
                    or (url.port is not None and not 1 <= url.port <= 65535)):
                raise ValueError()
        except ValueError:
            raise ValueError("{}流地址无效，请检查协议、主机和端口".format(name)) from None
    elif not Path(source).is_file():
        raise ValueError("{}文件不存在：{}".format(name, source))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def defaults():
    gate = read_json(RESOURCE_CONFIG_DIR / "gate_config.json")
    binding = read_json(CONFIG_DIR / "binding_config.json")
    binding["regions"] = {
        name: {"roi_xyxy": geometry.get("roi_xyxy"),
               "front_mode": DEFAULT_FRONT_MODE}
        for name, geometry in binding["regions"].items()
    }
    binding["topics"] = {"detections": "reidtest", "rfid": "pose/rfid"}
    return {
        "transport": "local",
        "gate": dict(gate, source=""),
        "reid": {
            "source": "", "rotate180": False, "threshold": 0.65,
            **REID_RUNTIME_DEFAULTS,
            "image_size": [1920, 1080], "enabled_regions": ["A"],
            "detector_weights": str(REID_MODEL_DIR / "yolov8s.pt"),
            "reid_weights": str(REID_MODEL_DIR / "msmt_sbs_R50.pth"),
        },
        "binding": binding,
        "mqtt": {
            "host": "127.0.0.1", "port": 1883, "username": "", "tls": False,
            "person_topic": "check_gate/person/collected",
            "output_topic": "pose/bindings", "output_enabled": False,
        },
        "debug": {
            "enabled": False,
            "field_test": dict(FIELD_TEST_DEFAULTS),
        },
        "output_dir": str(RESULT_DIR / "integration"),
    }


def region_error(name, geometry, image_size):
    width, height = image_size
    box = geometry.get("roi_xyxy")
    if box is None:
        return "{} 区缺少区域，请先框选".format(name)
    try:
        require_box(box, "roi_xyxy")
    except ValueError:
        return "{} 区坐标无效，请重新框选".format(name)
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        return "{} 区超出画面，请重新框选".format(name)
    return None


def active_binding(settings):
    # 工作台每个供框位只画一个区域，绑定算法沿用这个区域作为操作范围。
    binding = deepcopy(settings["binding"])
    binding["regions"] = {
        name: dict(binding["regions"][name],
                   operation_roi_xyxy=deepcopy(binding["regions"][name]["roi_xyxy"]))
        for name in settings["reid"]["enabled_regions"]}
    return binding


def validate(settings, sources=False, *, require_regions=True):
    if settings["transport"] not in ("local", "mqtt"):
        raise ValueError("消息连接方式必须为 local 或 mqtt")
    mqtt = settings["mqtt"]
    validate_broker(mqtt)
    topics = [mqtt["person_topic"], mqtt["output_topic"],
              *settings["binding"]["topics"].values()]
    for topic in topics:
        validate_topic(topic)
    if len(set(topics)) != len(topics):
        raise ValueError("旅客、位置、RFID 和绑定结果必须使用不同主题")
    reid = settings["reid"]
    width, height = reid["image_size"]
    if any(type(v) is not int or v < 16 for v in (width, height)):
        raise ValueError("供框台画面尺寸必须是大于 15 的整数")
    if not 0 <= reid["threshold"] <= 1:
        raise ValueError("重识别阈值必须位于 0 到 1")
    for name, lower, upper, label in (
            ("sample_interval_ms", 50, 1000, "识别间隔"),
            ("detector_image_size", 320, 1280, "检测输入尺寸"),
            ("preview_fps", 1, 30, "预览帧率")):
        value = reid.get(name, REID_RUNTIME_DEFAULTS[name])
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("{}必须是 {} 到 {} 的整数".format(label, lower, upper))
    if reid.get("detector_image_size", 960) % 32:
        raise ValueError("检测输入尺寸必须是 32 的倍数")
    if type(reid.get("flip_tta", True)) is not bool:
        raise ValueError("翻转增强必须为开关值")
    debug = settings.get("debug", {})
    if not isinstance(debug, dict):
        raise ValueError("调试配置必须为对象")
    if type(debug.get("enabled", False)) is not bool:
        raise ValueError("性能日志必须为开关值")
    field_test = debug.get("field_test", {})
    if not isinstance(field_test, dict):
        raise ValueError("现场测试记录配置必须为对象")
    if type(field_test.get("enabled", False)) is not bool:
        raise ValueError("现场测试记录必须为开关值")
    for key in ("test_id", "operator", "site", "scenario", "notes"):
        if not isinstance(field_test.get(key, ""), str):
            raise ValueError("现场测试记录的{}必须为文字".format(key))
    regions = settings["binding"]["regions"]
    if not isinstance(regions, dict):
        raise ValueError("供框台区域配置必须为对象")
    for name, geometry in regions.items():
        if name not in string.ascii_uppercase or len(name) != 1:
            raise ValueError("供框台区域编号必须为 A 到 Z")
        if not isinstance(geometry, dict):
            raise ValueError("{} 区配置必须为对象".format(name))
        if geometry.get("front_mode") not in FRONT_MODE_VALUES:
            raise ValueError("{} 区队首模式无效".format(name))
        if geometry.get("roi_xyxy") is not None:
            try:
                require_box(geometry["roi_xyxy"], "roi_xyxy")
            except ValueError:
                raise ValueError("{} 区坐标无效，请重新框选".format(name)) from None
    enabled = reid["enabled_regions"]
    if not isinstance(enabled, list) or any(not isinstance(key, str) for key in enabled):
        raise ValueError("启用区域必须为区域编号列表")
    if len(enabled) != len(set(enabled)):
        raise ValueError("启用区域编号不能重复")
    for key in enabled:
        if key not in regions:
            raise ValueError("缺少 {} 区配置，请新增区域或取消启用".format(key))
    if require_regions:
        if not enabled:
            raise ValueError("至少新增并启用一个供框台区域")
        for key in enabled:
            problem = region_error(key, regions[key], reid["image_size"])
            if problem:
                raise ValueError(problem)
        validate_config(active_binding(settings))
    else:
        # 编辑时允许区域暂未画完，共享参数仍要通过校验。
        binding = dict(settings["binding"], regions={})
        validate_config(binding, allow_empty_regions=True)
    roi = settings["gate"].get("roi")
    if roi is None and require_regions:
        raise ValueError("缺少闸机抓拍区，请先框选")
    if roi is not None and not (0 <= roi["x"] < roi["x"] + roi["w"] <= 1
                               and 0 <= roi["y"] < roi["y"] + roi["h"] <= 1):
        raise ValueError("闸机抓拍区域必须位于画面内")
    if not str(settings["output_dir"]).strip():
        raise ValueError("输出目录不能为空")
    if sources:
        for stage, name in (("gate", "闸机视频"), ("reid", "供框台视频")):
            validate_video_source(settings[stage]["source"].strip(), name)


def load_settings(path=DEFAULT_CONFIG):
    path = Path(path)
    settings = read_json(path) if path.exists() else defaults()
    settings.setdefault("debug", {})
    settings["debug"].setdefault("enabled", False)
    field_test = settings["debug"].setdefault("field_test", {})
    for key, value in FIELD_TEST_DEFAULTS.items():
        field_test.setdefault(key, value)
    for geometry in settings["binding"]["regions"].values():
        if geometry.get("roi_xyxy") is None and geometry.get("operation_roi_xyxy") is not None:
            geometry["roi_xyxy"] = geometry["operation_roi_xyxy"]
        geometry.pop("operation_roi_xyxy", None)
        geometry.setdefault("front_mode", DEFAULT_FRONT_MODE)
    for key, value in REID_RUNTIME_DEFAULTS.items():
        settings["reid"].setdefault(key, value)
    for values, key in _file_paths(settings):
        value = values[key]
        if value and not is_stream_source(value) and not Path(value).is_absolute():
            values[key] = str((path.resolve().parent / value).resolve())
    validate(settings, require_regions=False)
    return settings


def _file_paths(settings):
    yield settings, "output_dir"
    for stage in ("gate", "reid"):
        yield settings[stage], "source"
    for key in ("detector_weights", "reid_weights"):
        yield settings["reid"], key


def save_settings(settings, path=DEFAULT_CONFIG):
    validate(settings, require_regions=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".bak")
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    # MQTT 密码每次启动单独提供，不写入配置文件。
    saved = deepcopy(settings)
    for geometry in saved["binding"]["regions"].values():
        geometry.pop("operation_roi_xyxy", None)
    saved["mqtt"].pop("password", None)
    if getattr(sys, "frozen", False):
        for values, key in _file_paths(saved):
            value = values[key]
            if value and not is_stream_source(value):
                target = Path(value)
                if target.is_absolute() and target.resolve().is_relative_to(APP_DIR.resolve()):
                    values[key] = os.path.relpath(target, path.resolve().parent)
    try:
        temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        if read_json(temporary) != saved:
            raise OSError("配置回读不一致")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def new_run_directory(settings):
    parent = Path(settings["output_dir"]).resolve()
    path = parent / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_report(settings):
    modules = ("PyQt5", "cv2", "numpy", "paho.mqtt.client", "torch",
               "torchvision", "ultralytics", "lap")
    available = {}
    for name in modules:
        try:
            available[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available[name] = False
    files = {
        "gate_detector": GATE_MODEL_DIR / "yolo11n-pose.onnx",
        "gate_face": GATE_MODEL_DIR / "face_detection_yunet_2023mar.onnx",
        "person_detector": Path(settings["reid"]["detector_weights"]),
        "reidentification": Path(settings["reid"]["reid_weights"]),
    }
    weights = {name: {"path": str(path), "exists": path.is_file()}
               for name, path in files.items()}
    return {"ok": all(available.values()) and all(v["exists"] for v in weights.values()),
            "packaged": bool(getattr(sys, "frozen", False)),
            "python": sys.executable, "modules": available, "weights": weights}
