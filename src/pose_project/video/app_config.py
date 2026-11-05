import json
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_SOURCE_TYPES = {"file", "rtsp", "rtmp"}


@dataclass(eq=True)
class SourceConfig:
    source_type: str = "file"
    path_or_url: str = ""
    low_latency: bool = True
    infer_stride: int = 3
    display_fps: int = 15
    infer_interval_ms: int = 300


def load_source_config(path: Path) -> SourceConfig:
    path = Path(path)
    if not path.is_file():
        return SourceConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SourceConfig()
    if not isinstance(payload, dict):
        return SourceConfig()
    source_type = str(payload.get("source_type") or "file").lower()
    path_or_url = str(payload.get("path_or_url") or "")
    low_latency = bool(payload.get("low_latency", True))
    infer_stride = _safe_stride(payload.get("infer_stride", 3))
    display_fps = _safe_int(payload.get("display_fps", 15), default=15, min_value=1, max_value=30)
    infer_interval_ms = _safe_int(
        payload.get("infer_interval_ms", 300),
        default=300,
        min_value=50,
        max_value=5000,
    )
    if source_type not in VALID_SOURCE_TYPES:
        return SourceConfig()
    if source_type in {"rtsp", "rtmp"} and not path_or_url.lower().startswith(source_type + "://"):
        return SourceConfig()
    return SourceConfig(
        source_type=source_type,
        path_or_url=path_or_url,
        low_latency=low_latency,
        infer_stride=infer_stride,
        display_fps=display_fps,
        infer_interval_ms=infer_interval_ms,
    )


def save_source_config(config: SourceConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_type = str(config.source_type or "file").lower()
    path_or_url = str(config.path_or_url or "")
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError("invalid source type: {}".format(config.source_type))
    if source_type in {"rtsp", "rtmp"} and not path_or_url.lower().startswith(source_type + "://"):
        raise ValueError("{} source must start with {}://".format(source_type.upper(), source_type))
    path.write_text(
        json.dumps(
            asdict(SourceConfig(
                source_type=source_type,
                path_or_url=path_or_url,
                low_latency=bool(config.low_latency),
                infer_stride=_safe_stride(config.infer_stride),
                display_fps=_safe_int(config.display_fps, default=15, min_value=1, max_value=30),
                infer_interval_ms=_safe_int(
                    config.infer_interval_ms,
                    default=300,
                    min_value=50,
                    max_value=5000,
                ),
            )),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _safe_stride(value):
    try:
        stride = int(value)
    except (TypeError, ValueError):
        return 3
    return stride if stride >= 1 else 1


def _safe_int(value, default, min_value, max_value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, result))
