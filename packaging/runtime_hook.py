"""发布包启动设置：把库缓存放到可写目录，并关闭在线依赖自动安装。"""

import os
from pathlib import Path
import sys

base = Path(sys.executable).resolve().parent
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_CONFIG_DIR", str(base / "runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(base / "runtime/matplotlib"))
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
