# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
from PyInstaller.utils.hooks import copy_metadata

root = Path(SPECPATH).parent
app_name = os.environ.get("POSE_EXE_NAME", "PoseIntegration")
datas = [
    # Python 源码由模块依赖收集，数据区只放模型和默认配置。
    (str(root / "configs/gate_config.json"), "configs"),
    (str(root / "models/gate"), "models/gate"),
    (str(root / "models/reid"), "models/reid"),
]
for name in ("torch", "torchvision", "ultralytics", "ultralytics-thop",
             "lapx", "numpy", "opencv-contrib-python", "PyQt5", "paho-mqtt",
             "psutil", "py-cpuinfo", "seaborn"):
    datas += copy_metadata(name)

a = Analysis(
    [str(root / "start_integration.py")],
    pathex=[str(root / "src")],
    binaries=[], datas=datas,
    hiddenimports=[
        "pose_project.gate.processor", "pose_project.reid.search_worker",
        "torch", "torchvision", "torchvision.transforms", "torchvision.ops",
        "ultralytics", "ultralytics.nn.tasks", "ultralytics.trackers.byte_tracker",
        "ultralytics.trackers.bot_sort", "lap", "thop", "cpuinfo", "psutil",
    ],
    hookspath=[],
    runtime_hooks=[str(root / "packaging/runtime_hook.py")],
    excludes=[
        "mediapipe", "onnxruntime", "tensorflow", "tensorboard",
        "PySide2", "PySide6", "PyQt6", "IPython", "notebook", "jupyter",
        "pytest", "tkinter",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True, name=app_name, console=True,
    debug=False, strip=False, upx=False, contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=app_name)
