# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

root = Path(SPECPATH).parent
datas = [
    (str(root / "check_gate/person_extractor.py"), "check_gate"),
    (str(root / "check_gate/mqtt_publisher.py"), "check_gate"),
    (str(root / "check_gate/extractor_config.json"), "check_gate"),
    (str(root / "check_gate/models"), "check_gate/models"),
    (str(root / "reid1026001/main.py"), "reid1026001"),
    (str(root / "reid1026001/reid_core.py"), "reid1026001"),
    (str(root / "reid1026001/yolov8s.pt"), "reid1026001"),
    (str(root / "reid1026001/weights/msmt_sbs_R50.pth"), "reid1026001/weights"),
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
    exclude_binaries=True, name="PoseIntegration", console=True,
    debug=False, strip=False, upx=False, contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PoseIntegration")
