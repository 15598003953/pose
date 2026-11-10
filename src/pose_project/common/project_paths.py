"""统一资源路径：区分源码、发布资源与可写配置，不依赖命令启动目录。"""

from pathlib import Path
import sys


ROOT = (Path(sys._MEIPASS) if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[3])
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
CONFIG_DIR = APP_DIR / "configs"
MODEL_DIR = ROOT / "models"
GATE_MODEL_DIR = MODEL_DIR / "gate"
REID_MODEL_DIR = MODEL_DIR / "reid"
# 发布包的默认配置随资源打包；用户修改的配置写到 EXE 旁的 configs。
RESOURCE_CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RESULT_DIR = APP_DIR / "result"
