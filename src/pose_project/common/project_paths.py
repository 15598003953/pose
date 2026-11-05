"""Resource paths for the editable local project, independent of the working directory."""

import os
from pathlib import Path
import sys


ROOT = (Path(sys._MEIPASS) if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[3])
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
CONFIG_DIR = APP_DIR / "configs"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULT_DIR = APP_DIR / "result"
POSE_MODEL_PATH = str(MODEL_DIR / "pose_estimation_mediapipe_2023mar.onnx")
PERSON_MODEL_PATH = os.environ.get("PERSON_DET_MODEL") or str(
    MODEL_DIR / "person_detection_mediapipe_2023mar.onnx"
)
PUT_LUGGAGE_MODEL_PATH = str(MODEL_DIR / "put_luggage_centroid_model.json")
