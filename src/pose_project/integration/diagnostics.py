"""本地模型诊断：实际加载并推理，检查检测器、重识别特征形状和数值。"""

import importlib.metadata

import numpy as np

def check_models(settings):
    report = {"ok": False, "kind": "local-model-inference-only"}
    try:
        import torch

        from pose_project.gate import core as gate
        from pose_project.reid import core as module

        detector = gate.PersonDetector(settings["gate"]["opts"])
        if detector.yolo is None or not detector.face_engine.available:
            raise RuntimeError("闸机人体或人脸模型没有成功加载")
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        report["gate"] = {"person": detector.yolo.available,
                          "face": detector.face_engine.status_text()}
        person = module.PersonDetector(weights=settings["reid"]["detector_weights"],
                                       allow_download=False)
        boxes, _, _ = person.detect_track(np.zeros((640, 640, 3), dtype=np.uint8),
                                          imgsz=640)
        extractor = module.ReidExtractor(weights=settings["reid"]["reid_weights"])
        features = extractor.extract([np.zeros((192, 64, 3), dtype=np.uint8)], tta=True)
        if features.shape != (1, 2048) or not np.isfinite(features).all():
            raise RuntimeError("重识别特征形状或数值无效")
        report.update(
            ok=True, device=str(extractor.device), cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            person_boxes=len(boxes), feature_shape=list(features.shape),
            feature_norm=float(np.linalg.norm(features[0])),
            versions={name: importlib.metadata.version(name)
                      for name in ("torch", "torchvision", "ultralytics", "lapx")})
    except Exception as exc:
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
    return report
