# -*- coding: utf-8 -*-
"""人物重识别模型：行人检测、跨帧跟踪和外观特征提取。

- PersonDetector: 基于 YOLOv8 的行人检测（普通/跨帧跟踪两种模式），默认使用 CUDA 推理。
- ReidExtractor: 基于 FastReID SBS R50（MSMT17 预训练，2048 维特征）的外观特征提取器。
  架构为 ResNet50 + Non-Local 注意力 + GeM 池化 + BNNeck，
  权重文件为 models/reid/msmt_sbs_R50.pth。
  输出归一化特征，具体身份判定和会话锁定见 matcher.py。

架构对照 fastreid v0.1.1 源码复现（Apache-2.0 License）。
"""
# 环境变量必须在 Torch / Ultralytics 导入前设置。
from pose_project.reid import runtime
from pose_project.common.project_paths import REID_MODEL_DIR

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO

# Ultralytics 会关闭 OpenCV 多线程；恢复原程序使用的并行数量。
try:
    cv2.setNumThreads(4)
except Exception:
    pass


# ===================== FastReID SBS R50 架构 =====================
class NonLocalBlock(nn.Module):
    """Non-Local 注意力块（fastreid non_local.py，inter_channels=1）。"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = 1
        self.g = nn.Conv2d(in_channels, 1, 1)
        self.W = nn.Sequential(
            nn.Conv2d(1, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels, 1, 1)
        self.phi = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        b = x.size(0)
        g_x = self.g(x).view(b, 1, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(b, 1, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, 1, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / f.size(-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous().view(b, 1, *x.size()[2:])
        return self.W(y) + x


class Bottleneck(nn.Module):
    """标准 ResNet Bottleneck。"""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class FastReidResNet(nn.Module):
    """ResNet50 骨干（last_stride=1，layer2 末 2 个 + layer3 末 3 个 NL 块）。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inplanes = 64

        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 1)  # last_stride=1

        self.NL_2 = nn.ModuleList([NonLocalBlock(512) for _ in range(2)])
        self.NL_2_idx = [2, 3]
        self.NL_3 = nn.ModuleList([NonLocalBlock(1024) for _ in range(3)])
        self.NL_3_idx = [3, 4, 5]

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes * 4, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for blk in self.layer1:
            x = blk(x)
        nl = 0
        for i, blk in enumerate(self.layer2):
            x = blk(x)
            if i in self.NL_2_idx:
                x = self.NL_2[nl](x)
                nl += 1
        nl = 0
        for i, blk in enumerate(self.layer3):
            x = blk(x)
            if i in self.NL_3_idx:
                x = self.NL_3[nl](x)
                nl += 1
        for blk in self.layer4:
            x = blk(x)
        return x


class GeM(nn.Module):
    """Generalized Mean Pooling（p 可学习）。"""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)


class BNNeckHead(nn.Module):
    """GeM 池化 + BNNeck。eval 模式返回 BN 后的 2048 维特征。"""

    def __init__(self):
        super().__init__()
        self.pool_layer = GeM()
        self.bnneck = nn.BatchNorm2d(2048)

    def forward(self, x):
        v = self.pool_layer(x)
        return self.bnneck(v)[..., 0, 0]


class FastReidSBS(nn.Module):
    """FastReID SBS R50 完整模型（推理只保留 backbone + head）。"""

    def __init__(self):
        super().__init__()
        self.backbone = FastReidResNet()
        self.heads = BNNeckHead()

    def forward(self, x):
        return self.heads(self.backbone(x))


# ===================== 特征提取器 =====================
class ReidExtractor:
    """基于 FastReID SBS R50（MSMT17 预训练）的人物外观特征提取器。

    输出 L2 归一化的 2048 维特征。输入统一缩放到 384x128（训练分辨率），
    ImageNet 均值方差归一化。
    """

    WEIGHT_NAME = "msmt_sbs_R50.pth"

    def __init__(self, weights=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = weights or self._find_weights()
        model = FastReidSBS()
        state_dict = torch.load(weight_path, map_location="cpu",
                                weights_only=True)["model"]
        # 过滤推理不需要的键：pixel_mean/pixel_std 缓冲区（预处理参数）、
        # heads.classifier（训练用分类层，eval 模式只走 BNNeck 特征）
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("pixel_")
                      and not k.startswith("heads.classifier")}
        model.load_state_dict(state_dict, strict=True)
        self.model = model.eval().to(self.device)
        self.feat_dim = 2048
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((384, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def _find_weights(cls):
        path = REID_MODEL_DIR / cls.WEIGHT_NAME
        if path.is_file():
            return str(path)
        raise FileNotFoundError(
            f"未找到 FastReID 权重：{path}")

    @torch.no_grad()
    def extract(self, bgr_crops, tta=False):
        """对一组 BGR 裁剪图提取特征，返回 (N, 2048) 的 numpy 数组。

        tta=True 时做水平翻转增强：原图与翻转图特征取平均后重新归一化，
        可提升对行走方向变化的鲁棒性（推荐用于人物库图像）。
        """
        if not bgr_crops:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        tensors = []
        for c in bgr_crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb)
            tensors.append(t)
            if tta:
                tensors.append(torch.flip(t, dims=[2]))
        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch)
        feats = F.normalize(feats, p=2, dim=1)
        if tta:
            feats = F.normalize(feats.view(-1, 2, feats.size(1)).mean(dim=1),
                                p=2, dim=1)
        return feats.cpu().numpy()


# ===================== 行人检测器 =====================
class PersonDetector:
    """YOLOv8 行人检测器（class 0 = person），默认使用 CUDA 推理。

    优先使用精度更高的 yolov8s；不可用时回退 yolov8n。
    """

    # 本地权重损坏时的官方回退下载地址。
    DEFAULT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

    def __init__(self, weights=None, device=None, allow_download=True):
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        if weights is None:
            weights = self._pick_weights()
        try:
            self.model = YOLO(weights)
        except Exception:
            if not allow_download:
                raise
            self.model = YOLO(self.DEFAULT_URL)

    @staticmethod
    def _pick_weights():
        """按精度优先级选择本地权重：yolov8s > yolov8n。"""
        for name in ("yolov8s.pt", "yolov8n.pt"):
            path = REID_MODEL_DIR / name
            if path.is_file():
                return str(path)
        return "yolov8s.pt"  # 本地无权重时让 ultralytics 自动下载 s 版

    def _parse(self, res):
        boxes = res[0].boxes
        if len(boxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()

    def detect(self, frame, conf=0.3, imgsz=960):
        """普通检测（无跨帧跟踪）。返回 (xyxy, confs)。"""
        res = self.model(frame, classes=[0], conf=conf, imgsz=imgsz,
                         verbose=False, device=self.device)
        return self._parse(res)

    def reset_tracker(self):
        """重置跨帧跟踪器状态（开始新一次检索时调用）。

        ByteTrack 的 persist 模式会把上一次检索的轨迹 id 带到下一次，
        导致新视频前几帧的 track id 与旧轨迹历史错位，故每次检索前清空。
        """
        try:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) or []
            for tr in trackers:
                reset = getattr(tr, "reset", None)
                if callable(reset):
                    reset()
        except Exception:
            pass

    def detect_track(self, frame, conf=0.3, imgsz=960, persist=True):
        """跨帧跟踪检测（ByteTrack），可为同一行人维持稳定 track id。

        返回 (xyxy, confs, track_ids)；track_ids[i] 为 int 或 None。
        需要按帧顺序调用，并保持 persist=True。
        """
        res = self.model.track(frame, persist=persist, classes=[0], conf=conf,
                               imgsz=imgsz, tracker="bytetrack.yaml",
                               verbose=False, device=self.device)
        boxes = res[0].boxes
        if len(boxes) == 0:
            zeros4 = np.zeros((0, 4), dtype=np.float32)
            return zeros4, np.zeros((0,), dtype=np.float32), []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = ([int(i) for i in boxes.id.int().cpu().tolist()]
               if boxes.id is not None else [None] * len(boxes))
        return xyxy, confs, ids


# ===================== 工具函数 =====================
def crop_from_frame(frame, xyxy):
    """根据 xyxy 框从画面中裁剪人物，做边界裁剪。"""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2]


def valid_crop(crop, min_h=24, min_w=12):
    return crop is not None and crop.size > 0 and crop.shape[0] >= min_h and crop.shape[1] >= min_w


