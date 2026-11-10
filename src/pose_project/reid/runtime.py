"""重识别 CPU 线程设置；必须先于 Torch 和 Ultralytics 模型导入。"""

import os

# Ultralytics 导入时会读取这两个变量，导入后再设置不会产生同样的效果。
CPU_THREADS = 20
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

import torch

DET_THREADS = min(8, CPU_THREADS)
EXT_THREADS = CPU_THREADS


def set_phase_threads(threads):
    """检测与特征提取交替使用 CPU，避免两种阶段争抢线程。"""
    try:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
    except Exception:
        pass
