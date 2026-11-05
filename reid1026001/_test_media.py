# -*- coding: utf-8 -*-
"""测试素材准备：从现成的检索结果视频裁出短片段（mp4 / ts / 20 帧短片）。

供 _test_step3.py 与 _test_step3_ui.py 复用，保证测试可重复运行。
"""
import os
import shutil
import subprocess

SOURCE = "rtsp_reid_result_20261023_134358.mp4"


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for cand in (r"G:\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"):
        if os.path.exists(cand):
            return cand
    raise RuntimeError("未找到 ffmpeg，无法生成测试素材")


def _run(args):
    subprocess.run([_ffmpeg(), "-y", "-v", "error"] + args, check=True)


def ensure(short_only=False):
    """确保 _test_src.mp4 / _test_src.ts / _test_short.mp4 存在。"""
    if not short_only and not os.path.exists("_test_src.mp4"):
        if not os.path.exists(SOURCE):
            raise RuntimeError(f"缺少素材源视频：{SOURCE}")
        _run(["-i", SOURCE, "-t", "6", "-c:v", "libx264",
              "-preset", "veryfast", "-an", "_test_src.mp4"])
    if not short_only and not os.path.exists("_test_src.ts"):
        _run(["-i", "_test_src.mp4", "-c", "copy", "-f", "mpegts", "_test_src.ts"])
    if not os.path.exists("_test_short.mp4"):
        base = "_test_src.mp4" if os.path.exists("_test_src.mp4") else SOURCE
        _run(["-i", base, "-frames:v", "20", "-c:v", "libx264",
              "-preset", "veryfast", "-an", "_test_short.mp4"])
