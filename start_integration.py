"""联合工作台启动入口；自动加入 src 路径，源码运行时无需可编辑安装。"""

from pathlib import Path
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pose_project.apps.integration_app import main


if __name__ == "__main__":
    raise SystemExit(main())
