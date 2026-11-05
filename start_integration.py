"""Start the integration workbench without relying on an editable install."""

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
