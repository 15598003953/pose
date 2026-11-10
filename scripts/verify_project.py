"""验证主项目：模块与命令、真实模型、绑定模拟回放；姿态项目独立验证。"""

import argparse
import ast
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pose_project.__main__ import COMMANDS
from pose_project.common.project_paths import ROOT
from pose_project.integration.configuration import defaults
from pose_project.integration.diagnostics import check_models


def run_command(output, cwd, name, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "pose_project", *arguments],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(name + " failed: " + result.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("请选择新的空输出目录")
    output.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": datetime.now().astimezone().isoformat(),
              "project_root": str(ROOT)}
    sources = sorted((ROOT / "src/pose_project").rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parts = path.relative_to(ROOT / "src").with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        importlib.import_module(".".join(parts))
    report["parsed_and_imported_modules"] = len(sources)

    # 从无关目录启动，确保程序不再依赖当前工作目录或旧日期目录。
    with tempfile.TemporaryDirectory(prefix="passenger_verify_") as unrelated:
        for command in COMMANDS:
            run_command(output, unrelated, "help_" + command, command, "--help")
        report["cli_help_checks"] = len(COMMANDS)
        report["commands_work_from_unrelated_directory"] = True
        run_command(output, unrelated, "simulate", "simulate", "--output", str(output / "simulation"))
        run_command(output, unrelated, "replay", "replay",
                    "--input", str(output / "simulation/replay.jsonl"),
                    "--expected", str(output / "simulation/expected_bindings.jsonl"),
                    "--output", str(output / "replay"))
    evaluation = json.loads((output / "replay/evaluation.json").read_text(encoding="utf-8"))
    report["binding"] = {key: value for key, value in evaluation.items() if key != "cases"}
    report["models"] = check_models(defaults())
    report["ok"] = evaluation["all_events_match"] and report["models"]["ok"]
    report["not_verified"] = ["现场识别准确率", "真实 RFID", "生产 MQTT", "重新构建的 EXE"]
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
