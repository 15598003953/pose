"""构建 Windows 便携发布目录；收集模型、配置、运行库，并生成校验清单。"""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_project.integration.configuration import load_settings


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_vc_runtime(bundle):
    # Qt 自带的旧运行库会阻止当前 Torch 启动，发布时统一使用构建环境的版本。
    names = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
    for name in names:
        source = Path(sys.base_prefix) / name
        if not source.is_file():
            raise FileNotFoundError("Missing build-environment runtime: " + str(source))
        targets = [path for path in (bundle / "_internal").rglob("*")
                   if path.is_file() and path.name.lower() == name]
        for target in targets:
            shutil.copy2(source, target)


def write_manifest(bundle):
    manifest = [{"path": path.relative_to(bundle).as_posix(), "size": path.stat().st_size,
                 "sha256": sha256(path)}
                for path in sorted(bundle.rglob("*"))
                if path.is_file() and path != bundle / "manifest.json"]
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_bundle(bundle, archive):
    bundle, archive = Path(bundle), Path(archive)
    if archive.exists() or archive.with_suffix(".zip.sha256").exists():
        raise FileExistsError("Archive already exists: " + str(archive))
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as packed:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                packed.write(path, Path(bundle.name) / path.relative_to(bundle))
    with zipfile.ZipFile(archive) as packed:
        bad = packed.testzip()
        if bad:
            raise OSError("ZIP verification failed: " + bad)
    checksum = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(
        checksum + "  " + archive.name + "\n", encoding="ascii")
    print("ARCHIVE " + str(archive), flush=True)
    print("SHA256 " + checksum, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name", default="PoseIntegration", help="Executable and bundle name")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32" or platform.architecture()[0] != "64bit":
        parser.error("Build this package with 64-bit Python on Windows")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", args.name):
        parser.error("Name must start with a letter and contain only letters, digits, _ or -")
    output = (args.output or ROOT / "dist" /
              (args.name + "_win64_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    output.mkdir(parents=True)
    work = ROOT / "build" / output.name
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm",
               "--distpath", str(output), "--workpath", str(work),
               str(ROOT / "packaging/integration.spec")]
    print("Building " + str(output), flush=True)
    subprocess.run(command, cwd=ROOT, check=True,
                   env=dict(os.environ, POSE_EXE_NAME=args.name))
    bundle = output / args.name
    prepare_vc_runtime(bundle)
    config_dir = bundle / "configs"
    config_dir.mkdir()
    settings = load_settings()
    settings["transport"] = "local"
    settings["gate"]["source"] = ""
    settings["reid"].update(
        source="", detector_weights="../_internal/models/reid/yolov8s.pt",
        reid_weights="../_internal/models/reid/msmt_sbs_R50.pth")
    settings["mqtt"].update(host="127.0.0.1", username="", output_enabled=False)
    settings["mqtt"].pop("password", None)
    settings["output_dir"] = "../result/integration"
    (config_dir / "integration_config.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "configs/binding_config.json", config_dir / "binding_config.json")
    for name in ("start_server.cmd", "check_environment.cmd"):
        text = (ROOT / "packaging" / name).read_text(encoding="utf-8")
        (bundle / name).write_text(text.replace("PoseIntegration", args.name),
                                  encoding="utf-8", newline="\r\n")
    text = (ROOT / "docs/EXE_DEPLOYMENT.md").read_text(encoding="utf-8")
    (bundle / "README.md").write_text(text.replace("PoseIntegration", args.name), encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
    versions = {}
    license_dir = bundle / "licenses"
    license_dir.mkdir()
    for name in ("pyinstaller", "pyinstaller-hooks-contrib", "torch", "torchvision",
                 "ultralytics", "ultralytics-thop", "numpy", "opencv-contrib-python",
                 "PyQt5", "paho-mqtt", "lapx", "psutil", "py-cpuinfo", "seaborn"):
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for file in distribution.files or []:
            if any(part.lower().startswith(("license", "copying", "notice"))
                   for part in file.parts):
                source = Path(distribution.locate_file(file))
                if source.is_file() and ".dist-info" in str(file):
                    destination = license_dir / name / Path(*file.parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    (bundle / "build_info.json").write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(),
        "executable": args.name + ".exe",
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": versions, "layout": "PyInstaller onedir, console enabled",
        "optional_modules_excluded": ["mediapipe", "onnxruntime"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(bundle)
    print("BUNDLE " + str(bundle), flush=True)
    if not args.no_archive:
        archive_bundle(bundle, output.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
