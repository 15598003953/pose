"""Verify a copied EXE bundle without Conda or source paths in its environment."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_json(text):
    for index, character in enumerate(text):
        if character == "{":
            try:
                result, _ = json.JSONDecoder().raw_decode(text[index:])
                if isinstance(result, dict) and "ok" in result:
                    return result
            except ValueError:
                pass
    raise ValueError("No JSON report in executable output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-video", type=Path, required=True)
    parser.add_argument("--reid-video", type=Path, required=True)
    parser.add_argument("--rtsp-server", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Verification output must be new")
    output.mkdir(parents=True)
    portable = output / "portable" / "PoseIntegration"
    print("Copying release to " + str(portable), flush=True)
    shutil.copytree(args.bundle.resolve(), portable)
    manifest = json.loads((portable / "manifest.json").read_text(encoding="utf-8"))
    for index, entry in enumerate(manifest):
        file = portable / entry["path"]
        if file.stat().st_size != entry["size"] or sha256(file) != entry["sha256"]:
            raise RuntimeError("Manifest mismatch: " + entry["path"])
        if index and index % 1000 == 0:
            print("Verified {} files".format(index), flush=True)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("CONDA", "PYTHON", "_CE_"))
           and key.upper() not in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
                                  "QML2_IMPORT_PATH", "CUDA_PATH", "CUDA_HOME")}
    windows = Path(os.environ.get("WINDIR", "C:/Windows"))
    env.update(PATH=str(windows / "System32") + os.pathsep + str(windows),
               QT_QPA_PLATFORM="offscreen")
    executable = portable / "PoseIntegration.exe"
    checks = {}
    report = {"ok": False, "bundle": str(portable), "manifest_files": len(manifest),
              "conda_environment_removed": True, "checks": checks}

    def run(name, *arguments, extra_env=None, timeout=180):
        started = time.monotonic()
        result = subprocess.run([str(executable), *arguments], cwd=output,
            env=dict(env, **(extra_env or {})), capture_output=True, encoding="utf-8",
            errors="replace", timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
        (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        checks[name] = {"exit_code": result.returncode,
                        "seconds": round(time.monotonic() - started, 3)}
        print("{}: exit={} {:.1f}s".format(name, result.returncode,
                                          checks[name]["seconds"]), flush=True)
        if result.returncode:
            raise RuntimeError(name + " failed; see " + str(output / (name + ".log")))
        return result.stdout

    processes, handles = [], []
    try:
        info = report_json(run("dependencies", "--check"))
        checks["dependencies"]["report"] = info
        if not info.get("ok") or not info.get("packaged"):
            raise RuntimeError("Executable dependency check did not pass")
        for name, options in (("models_gpu", {}), ("models_cpu", {"CUDA_VISIBLE_DEVICES": "-1"})):
            info = report_json(run(name, "--check-models", extra_env=options))
            checks[name]["report"] = info
            if not info.get("ok") or (name == "models_cpu" and info.get("device") != "cpu"):
                raise RuntimeError(name + " did not produce the expected model report")
        info = report_json(run("binding", "--self-test"))
        checks["binding"]["report"] = info
        if not info.get("ok"):
            raise RuntimeError("Binding self-test failed")
        run("gui", "--duration", "2", timeout=60)
        media = output / "media"
        media.mkdir()
        for stage, source in (("gate", args.gate_video), ("reid", args.reid_video)):
            shutil.copy2(source, media / (stage + source.suffix))
        settings = json.loads((portable / "configs/integration_config.json").read_text(encoding="utf-8"))
        settings["gate"]["source"] = str(media / ("gate" + args.gate_video.suffix))
        settings["reid"]["source"] = str(media / ("reid" + args.reid_video.suffix))
        settings["output_dir"] = str(output / "sessions")
        for key in ("detector_weights", "reid_weights"):
            settings["reid"][key] = str((portable / "configs" / settings["reid"][key]).resolve())
        local_config = output / "file_config.json"
        local_config.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        run("local_video", "--config", str(local_config), "--headless", "--duration", "15")
        for path in (output / "sessions").glob("*/summary.json"):
            counts = json.loads(path.read_text(encoding="utf-8"))
            if counts.get("invalid") or counts.get("gate_errors") or counts.get("reid_errors"):
                raise RuntimeError("Local video reported invalid messages or worker errors")

        if args.rtsp_server and args.ffmpeg:
            import socket
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            server_config = output / "rtsp_server.yml"
            server_config.write_text(
                "rtsp: yes\nrtspAddress: 127.0.0.1:{}\nrtspTransports: [tcp]\n"
                "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\npaths:\n  gate:\n  tray:\n".format(port),
                encoding="utf-8")
            handle = (output / "rtsp_server.log").open("wb")
            handles.append(handle)
            processes.append(subprocess.Popen([str(args.rtsp_server.resolve()), str(server_config)],
                cwd=output, env=env, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW))
            time.sleep(.7)
            for stage in ("gate", "reid"):
                name = "gate" if stage == "gate" else "tray"
                handle = (output / (name + "_publisher.log")).open("wb")
                handles.append(handle)
                processes.append(subprocess.Popen([
                    str(args.ffmpeg.resolve()), "-hide_banner", "-loglevel", "error",
                    "-re", "-stream_loop", "-1", "-i", settings[stage]["source"],
                    "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p", "-g", "15", "-f", "rtsp", "-rtsp_transport", "tcp",
                    "rtsp://127.0.0.1:{}/{}".format(port, name)],
                    cwd=output, env=env, stdout=handle, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW))
                time.sleep(1)
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("Loopback RTSP fixture did not start")
            rtsp = deepcopy(settings)
            rtsp["gate"]["source"] = "rtsp://127.0.0.1:{}/gate".format(port)
            rtsp["reid"]["source"] = "rtsp://127.0.0.1:{}/tray".format(port)
            rtsp["output_dir"] = str(output / "rtsp_sessions")
            config = output / "rtsp_config.json"
            config.write_text(json.dumps(rtsp, ensure_ascii=False, indent=2), encoding="utf-8")
            run("dual_rtsp", "--config", str(config), "--headless", "--duration", "15")
            for path in (output / "rtsp_sessions").glob("*/summary.json"):
                counts = json.loads(path.read_text(encoding="utf-8"))
                if counts.get("invalid") or counts.get("gate_errors") or counts.get("reid_errors"):
                    raise RuntimeError("RTSP reported invalid messages or worker errors")
            messages = list((output / "rtsp_sessions").glob("*/messages.jsonl"))
            if not messages or not any(path.read_text(encoding="utf-8").strip() for path in messages):
                raise RuntimeError("Dual RTSP run produced no position messages")
        report["ok"] = True
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()
        (output / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
