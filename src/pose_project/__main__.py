"""统一命令入口；按命令延迟加载对应界面或脚本，避免无关模型阻塞启动。"""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "integrate": ("apps.integration_app", "Open the three-stage integration workbench"),
    "gate": ("apps.gate_app", "Open the gate capture workbench"),
    "reid": ("apps.reid_app", "Open the re-identification workbench"),
    "binding": ("apps.binding_app", "Open the passenger-to-tray workbench"),
    "simulate": ("binding.generate_binding_simulation", "Generate simulated messages"),
    "replay": ("binding.replay_binding", "Replay and evaluate binding messages"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pose", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    args, forwarded = parser.parse_known_args(argv)
    module = import_module("pose_project." + COMMANDS[args.command][0])
    original_argv = sys.argv
    try:
        sys.argv = [parser.prog + " " + args.command, *forwarded]
        return module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
