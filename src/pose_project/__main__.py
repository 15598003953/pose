"""Unified entry point for the local project."""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "integrate": ("apps.integration_app", "Open the three-stage integration workbench"),
    "video": ("apps.pose_app", "Open the video analysis application"),
    "binding": ("apps.binding_app", "Open the passenger-to-tray workbench"),
    "simulate": ("binding.generate_binding_simulation", "Generate simulated messages"),
    "replay": ("binding.replay_binding", "Replay and evaluate binding messages"),
    "label": ("training.label_video_clips", "Label video clips"),
    "extract": ("training.extract_clip_pose_features", "Extract clip pose features"),
    "train": ("training.train_clip_classifier", "Train the clip classifier"),
    "offline": ("apps.run_put_luggage_offline", "Analyze a video offline"),
    "demo": ("apps.demo", "Run the pose estimation demo"),
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
