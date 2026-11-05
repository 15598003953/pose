"""Replay reception-ordered messages and optionally compare independent labels."""

import argparse
import json
from pathlib import Path

from pose_project.binding.passenger_tray_binding import PassengerTrayBinder, require_int, require_text
from pose_project.common.project_paths import CONFIG_DIR, DATA_DIR, RESULT_DIR


DEFAULT_CONFIG = CONFIG_DIR / "binding_config.json"
DEFAULT_INPUT = DATA_DIR / "binding_simulation" / "replay.jsonl"
DEFAULT_OUTPUT = RESULT_DIR / "binding_replay"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each record must be a JSON object")
            except ValueError as exc:
                raise ValueError("{}:{}: {}".format(path, number, exc)) from exc
            yield row


def replay(messages, config):
    binder = PassengerTrayBinder(config)
    decisions = []
    count = 0
    for count, message in enumerate(messages, 1):
        decisions.extend(binder.ingest(message))
    decisions.extend(binder.finish())
    return decisions, count


def event_key(row):
    return (
        require_text(row["roi_id"], "roi_id"),
        require_text(row["rfid"], "rfid"),
        require_int(row["rfid_received_at_ms"], "rfid_received_at_ms"),
    )


def evaluate(decisions, expected):
    actual = {}
    for row in decisions:
        key = event_key(row)
        if key in actual:
            raise ValueError("duplicate decision event: {}".format(key))
        actual[key] = row
    labels = {}
    for row in expected:
        key = event_key(row)
        require_text(row["passenger_id"], "passenger_id")
        if key in labels:
            raise ValueError("duplicate expected event: {}".format(key))
        labels[key] = row
    counts = dict(matched=0, wrong=0, unbound=0, deferred=0, missing=0, unexpected=0)
    cases = []
    for key, label in labels.items():
        decision = actual.get(key)
        if decision is None:
            outcome = "missing"
        elif decision["status"] != "bound":
            outcome = decision["status"]
        elif decision["passenger_id"] == label["passenger_id"]:
            outcome = "matched"
        else:
            outcome = "wrong"
        counts[outcome] += 1
        cases.append({
            "roi_id": key[0], "rfid": key[1], "rfid_received_at_ms": key[2],
            "scenario": label.get("scenario"),
            "expected_passenger_id": label["passenger_id"],
            "actual_passenger_id": decision["passenger_id"] if decision else None,
            "outcome": outcome,
        })
    for key, decision in actual.items():
        if key not in labels:
            counts["unexpected"] += 1
            cases.append({
                "roi_id": key[0], "rfid": key[1], "rfid_received_at_ms": key[2],
                "expected_passenger_id": None,
                "actual_passenger_id": decision["passenger_id"], "outcome": "unexpected",
            })
    return {
        "expected_events": len(labels),
        "decision_events": len(actual),
        **counts,
        "match_rate": counts["matched"] / len(labels) if labels else None,
        "all_events_match": bool(labels) and counts["matched"] == len(labels)
        and not counts["unexpected"],
        "cases": cases,
    }


def write_results(output, decisions, run_info, evaluation=None):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "decisions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "bindings.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in decisions:
            if row["binding"] is not None:
                handle.write(json.dumps(row["binding"], ensure_ascii=False) + "\n")
    reports = {"run.json": run_info}
    if evaluation is not None:
        reports["evaluation.json"] = evaluation
    for filename, report in reports.items():
        with (output / filename).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--expected", type=Path, help="Independent labels for posthoc evaluation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        with args.config.open(encoding="utf-8") as handle:
            config = json.load(handle)
        decisions, count = replay(read_jsonl(args.input), config)
        evaluation = None
        if args.expected is not None:
            evaluation = evaluate(decisions, read_jsonl(args.expected))
        run_info = {
            "input": str(args.input.resolve()),
            "expected": str(args.expected.resolve()) if args.expected else None,
            "message_count": count,
            "decision_count": len(decisions),
            "config": config,
        }
        output = write_results(args.output, decisions, run_info, evaluation)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    print("Replay results: {}".format(output))
    print("Messages: {}; decisions: {}".format(count, len(decisions)))
    if evaluation is not None:
        print("Matched: {matched}/{expected_events}; wrong: {wrong}; unbound: {unbound}; "
              "deferred: {deferred}; missing: {missing}; unexpected: {unexpected}".format(
                  **evaluation
              ))
        return 0 if evaluation["all_events_match"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
