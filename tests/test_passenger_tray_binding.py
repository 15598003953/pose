from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from pose_project.binding.generate_binding_simulation import DEFAULT_CONFIG as SIMULATION_CONFIG, generate_dataset
from pose_project.binding.passenger_tray_binding import PassengerTrayBinder
from pose_project.binding.replay_binding import DEFAULT_CONFIG, evaluate, main, read_jsonl, replay, write_results


BASE = 1000000


def snapshot(at, inside=("person-1",), outside=(), roi="A", delay=0):
    center = 400 if roi == "A" else 1360
    people = [
        {"passenger_id": name, "bbox_xyxy": [center - 80, 255, center + 80, 495]}
        for name in inside
    ] + [
        {"passenger_id": name, "bbox_xyxy": [center + 170, 255, center + 330, 495]}
        for name in outside
    ]
    return {
        "topic": "simulation/binding/detections", "received_at_ms": BASE + at + delay,
        "payload": {"roi_id": roi, "captured_at_ms": BASE + at, "detections": people},
    }


def rfid(at, tag="tray-1", roi="A"):
    return {
        "topic": "simulation/binding/rfid", "received_at_ms": BASE + at,
        "payload": {"roi_id": roi, "rfid": tag},
    }


class PassengerTrayBindingTest(unittest.TestCase):
    def setUp(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            self.config = json.load(handle)

    def decisions_for(self, messages):
        messages = sorted(messages, key=lambda row: row["received_at_ms"])
        return replay(messages, self.config)[0]

    def continuous(self, end=6000, **kwargs):
        return [snapshot(at, **kwargs) for at in range(0, end + 1, 200)]

    def test_operation_position_filters_companions_and_short_passersby(self):
        messages = [
            snapshot(
                at, inside=("person-1", "passerby") if at >= 4000 else ("person-1",),
                outside=("companion",)
            )
            for at in range(0, 6001, 200)
        ]
        decision, = self.decisions_for(messages + [rfid(6020)])
        self.assertEqual("person-1", decision["passenger_id"])
        candidates = {row["passenger_id"]: row for row in decision["candidates"]}
        self.assertEqual(6000, candidates["person-1"]["dwell_ms"])
        self.assertEqual(2000, candidates["passerby"]["dwell_ms"])
        self.assertFalse(candidates["passerby"]["eligible"])
        self.assertNotIn("companion", candidates)

    def test_five_second_boundary_uses_elapsed_time_and_no_future_sample(self):
        decisions = self.decisions_for(self.continuous(5000) + [
            rfid(4999, "too-soon"), rfid(5000, "at-threshold")
        ])
        self.assertEqual("unbound", decisions[0]["status"])
        self.assertEqual("bound", decisions[1]["status"])
        self.assertEqual(4800, decisions[0]["candidates"][0]["dwell_ms"])
        self.assertEqual(5000, decisions[1]["candidates"][0]["dwell_ms"])

    def test_window_clips_intervals_and_does_not_extrapolate_past_last_sample(self):
        decision, = self.decisions_for(self.continuous(20000) + [rfid(20100)])
        self.assertEqual(9900, decision["candidates"][0]["dwell_ms"])
        self.assertEqual(100, decision["position_age_ms"])

    def test_short_departure_pauses_dwell_and_return_reuses_recent_history(self):
        messages = [
            snapshot(
                at, inside=() if 11000 <= at < 14000 else ("person-1",),
                outside=("person-1",) if 11000 <= at < 14000 else ()
            )
            for at in range(0, 15001, 200)
        ]
        away, returned = self.decisions_for(messages + [
            rfid(12000, "while-away"), rfid(15000, "after-return")
        ])
        self.assertEqual("unbound", away["status"])
        self.assertGreaterEqual(away["candidates"][0]["dwell_ms"], 5000)
        self.assertFalse(away["candidates"][0]["present_in_latest_snapshot"])
        self.assertEqual("person-1", returned["passenger_id"])
        self.assertEqual(6800, returned["candidates"][0]["dwell_ms"])

    def test_handover_excludes_previous_operator_still_in_outer_roi(self):
        messages = [
            snapshot(at, inside=("previous",) if at < 6000 else ("next",),
                     outside=("next",) if at < 6000 else ("previous",))
            for at in range(0, 11201, 200)
        ]
        early, later = self.decisions_for(messages + [
            rfid(6500, "new-person-not-ready"), rfid(11200, "new-person-ready")
        ])
        self.assertEqual("unbound", early["status"])
        previous = next(row for row in early["candidates"]
                        if row["passenger_id"] == "previous")
        self.assertGreaterEqual(previous["dwell_ms"], 5000)
        self.assertFalse(previous["eligible"])
        self.assertEqual("next", later["passenger_id"])

    def test_multiple_trays_preserve_passenger_history(self):
        decisions = self.decisions_for(self.continuous(6400) + [
            rfid(6000, "tray-1"), rfid(6200, "tray-2"), rfid(6400, "tray-3")
        ])
        self.assertEqual(["person-1"] * 3, [row["passenger_id"] for row in decisions])

    def test_binding_returns_region_stable_passenger_id_and_each_tray_rfid(self):
        passenger_id = "0123456789abcdef0123456789abcdef-2026-10-20"
        for roi in ("A", "B"):
            for tracking_id in (passenger_id, "{}-{}".format(roi, passenger_id)):
                with self.subTest(roi=roi, tracking_id=tracking_id):
                    decisions = self.decisions_for(
                        self.continuous(6400, inside=(tracking_id,), roi=roi) + [
                            rfid(6000, "tray-1", roi), rfid(6200, "tray-2", roi),
                            rfid(6400, "tray-3", roi),
                        ]
                    )
                    self.assertEqual([
                        {"roi_id": roi, "passenger_id": passenger_id, "rfid": tag}
                        for tag in ("tray-1", "tray-2", "tray-3")
                    ], [row["binding"] for row in decisions])
                    self.assertEqual([tracking_id] * 3, [
                        row["passenger_id"] for row in decisions
                    ])

    def test_simultaneous_rois_have_independent_history(self):
        decisions = self.decisions_for(
            self.continuous(inside=("a-person",), roi="A")
            + self.continuous(inside=("b-person",), roi="B")
            + [rfid(6000, "a-tray", "A"), rfid(6000, "b-tray", "B")]
        )
        self.assertEqual(["a-person", "b-person"], [
            row["passenger_id"] for row in decisions
        ])

    def test_late_pre_event_snapshot_changes_decision_only_within_deadline(self):
        for delay, status in ((400, "unbound"), (600, "unbound"), (601, "bound")):
            with self.subTest(delay=delay):
                decision, = self.decisions_for(
                    self.continuous(5800)
                    + [rfid(6000), snapshot(6000, inside=(), delay=delay)]
                )
                self.assertEqual(status, decision["status"])
                self.assertEqual(BASE + 6600, decision["decided_at_ms"])

    def test_older_snapshot_arriving_later_does_not_replace_newer_capture(self):
        decision, = self.decisions_for(
            self.continuous(5600)
            + [snapshot(5800, delay=500), snapshot(6000, inside=()), rfid(6000)]
        )
        self.assertEqual("unbound", decision["status"])
        self.assertEqual(BASE + 6000, decision["latest_snapshot_at_ms"])

    def test_waiting_never_uses_post_event_positions(self):
        decision, = self.decisions_for(
            self.continuous()
            + [snapshot(at, inside=("new-arrival",)) for at in (6200, 6400, 6600)]
            + [rfid(6000)]
        )
        self.assertEqual("person-1", decision["passenger_id"])
        self.assertEqual(BASE + 6000, decision["reference_at_ms"])
        self.assertEqual(BASE + 6000, decision["latest_snapshot_at_ms"])

    def test_configurable_rfid_offset_moves_reference_backwards(self):
        self.config["binding_parameters"]["rfid_time_offset_ms"] = 1000
        messages = self.continuous(5000) + [
            snapshot(at, inside=("next-person",)) for at in range(5200, 6001, 200)
        ]
        decision, = self.decisions_for(messages + [rfid(6000)])
        self.assertEqual("person-1", decision["passenger_id"])
        self.assertEqual(BASE + 5000, decision["reference_at_ms"])

    def test_empty_missing_and_stale_positions_never_reuse_old_presence(self):
        cases = [
            ([rfid(6000)], "no_position_data"),
            (self.continuous(5000) + [rfid(6000)], "stale_position_data"),
            (self.continuous(5800) + [snapshot(6000, inside=()), rfid(6000)],
             "no_eligible_candidate"),
        ]
        for messages, reason in cases:
            with self.subTest(reason=reason):
                decision, = self.decisions_for(messages)
                self.assertEqual("unbound", decision["status"])
                self.assertIsNone(decision["passenger_id"])
                self.assertIsNone(decision["binding"])
                self.assertEqual(reason, decision["reason"])

    def test_long_data_gaps_do_not_count_as_continuous_dwell(self):
        messages = [snapshot(at) for at in range(0, 10001, 1000)]
        decision, = self.decisions_for(messages + [rfid(10000)])
        self.assertEqual("unbound", decision["status"])
        self.assertEqual(0, decision["candidates"][0]["dwell_ms"])

    def test_duplicate_snapshots_cannot_inflate_dwell(self):
        self.config["binding_parameters"]["min_dwell_ms"] = 7000
        messages = [
            row for original in self.continuous()
            for row in (original, deepcopy(original), deepcopy(original))
        ]
        decision, = self.decisions_for(messages + [rfid(6000)])
        self.assertEqual("unbound", decision["status"])
        self.assertEqual(6000, decision["candidates"][0]["dwell_ms"])

    def test_multiple_eligible_people_are_recorded_without_selecting_one(self):
        decision, = self.decisions_for(
            self.continuous(inside=("person-1", "person-2")) + [rfid(6000)]
        )
        self.assertEqual("deferred", decision["status"])
        self.assertEqual("multiple_candidates", decision["reason"])
        self.assertIsNone(decision["passenger_id"])
        self.assertIsNone(decision["binding"])
        self.assertTrue(all(row["eligible"] for row in decision["candidates"]))

    def test_history_is_bounded_during_a_long_running_stream(self):
        binder = PassengerTrayBinder(self.config)
        for message in self.continuous(60000):
            self.assertEqual([], binder.ingest(message))
        self.assertLessEqual(len(binder.history["A"]), 60)
        binder.ingest(rfid(60000))
        decision, = binder.finish()
        self.assertEqual("person-1", decision["passenger_id"])

    def test_explicit_timer_can_finalize_without_another_message(self):
        binder = PassengerTrayBinder(self.config)
        for message in self.continuous() + [rfid(6000)]:
            self.assertEqual([], binder.ingest(message))
        self.assertEqual([], binder.advance(BASE + 6599))
        decision, = binder.advance(BASE + 6600)
        self.assertEqual("bound", decision["status"])
        self.assertEqual([], binder.finish())

    def test_invalid_config_and_bad_messages_fail_with_explicit_errors(self):
        for key, value in (
            ("min_dwell_ms", 10001), ("binding_wait_ms", -1),
            ("max_sample_gap_ms", 0), ("max_position_age_ms", True)
        ):
            config = deepcopy(self.config)
            config["binding_parameters"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PassengerTrayBinder(config)
        config = deepcopy(self.config)
        config["regions"]["A"]["operation_roi_xyxy"] = [900, 0, 1000, 100]
        with self.assertRaisesRegex(ValueError, "inside"):
            PassengerTrayBinder(config)
        messages = [
            snapshot(0, inside=("same", "same")),
            snapshot(0, roi="C"),
            dict(snapshot(0), received_at_ms=BASE - 1),
            dict(snapshot(0), topic="unknown"),
        ]
        bad_box = snapshot(0)
        bad_box["payload"]["detections"][0]["bbox_xyxy"][0] = float("nan")
        messages.append(bad_box)
        for message in messages:
            with self.subTest(message=message), self.assertRaises(ValueError):
                PassengerTrayBinder(self.config).ingest(message)
        binder = PassengerTrayBinder(self.config)
        binder.ingest(snapshot(0))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            binder.ingest(snapshot(0, inside=()))
        binder.ingest(snapshot(200))
        with self.assertRaisesRegex(ValueError, "ordered"):
            binder.ingest(snapshot(0))
        with self.assertRaisesRegex(ValueError, "backwards"):
            binder.advance(BASE)


class BindingReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            cls.config = json.load(handle)
        with SIMULATION_CONFIG.open(encoding="utf-8") as handle:
            simulation = json.load(handle)
        cls.messages, cls.expected = generate_dataset(simulation)
        cls.decisions, cls.count = replay(cls.messages, cls.config)

    def test_all_nine_independent_labels_match(self):
        report = evaluate(self.decisions, self.expected)
        self.assertEqual(3609, self.count)
        self.assertEqual(9, len(self.decisions))
        self.assertTrue(report["all_events_match"])
        self.assertEqual(9, report["matched"])
        self.assertEqual(0, report["wrong"])
        self.assertEqual(0, report["unbound"])
        for decision in self.decisions:
            eligible = [row for row in decision["candidates"] if row["eligible"]]
            self.assertEqual(1, len(eligible))
            self.assertEqual(decision["passenger_id"], eligible[0]["passenger_id"])
            binding = decision["binding"]
            self.assertEqual({"roi_id", "passenger_id", "rfid"}, set(binding))
            self.assertEqual(decision["roi_id"], binding["roi_id"])
            self.assertEqual(decision["rfid"], binding["rfid"])
            self.assertEqual(
                decision["passenger_id"][2:], binding["passenger_id"]
            )
            self.assertRegex(
                binding["passenger_id"], r"^[0-9a-f]{32}-2026-10-20$"
            )

    def test_labels_are_used_after_binding_and_wrong_labels_are_reported(self):
        expected = deepcopy(self.expected)
        expected[0]["passenger_id"] = "intentionally-wrong-label"
        report = evaluate(self.decisions, expected)
        self.assertEqual(8, report["matched"])
        self.assertEqual(1, report["wrong"])
        self.assertFalse(report["all_events_match"])
        self.assertNotEqual(expected[0]["passenger_id"], self.decisions[0]["passenger_id"])

    def test_evaluation_counts_missing_extra_unbound_and_deferred_events(self):
        decisions = deepcopy(self.decisions[:4])
        expected = deepcopy(self.expected[:4])
        decisions[0].update(status="unbound", passenger_id=None)
        decisions[1].update(status="deferred", passenger_id=None)
        decisions[2]["passenger_id"] = "wrong"
        decisions[3]["rfid"] = "unexpected-tray"
        report = evaluate(decisions, expected)
        for key in ("unbound", "deferred", "wrong", "missing", "unexpected"):
            self.assertEqual(1, report[key])
        self.assertEqual(0, report["matched"])
        with self.assertRaisesRegex(ValueError, "duplicate decision"):
            evaluate([self.decisions[0], self.decisions[0]], self.expected)
        with self.assertRaisesRegex(ValueError, "duplicate expected"):
            evaluate(self.decisions, [self.expected[0], self.expected[0]])

    def test_cli_writes_utf8_results_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            input_path, labels_path = folder / "input.jsonl", folder / "labels.jsonl"
            config_path, output = folder / "config.json", folder / "output"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            for path, rows in ((input_path, self.messages), (labels_path, self.expected)):
                path.write_text("".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ), encoding="utf-8")
            args = [
                "--config", str(config_path), "--input", str(input_path),
                "--expected", str(labels_path), "--output", str(output)
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(args))
            saved = list(read_jsonl(output / "decisions.jsonl"))
            self.assertEqual(self.decisions, saved)
            self.assertEqual(
                [row["binding"] for row in self.decisions],
                list(read_jsonl(output / "bindings.jsonl")),
            )
            report = json.loads((output / "evaluation.json").read_bytes().decode("utf-8"))
            self.assertEqual(9, report["matched"])
            run_info = json.loads((output / "run.json").read_bytes().decode("utf-8"))
            self.assertEqual(self.config, run_info["config"])
            self.assertNotIn("tracks", run_info["config"])
            original = (output / "decisions.jsonl").read_bytes()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
            self.assertEqual(2, raised.exception.code)
            self.assertEqual(original, (output / "decisions.jsonl").read_bytes())
            labels = deepcopy(self.expected)
            labels[0]["passenger_id"] = "wrong"
            labels_path.write_text("".join(
                json.dumps(row) + "\n" for row in labels
            ), encoding="utf-8")
            args[-1] = str(folder / "mismatch")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(1, main(args))

    def test_invalid_jsonl_reports_path_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("{}\n[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
                list(read_jsonl(path))

    def test_bindings_file_contains_only_successful_bindings(self):
        messages = [
            snapshot(at, inside=("person-1",) if at <= 6000
                     else ("person-1", "person-2"))
            for at in range(0, 12001, 200)
        ] + [rfid(4000, "too-soon"), rfid(6000, "bound-tray"),
             rfid(12000, "ambiguous-tray")]
        decisions, count = replay(
            sorted(messages, key=lambda row: row["received_at_ms"]), self.config
        )
        self.assertEqual(["unbound", "bound", "deferred"], [
            row["status"] for row in decisions
        ])
        with tempfile.TemporaryDirectory() as tmp:
            write_results(tmp, decisions, {"message_count": count})
            self.assertEqual([
                {"roi_id": "A", "passenger_id": "person-1", "rfid": "bound-tray"}
            ], list(read_jsonl(Path(tmp) / "bindings.jsonl")))


if __name__ == "__main__":
    unittest.main()
