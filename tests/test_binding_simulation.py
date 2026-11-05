from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from pose_project.binding.generate_binding_simulation import (
    DEFAULT_CONFIG,
    generate_dataset,
    validate_config,
    write_dataset,
)


class BindingSimulationTest(unittest.TestCase):
    def setUp(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.messages, self.expected = generate_dataset(self.config)
        self.detections = [
            row for row in self.messages
            if row["topic"] == self.config["topics"]["detections"]
        ]
        self.rfid = [
            row for row in self.messages
            if row["topic"] == self.config["topics"]["rfid"]
        ]
        self.start_ms = int(
            datetime.fromisoformat(self.config["start_time"]).timestamp() * 1000
        )

    def snapshot(self, region_id, offset_ms):
        return next(
            row["payload"] for row in self.detections
            if row["payload"]["roi_id"] == region_id
            and row["payload"]["captured_at_ms"] == self.start_ms + offset_ms
        )

    def expected_id(self, rfid):
        return next(row["passenger_id"] for row in self.expected if row["rfid"] == rfid)

    def person_box(self, snapshot, passenger_id):
        return next(
            person["bbox_xyxy"] for person in snapshot["detections"]
            if person["passenger_id"] == passenger_id
        )

    def in_operation_area(self, region_id, box):
        x1, y1, x2, y2 = self.config["regions"][region_id]["operation_roi_xyxy"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def test_default_counts_and_time_order(self):
        self.assertEqual(3600, len(self.detections))
        self.assertEqual(9, len(self.rfid))
        self.assertEqual(9, len(self.expected))
        arrival_times = [row["received_at_ms"] for row in self.messages]
        self.assertEqual(sorted(arrival_times), arrival_times)
        for region_id in self.config["regions"]:
            times = sorted(
                row["payload"]["captured_at_ms"] for row in self.detections
                if row["payload"]["roi_id"] == region_id
            )
            self.assertTrue(all(b - a == 200 for a, b in zip(times, times[1:])))

    def test_same_seed_is_reproducible(self):
        messages, expected = generate_dataset(deepcopy(self.config))
        self.assertEqual(self.messages, messages)
        self.assertEqual(self.expected, expected)

    def test_id_date_is_fixed_at_first_appearance_across_midnight(self):
        config = deepcopy(self.config)
        config["start_time"] = "2026-10-20T23:58:00+08:00"
        messages, expected = generate_dataset(config)
        labels = {row["rfid"]: row["passenger_id"] for row in expected}
        previous_id = labels["RFID-A-0001"]
        self.assertTrue(previous_id.endswith("-2026-10-20"))
        self.assertTrue(labels["RFID-A-0005"].endswith("-2026-10-21"))
        self.assertEqual(previous_id, labels["RFID-A-0004"])
        midnight_ms = int(
            datetime.fromisoformat("2026-10-21T00:00:00+08:00").timestamp() * 1000
        )
        self.assertTrue(any(
            row["topic"] == config["topics"]["detections"]
            and row["payload"]["captured_at_ms"] >= midnight_ms
            and any(
                person["passenger_id"] == previous_id
                for person in row["payload"]["detections"]
            )
            for row in messages
        ))

    def test_empty_snapshots_continue_after_a_track_ends(self):
        config = deepcopy(self.config)
        config["duration_ms"] = 2000
        config["tracks"] = [config["tracks"][0]]
        config["tracks"][0]["segments"] = [
            {"start_ms": 0, "end_ms": 1000, "center": [400, 375]}
        ]
        config["push_events"] = []
        messages, expected = generate_dataset(config)
        self.assertEqual([], expected)
        self.assertEqual(20, len(messages))
        for row in messages:
            payload = row["payload"]
            if payload["roi_id"] == "B" or payload["captured_at_ms"] >= self.start_ms + 1000:
                self.assertEqual([], payload["detections"])
            else:
                self.assertEqual(1, len(payload["detections"]))

    def test_ids_are_stable_unique_and_have_region_and_date(self):
        ids = {
            person["passenger_id"]
            for row in self.detections for person in row["payload"]["detections"]
        }
        self.assertEqual(len(self.config["tracks"]), len(ids))
        for row in self.detections:
            for person in row["payload"]["detections"]:
                self.assertRegex(
                    person["passenger_id"],
                    "^{}-[0-9a-f]{{32}}-2026-10-20$".format(row["payload"]["roi_id"]),
                )
        self.assertEqual(self.expected_id("RFID-A-0001"), self.expected_id("RFID-A-0004"))
        self.assertNotEqual(
            self.expected_id("RFID-A-0004"), self.expected_id("RFID-A-0005")
        )

    def test_brief_absence_preserves_identity_and_history_is_not_a_payload_label(self):
        passenger_id = self.expected_id("RFID-A-0003")
        before = self.person_box(self.snapshot("A", 96000), passenger_id)
        away = self.person_box(self.snapshot("A", 98000), passenger_id)
        returned = self.person_box(self.snapshot("A", 101000), passenger_id)
        self.assertTrue(self.in_operation_area("A", before))
        self.assertFalse(self.in_operation_area("A", away))
        self.assertTrue(self.in_operation_area("A", returned))
        self.assertEqual(self.expected_id("RFID-A-0001"), passenger_id)
        for row in self.detections:
            self.assertEqual(
                {"roi_id", "captured_at_ms", "detections"}, set(row["payload"])
            )
            for person in row["payload"]["detections"]:
                self.assertEqual({"passenger_id", "bbox_xyxy"}, set(person))

    def test_handover_keeps_old_person_in_outer_roi(self):
        snapshot = self.snapshot("A", 192000)
        previous = self.person_box(snapshot, self.expected_id("RFID-A-0004"))
        current = self.person_box(snapshot, self.expected_id("RFID-A-0005"))
        self.assertFalse(self.in_operation_area("A", previous))
        self.assertTrue(self.in_operation_area("A", current))
        self.assertGreaterEqual(len(snapshot["detections"]), 4)

    def test_rfid_has_no_hardware_timestamp_or_expected_identity(self):
        labels = {row["rfid"]: row for row in self.expected}
        for row in self.rfid:
            self.assertEqual({"roi_id", "rfid"}, set(row["payload"]))
            label = labels[row["payload"]["rfid"]]
            self.assertEqual(row["received_at_ms"], label["rfid_received_at_ms"])
            delay = row["received_at_ms"] - label["simulated_push_at_ms"]
            self.assertGreaterEqual(delay, self.config["photoelectric_delay_ms"][0])
            self.assertLessEqual(delay, self.config["photoelectric_delay_ms"][1])

    def test_detection_delay_can_deliver_older_frames_later(self):
        a_rows = [
            row for row in self.detections if row["payload"]["roi_id"] == "A"
        ]
        capture_times = [row["payload"]["captured_at_ms"] for row in a_rows]
        self.assertNotEqual(sorted(capture_times), capture_times)
        for row in self.detections:
            delay = row["received_at_ms"] - row["payload"]["captured_at_ms"]
            self.assertGreaterEqual(delay, self.config["detection_delay_ms"][0])
            self.assertLessEqual(delay, self.config["detection_delay_ms"][1])

    def test_binding_thresholds_do_not_determine_ground_truth(self):
        config = deepcopy(self.config)
        config["binding_parameters"]["min_dwell_ms"] = 9000
        messages, expected = generate_dataset(config)
        self.assertEqual(self.messages, messages)
        self.assertEqual(self.expected, expected)

    def test_invalid_timing_geometry_and_identity_are_rejected(self):
        mutations = [
            lambda c: c.update(update_interval_ms=0),
            lambda c: c.update(detection_delay_ms=[450, 20]),
            lambda c: c.update(start_time="2026-10-20T09:00:00"),
            lambda c: c["binding_parameters"].update(min_dwell_ms=10001),
            lambda c: c["regions"]["A"].update(operation_roi_xyxy=[0, 0, 2000, 500]),
            lambda c: c["tracks"][1].update(name=c["tracks"][0]["name"]),
            lambda c: c["push_events"][0].update(passenger="b_operator_1"),
            lambda c: c["push_events"][1].update(rfid=c["push_events"][0]["rfid"]),
        ]
        for mutate in mutations:
            config = deepcopy(self.config)
            mutate(config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_written_logs_roundtrip_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "simulation"
            path, counts = write_dataset(
                output, self.config, self.messages, self.expected
            )
            with (path / "replay.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual(self.messages, [json.loads(line) for line in handle])
            with (path / "manifest.json").open(encoding="utf-8") as handle:
                self.assertEqual(counts, json.load(handle)["counts"])
            original = (path / "expected_bindings.jsonl").read_bytes()
            with self.assertRaises(ValueError):
                write_dataset(output, self.config, self.messages, self.expected)
            self.assertEqual(original, (path / "expected_bindings.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
