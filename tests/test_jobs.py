import json
import tempfile
import unittest
import warnings
from datetime import datetime
from pathlib import Path

from wisperauto.jobs import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_POSTPROCESSING,
    STATUS_TRANSCRIBING,
    JobRecord,
    JobStore,
    recover_interrupted_records,
    utc_now,
)


class JobStoreTest(unittest.TestCase):
    def test_delete_removes_latest_record_and_compacts_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(Path(tmpdir) / "history.jsonl")
            first = JobRecord(id="one", source_name="one.mp3", source_path="/tmp/one.mp3")
            second = JobRecord(id="two", source_name="two.mp3", source_path="/tmp/two.mp3")
            store.append(first)
            store.append(second)
            store.update(first, progress=50)

            deleted = store.delete("one")
            remaining = store.latest()

            assert deleted is not None
            self.assertEqual(deleted.id, "one")
            self.assertEqual([record.id for record in remaining], ["two"])

    def test_latest_ignores_non_object_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.jsonl"
            valid = JobRecord(id="valid", source_name="valid.mp3", source_path="/tmp/valid.mp3")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                "[]\n"
                + json.dumps(valid.to_dict(), ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            store = JobStore(history_path)

            latest = store.latest()

            self.assertEqual([record.id for record in latest], ["valid"])

    def test_recover_interrupted_records_marks_stale_transcription_retryable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(Path(tmpdir) / "history.jsonl")
            stale = JobRecord(
                id="stale",
                source_name="stale.mp3",
                source_path="/tmp/stale.mp3",
                status=STATUS_TRANSCRIBING,
                progress=45,
                phase="transcribing",
                message="Transcription en cours.",
                eta_seconds=120,
            )
            done = JobRecord(
                id="done",
                source_name="done.mp3",
                source_path="/tmp/done.mp3",
                status=STATUS_DONE,
            )
            store.append(stale)
            store.append(done)

            recovered = recover_interrupted_records(store, store.latest())
            by_id = {record.id: record for record in recovered}

            self.assertEqual(by_id["stale"].status, STATUS_CANCELLED)
            self.assertEqual(by_id["stale"].eta_seconds, None)
            self.assertIn("interrompu", by_id["stale"].message)
            self.assertEqual(by_id["done"].status, STATUS_DONE)
            latest_by_id = {record.id: record for record in store.latest()}
            self.assertEqual(latest_by_id["stale"].status, STATUS_CANCELLED)

    def test_recover_interrupted_records_marks_stale_postprocess_retryable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(Path(tmpdir) / "history.jsonl")
            stale = JobRecord(
                id="stale",
                source_name="stale.mp3",
                source_path="/tmp/stale.mp3",
                status=STATUS_POSTPROCESSING,
                progress=94,
                phase="postprocess",
                message="Post-traitement intelligent en cours.",
                eta_seconds=None,
            )
            store.append(stale)

            recovered = recover_interrupted_records(store, store.latest())

            self.assertEqual(recovered[0].status, STATUS_CANCELLED)
            self.assertIn("interrompu", recovered[0].message)

    def test_utc_now_is_timezone_aware_without_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            value = utc_now()

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        self.assertIsNotNone(parsed.tzinfo)


if __name__ == "__main__":
    unittest.main()
