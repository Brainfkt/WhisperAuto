import tempfile
import unittest
from pathlib import Path

from wisperauto.jobs import JobRecord, JobStore


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

            self.assertEqual(deleted.id, "one")
            self.assertEqual([record.id for record in remaining], ["two"])


if __name__ == "__main__":
    unittest.main()
