from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestration.formal_total_supervisor import source_snapshot


class FormalTotalSupervisorSnapshotTest(unittest.TestCase):
    def test_test_sources_are_frozen_and_change_tree_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("estimator", "runtime", "scripts", "orchestration", "tests"):
                (root / name).mkdir()
                (root / name / (name + "_source.py")).write_text(
                    "VALUE = %r\n" % name,
                    encoding="utf-8",
                )
            before = source_snapshot(root)
            test_name = "tests/tests_source.py"
            self.assertIn(test_name, before["files"])
            (root / test_name).write_text("VALUE = 'changed'\n", encoding="utf-8")
            after = source_snapshot(root)
            self.assertNotEqual(before["files"][test_name], after["files"][test_name])
            self.assertNotEqual(before["tree_sha256"], after["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
