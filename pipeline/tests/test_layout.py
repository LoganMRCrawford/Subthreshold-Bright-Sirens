"""Synthetic legacy-layout safety tests; no real queue, paths or JAX."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "archive/layout.py"
SPEC = importlib.util.spec_from_file_location("legacy_layout_tests", SOURCE)
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)


class LayoutTests(unittest.TestCase):
    def make_tree(self, root):
        (root / "archive").mkdir()
        for name in layout.NAMES:
            directory = root / "archive" / name
            directory.mkdir()
            (directory / "data").write_bytes(b"preserved bytes")
            (directory / "reference").symlink_to("data")

    def identities(self, root, archived=True):
        base = root / "archive" if archived else root
        return {name: layout.tree_identity(base / name) for name in layout.NAMES}

    def test_archived_status_is_read_only_without_slurm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            before = self.identities(root)
            with patch.object(layout.subprocess, "run", side_effect=AssertionError("No Slurm query")):
                result = layout.layout_status(root)
            self.assertEqual(result["layout_mode"], "physically_archived")
            self.assertFalse(result["path_changes_performed"])
            self.assertEqual(self.identities(root), before)

    def test_source_only_checkout_reports_not_installed_without_slurm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            before = list(root.iterdir())
            with patch.object(layout.subprocess, "run", side_effect=AssertionError("No Slurm query")):
                result = layout.layout_status(root)
            self.assertEqual(result["layout_mode"], "not_installed")
            self.assertEqual(set(result["states"].values()), {"not_installed"})
            self.assertFalse(result["legacy_data_available"])
            self.assertFalse(result["path_changes_performed"])
            self.assertEqual(list(root.iterdir()), before)

    def test_missing_trees_cannot_authorize_applied_migration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "archive").mkdir()
            for action in ("archive", "restore"):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    layout.migrate(root, action, "", root / f"{action}_receipt")
                self.assertFalse((root / f"{action}_receipt").exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["archive"])

    def test_missing_and_populated_trees_are_mixed_and_not_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "archive").mkdir()
            (root / "archive" / layout.NAMES[0]).mkdir()
            result = layout.layout_status(root)
            self.assertEqual(result["layout_mode"], "mixed")
            self.assertFalse(result["legacy_data_available"])
            with self.assertRaises(ValueError):
                layout.describe(root, "restore")

    def test_complete_restore_archive_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            before = self.identities(root)
            layout.migrate(root, "restore", "", root / "restore_receipt")
            self.assertEqual(layout.layout_status(root)["layout_mode"], "original_paths")
            self.assertEqual(self.identities(root, archived=False), before)
            for name in layout.NAMES:
                self.assertEqual((root / "archive" / name).readlink(), Path("..") / name)
            layout.migrate(root, "archive", "", root / "archive_receipt")
            self.assertEqual(self.identities(root), before)
            self.assertTrue(all(not (root / name).exists() for name in layout.NAMES))

    def test_mixed_status_does_not_attempt_repair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            name = layout.NAMES[0]
            (root / "archive" / name).rename(root / name)
            (root / "archive" / name).symlink_to(Path("..") / name, target_is_directory=True)
            self.assertEqual(layout.layout_status(root)["layout_mode"], "mixed")
            with self.assertRaises(ValueError):
                layout.describe(root, "restore")

    def test_unexpected_path_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            (root / layout.NAMES[0]).mkdir()
            with self.assertRaises(ValueError):
                layout.layout_status(root)

    def test_active_queue_prevents_any_move(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            before = self.identities(root)
            with self.assertRaises(ValueError):
                layout.migrate(root, "restore", "123|RUNNING", root / "operation")
            self.assertFalse((root / "operation").exists())
            self.assertEqual(self.identities(root), before)

    def test_rename_root_mtime_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            before = self.identities(root)
            destinations = {name: root / "archive" / name for name in layout.NAMES}
            for directory in destinations.values():
                info = directory.lstat()
                layout.os.utime(directory, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
            restored = layout.preserve_renamed_root_times(destinations, before, self.identities(root))
            self.assertEqual(restored, list(layout.NAMES))
            self.assertEqual(self.identities(root), before)

    def test_file_changes_and_wrong_root_inode_still_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            before = self.identities(root)
            destinations = {name: root / "archive" / name for name in layout.NAMES}
            for path, field in (("data", "mtime_ns"), ("data", "bytes"), (".", "inode")):
                after = copy.deepcopy(before)
                after[layout.NAMES[1]][path][field] += 1
                with self.assertRaises(ValueError):
                    layout.preserve_renamed_root_times(destinations, before, after)
            self.assertEqual(self.identities(root), before)

    def test_injected_second_rename_failure_rolls_back_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            rename = Path.rename

            def failing_rename(source, target):
                if source == root / "archive" / layout.NAMES[1] and Path(target) == root / layout.NAMES[1]:
                    raise OSError("Injected second move failure")
                return rename(source, target)

            with patch.object(Path, "rename", failing_rename):
                with self.assertRaises(OSError):
                    layout.migrate(root, "restore", "", root / "operation")
            self.assertEqual(layout.layout_status(root)["layout_mode"], "physically_archived")
            failure = json.loads((root / "operation/failure.json").read_text())
            self.assertEqual(failure["rollback_errors"], [])
            for name in layout.NAMES:
                self.assertEqual((root / "archive" / name / "data").read_bytes(), b"preserved bytes")

    def test_status_cli_does_not_query_queue(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tree(root)
            with patch.object(layout, "__file__", str(root / "archive/layout.py")), \
                 patch("sys.argv", ["layout.py", "status"]), \
                 patch.object(layout.subprocess, "run", side_effect=AssertionError("No queue query")), \
                 redirect_stdout(io.StringIO()) as output:
                layout.main()
            self.assertEqual(json.loads(output.getvalue())["layout_mode"], "physically_archived")

    def test_status_cli_source_only_checkout_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(layout, "__file__", str(root / "archive/layout.py")), \
                 patch("sys.argv", ["layout.py", "status"]), \
                 patch.object(layout.subprocess, "run", side_effect=AssertionError("No queue query")), \
                 redirect_stdout(io.StringIO()) as output:
                layout.main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["layout_mode"], "not_installed")
            self.assertFalse(result["path_changes_performed"])
            self.assertEqual(list(root.iterdir()), [])

    def test_status_apply_is_refused(self):
        with patch("sys.argv", ["layout.py", "status", "--apply"]), \
             patch.object(layout, "migrate", side_effect=AssertionError("Never migrate")), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                layout.main()
        self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()