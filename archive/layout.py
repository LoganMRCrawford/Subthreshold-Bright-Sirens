"""Preview/archive/restore legacy directory layout without rewriting any run file."""

import argparse
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

NAMES = ("LC_grb_informed_pipeline", "LC_noise_improved", "SFM_baseline_GW170817_GRB_extended")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def present(path):
    return path.exists() or path.is_symlink()


def state(root, name):
    old, archived = root / name, root / "archive" / name
    original_directory = old.is_dir() and not old.is_symlink()
    original_link = archived.is_symlink() and archived.readlink() == Path("..") / name
    if not present(old) and not present(archived):
        return "not_installed"
    if original_directory and original_link:
        return "original_paths"
    if not present(old) and archived.is_dir() and not archived.is_symlink():
        return "physically_archived"
    raise ValueError(f"Unexpected or partially migrated layout for {name}; manual reconciliation required")


def tree_identity(directory):
    """Record same-filesystem inode identities; never follow directory symlinks."""
    records = {}
    paths = [directory]
    for parent, directories, files in os.walk(directory, followlinks=False):
        paths.extend(Path(parent) / name for name in directories + files)
    for path in paths:
        info = path.lstat()
        item = {"device": info.st_dev, "inode": info.st_ino,
                "mode": info.st_mode, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        if stat.S_ISLNK(info.st_mode):
            item["symlink_target"] = str(path.readlink())
        records[str(path.relative_to(directory))] = item
    return records


def preserve_renamed_root_times(destinations, before, after):
    """Some filesystems change the moved root mtime; reject every other change."""
    for name in NAMES:
        candidate = {path: dict(info) for path, info in after[name].items()}
        candidate["."]["mtime_ns"] = before[name]["."]["mtime_ns"]
        if candidate != before[name]:
            raise ValueError(f"Tree identity changed beyond rename-root timestamp: {name}")
    restored = []
    for name in NAMES:
        if after[name]["."]["mtime_ns"] != before[name]["."]["mtime_ns"]:
            directory = destinations[name]
            info = directory.lstat()
            os.utime(directory, ns=(info.st_atime_ns, before[name]["."]["mtime_ns"]), follow_symlinks=False)
            restored.append(name)
    return restored


def describe(root, action):
    states = {name: state(root, name) for name in NAMES}
    required = "original_paths" if action == "archive" else "physically_archived"
    if any(value != required for value in states.values()):
        raise ValueError(f"{action} requires all three trees in state {required}; found {states}")
    pairs = [(root / name, root / "archive" / name) if action == "archive"
             else (root / "archive" / name, root / name) for name in NAMES]
    if any(source.stat().st_dev != target.parent.stat().st_dev for source, target in pairs):
        raise ValueError("Only same-filesystem metadata renames are allowed; no copying/deleting across devices")
    return {"action": action, "states_before": states,
            "moves": [{"source": str(source), "target": str(target)} for source, target in pairs],
            "rewrites_checkpoint_or_manifest_bytes": False,
            "warning": "Old frozen jobs require ORIGINAL paths. Restore the complete original layout before legacy verification/resume; do not rewrite their identities."}


def layout_status(root):
    """Describe populated layouts or a source-only checkout without Slurm."""
    states = {name: state(root, name) for name in NAMES}
    modes = set(states.values())
    return {"layout_mode": next(iter(modes)) if len(modes) == 1 else "mixed",
            "states": states, "active_workflow": "pipeline/run.py",
            "legacy_trees_local_only": True, "path_changes_performed": False,
            "legacy_data_available": all(value != "not_installed" for value in states.values()),
            "note": "not_installed is expected in a source-only checkout; no local historical trees were distributed or created.",
            "warning": "Do not relocate legacy paths while jobs are queued or running. Frozen launchers compare absolute paths; this is a software compatibility constraint, not a property of the numerical state."}


def migrate(root, action, queue_text, record_directory):
    root = root.resolve()
    if queue_text.strip():
        raise ValueError("Require an empty user Slurm queue before changing historical paths")
    plan = describe(root, action)
    record_directory.mkdir(parents=True, exist_ok=False)
    before = {name: tree_identity(root / name if action == "archive" else root / "archive" / name) for name in NAMES}
    write(record_directory / "before.json", {"created_utc": now(), "plan": plan, "tree_identities": before})
    completed = []
    try:
        for name in NAMES:
            old, archived = root / name, root / "archive" / name
            if action == "archive":
                if not archived.is_symlink() or archived.readlink() != Path("..") / name:
                    raise ValueError("Archive reference changed during relocation")
                archived.unlink()  # Remove ONLY the reference; the directory data is renamed intact.
                try:
                    old.rename(archived)
                except Exception:
                    archived.symlink_to(Path("..") / name, target_is_directory=True)
                    raise
            else:
                if present(old):
                    raise ValueError("Refusing to overwrite an original-path object")
                archived.rename(old)
                try:
                    archived.symlink_to(Path("..") / name, target_is_directory=True)
                except Exception:
                    old.rename(archived)
                    raise
            completed.append(name)
            write(record_directory / f"moved_{len(completed)}.json", {"created_utc": now(), "name": name, "action": action})
        destinations = {name: root / "archive" / name if action == "archive" else root / name for name in NAMES}
        after = {name: tree_identity(directory) for name, directory in destinations.items()}
        restored_times = preserve_renamed_root_times(destinations, before, after)
        if restored_times:
            after = {name: tree_identity(directory) for name, directory in destinations.items()}
        if before != after:
            raise ValueError("Tree inode, byte-count, mode, symlink or modification-time identity changed")
        write(record_directory / "result.json", {"created_utc": now(), "status": "completed",
              "action": action, "all_tree_inode_identities_unchanged": True,
              "rename_root_mtime_restored": restored_times,
              "new_states": {name: state(root, name) for name in NAMES}})
    except Exception as error:
        rollback_errors = []
        for name in reversed(completed):
            try:
                old, archived = root / name, root / "archive" / name
                if action == "archive":
                    if present(old):
                        raise ValueError("Cannot roll back over a new original-path object")
                    archived.rename(old)
                    archived.symlink_to(Path("..") / name, target_is_directory=True)
                else:
                    if not archived.is_symlink():
                        raise ValueError("Cannot roll back over a new archive object")
                    archived.unlink()
                    old.rename(archived)
            except Exception as rollback_error:
                rollback_errors.append({"name": name, "error": str(rollback_error)})
        write(record_directory / "failure.json", {"created_utc": now(), "error": str(error),
                                                 "rollback_errors": rollback_errors})
        raise
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "archive", "restore"), help="status is read-only; restore reinstates original paths and archive references")
    parser.add_argument("--apply", action="store_true", help="Actually rename directories; requires an empty user Slurm queue")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "status":
        if args.apply:
            parser.error("status is read-only and cannot be combined with --apply")
        print(json.dumps(layout_status(root), indent=2))
        return
    print(json.dumps(describe(root, args.action), indent=2))
    if not args.apply:
        print("PREVIEW ONLY. No paths or files changed.")
        return
    queue = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%j|%T"],
                           capture_output=True, text=True, check=True).stdout
    record = root / "archive/layout_operations" / datetime.now(timezone.utc).strftime(args.action+"_%Y%m%dT%H%M%S%fZ")
    migrate(root, args.action, queue, record)
    print("LAYOUT_CHANGE_VERIFIED", record)


if __name__ == "__main__":
    main()