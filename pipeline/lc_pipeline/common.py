"""Small provenance, path and input helpers shared by the LC commands."""

import hashlib
import io
import json
import math
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
DETECTORS = {"H1", "L1", "V1"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_name(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"GRB\d{9}", value):
        raise ValueError(f"Expected a GBM identifier such as GRB190728271, got {value!r}")
    return value


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def write_json(path, value):
    """Exclusive creation: never silently replace a run or provenance record."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def new_directory(path: str | Path) -> Path:
    """Create a fresh canonical output directory; overwriting is never implicit."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_path(value, base: Path) -> Path:
    """Interpret a configuration path relative to its file, not the shell cwd."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Input paths must be nonempty strings")
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def check_hashes(root, files):
    root = Path(root).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError("Expected a nonempty file-hash manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Changed or unsafe frozen input: {relative}")


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Event metadata must contain only primitive values, not executable pickle classes")


def metadata(directory: str | Path) -> dict:
    """Read only primitive pickle values and enforce the fixed timing contract."""
    directory = Path(directory)
    event = event_name(directory.name)
    path = directory / f"{event}_metadata.pkl"
    try:
        value = PrimitiveUnpickler(io.BytesIO(path.read_bytes())).load()
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("Invalid or truncated primitive event metadata") from error
    if not isinstance(value, dict) or value.get("GRB") != event:
        raise ValueError("Metadata event does not match directory name")
    network = value.get("detectors", [])
    if (
        not isinstance(network, list)
        or not network
        or not all(isinstance(detector, str) for detector in network)
        or len(set(network)) != len(network)
        or set(network) - DETECTORS
    ):
        raise ValueError("Invalid detector network")
    if any(not math.isfinite(float(value[key])) for key in ("GPS", "start", "end", "psd_start", "psd_end")):
        raise ValueError("Event times must be finite")
    checks = (
        (value["end"], value["GPS"]),
        (value["end"] - value["start"], 128.),
        (value["psd_end"] - value["psd_start"], 1024.),
        (value["start"] - value["psd_end"], 16.),
    )
    if any(
        not math.isfinite(float(observed)) or abs(observed - expected) > 1e-6
        for observed, expected in checks
    ):
        raise ValueError("This fixed-model workflow requires the 128s on-source / 1024s context / 16s gap contract")
    return value


def snapshot_files(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}