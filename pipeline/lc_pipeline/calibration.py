"""Rebuild the reviewed GW170817 cache from pinned public release bytes.

This is CPU input preparation only. It never substitutes a newer release,
changes existing inputs, estimates a PSD, imports JAX or submits a search.
"""

import hashlib
import os
import pickle
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from urllib.parse import urlsplit

from .common import now, read_json, sha256, write_json

RELEASE_MANIFEST = Path(__file__).parent / "releases/gw170817-v2.json"
NETWORK = ("H1", "L1", "V1")
BLOCK_BYTES = 1024 * 1024


def load_release() -> dict:
    """Return the maintained, explicitly reviewed release specification."""
    release = read_json(RELEASE_MANIFEST)
    if release["event"] != "GRB170817529" or set(release["detectors"]) != set(NETWORK):
        raise ValueError("Unexpected GW170817 release/event or detector network")
    return release


def verify_release_file(path: Path, source: dict) -> None:
    """Require both the exact reviewed size and complete-file SHA256."""
    if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
        raise ValueError(f"GW170817 release checksum/size mismatch: {path.name}")


def acquire_release_file(session, source: dict, target: Path, source_dir: Path | None) -> str:
    """Stream one bounded HTTPS download, or copy a pinned local release file."""
    if source_dir is not None:
        original = source_dir / source["filename"]
        verify_release_file(original, source)
        with original.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, BLOCK_BYTES)
        resolved_url = source["url"]
    else:
        with session.get(source["url"], stream=True, timeout=(10, 90)) as response:
            response.raise_for_status()
            if urlsplit(response.url).scheme != "https":
                raise ValueError("GW170817 release redirect must remain HTTPS")
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) > source["bytes"]:
                raise ValueError("GW170817 download exceeds the pinned byte limit")
            received = 0
            with target.open("xb") as outgoing:
                for block in response.iter_content(BLOCK_BYTES):
                    received += len(block)
                    if received > source["bytes"]:
                        raise ValueError("GW170817 download exceeds the pinned byte limit")
                    outgoing.write(block)
            resolved_url = response.url
    verify_release_file(target, source)
    return resolved_url


def sample_digest(values) -> str:
    """Hash strain values independently of HDF serialization and machine endian."""
    import numpy as np

    samples = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(samples.tobytes()).hexdigest()


def extract_cache(path: Path, release: dict, source: dict, target: Path) -> dict:
    """Copy the audited sample interval exactly; no resampling or new cleaning."""
    import h5py
    import numpy as np
    from gwpy.timeseries import TimeSeries

    rate = release["sample_rate_hz"]
    with h5py.File(path, "r") as stream:
        strain = stream["strain/Strain"]
        start = float(strain.attrs["Xstart"])
        spacing = float(strain.attrs["Xspacing"])
        if (
            start != release["release_start_gps"]
            or spacing != 1 / rate
            or strain.shape != (release["release_duration_s"] * rate,)
        ):
            raise ValueError("Pinned release HDF sample grid differs from the review")
        first = round((release["cache_start_gps"] - start) * rate)
        count = release["cache_samples"]
        if first < 0 or first + count > len(strain):
            raise ValueError("Reviewed cache interval is not contained in the release")
        actual_start = start + first / rate
        if actual_start != release["cache_start_gps"]:
            raise ValueError("Reviewed cache epoch does not lie on the release grid")
        values = strain[first:first + count]

    if not np.isfinite(values).all() or sample_digest(values) != source["sample_sha256"]:
        raise ValueError("Extracted strain samples differ from the audited cache")

    # Match the historical sample epoch, not the fractional gamma-trigger epoch.
    # The metadata retains the requested gamma-trigger-based window separately.
    series = TimeSeries(
        values, t0=actual_start, sample_rate=rate, unit="strain", name="strain"
    )
    series.write(target, format="hdf5")
    returned = TimeSeries.read(target, format="hdf5")
    if (
        float(returned.t0.value) != actual_start
        or float(returned.sample_rate.value) != rate
        or not np.array_equal(returned.value, values)
    ):
        raise ValueError("HDF cache write/read changed the strain or sample grid")

    return {
        "sha256": sha256(target),
        "sample_sha256": source["sample_sha256"],
        "samples": count,
        "sample_rate_hz": rate,
        "start_gps": actual_start,
        "end_gps": actual_start + count / rate,
        "matches_historical_reviewed_samples": True,
    }


def event_metadata(release: dict) -> dict:
    """Construct the fixed engine's primitive metadata without a legacy pickle."""
    gps = release["gamma_trigger_gps"]
    return {
        "GRB": release["event"],
        "GPS": gps,
        "LIGO_run": "O2",
        "start": gps - 128,
        "end": gps,
        "psd_start": gps - 1168,
        "psd_end": gps - 144,
        "detectors": list(NETWORK),
        "duration": 128,
        "psd_duration": 1024,
        "psd_pad": 16,
    }


def prepare_cache(directory: Path, release: dict, source_dir: Path | None) -> dict:
    """Prepare and validate all three detectors before the cache is published."""
    import requests

    from .data import inspect

    raw = directory / "released"
    raw.mkdir()
    channels, sources = {}, {}
    with requests.Session() as session:
        for detector in NETWORK:
            source = release["detectors"][detector]
            original = raw / source["filename"]
            print(f"{detector}: acquiring pinned GW170817 v2 cleaned release", flush=True)
            resolved_url = acquire_release_file(session, source, original, source_dir)
            target = directory / f"{release['event']}_{detector}.hdf5"
            channels[detector] = extract_cache(original, release, source, target)
            sources[detector] = {**source, "resolved_url": resolved_url}

    meta = event_metadata(release)
    metadata_path = directory / f"{release['event']}_metadata.pkl"
    with metadata_path.open("xb") as stream:
        pickle.dump(meta, stream)
        stream.flush()
        os.fsync(stream.fileno())
    inspect(directory)
    record = {
        "created_utc": now(),
        "status": "pinned_release_and_audited_sample_checks_passed",
        "event": release["event"],
        "acquisition": release["release"],
        "release_manifest_sha256": sha256(RELEASE_MANIFEST),
        "metadata_sha256": sha256(metadata_path),
        "channels": channels,
        "sources": sources,
        "source_mode": "pinned_local_files" if source_dir else "pinned_https_download",
        "limitations": [
            "Pins the reviewed release and sample values, not independent glitch-subtraction validation.",
            "No resampling, gating, additional cleaning, PSD fitting or likelihood evaluation.",
            "New HDF/metadata serialization need not match old cache file hashes; never replace frozen inputs.",
            "The 4-kHz release high-frequency conditioning, model validity and convergence remain separate checks.",
        ],
    }
    write_json(directory / "release.json", release)
    write_json(directory / "data.json", record)
    return record


def acquire(args: Namespace) -> int:
    """Publish a new cache atomically; an incomplete detector network stays hidden."""
    release = load_release()
    output = args.output.resolve()
    if output.name != release["event"]:
        raise ValueError("GW170817 output directory must end in GRB170817529")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Calibration output already exists; choose a fresh directory")
    source_dir = args.source_dir.resolve() if args.source_dir else None
    if source_dir is not None and not source_dir.is_dir():
        raise ValueError("Local release --source-dir does not exist")

    output.parent.mkdir(parents=True, exist_ok=True)
    claim = output.parent / f".{output.name}.acquisition_claim.json"
    write_json(claim, {"created_utc": now(), "output": str(output), "release": release["release"]})
    pending = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent))
    directory = pending / output.name
    directory.mkdir()
    try:
        prepare_cache(directory, release, source_dir)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Calibration output appeared during acquisition")
        directory.rename(output)
    except Exception as error:
        write_json(pending / "failure.json", {"created_utc": now(), "error": str(error)})
        raise ValueError(
            f"Calibration acquisition failed; no cache published. Inspect {pending} "
            "and the acquisition claim before retrying."
        ) from error
    pending.rmdir()
    print("GW170817 reviewed-release cache ready:", output, flush=True)
    return 0