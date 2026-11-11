"""Validate, freeze, submit and summarize location-bound whole-sky/map runs.

Host-side orchestration is kept separate from the byte-protected numerical
engine. Only ``run`` imports JAX, after allocation and frozen-source checks.
"""

import importlib.metadata
import math
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .common import (
    PIPELINE,
    check_hashes,
    event_name,
    metadata,
    new_directory,
    now,
    read_json,
    resolve_path,
    sha256,
    snapshot_files,
    write_json,
)

SAMPLER_SHA = "13b4c55c6131bb35c6b1041519962073b52d76c7d81c4c7cd94fa16498ec9302"
SCIENCE = ("sample_grb_sequential.py", "sequential_sky_prior.py", "grb_localization.py")
PACKAGES = (
    "jax", "jaxlib", "blackjax", "jimgw", "jax-healpy", "ripplegw",
    "numpy", "scipy", "astropy", "astropy-healpix", "anesthetic", "gwpy",
)
SAMPLING_DEFAULTS = {
    "n_live": 500,
    "n_mcmc": 192,
    "num_delete": 150,
    "checkpoint_every": 1,
}
SLURM_DEFAULTS = {
    "account": "",
    "partition": "",
    "qos": "standard",
    "cpus": 8,
    "hours": 12,
    "max_concurrent": 2,
}
MODEL_LIMITATIONS = [
    "Inherited Mc1–2/component0.5–7.7/q0.125–1/spins±0.05/tides0–5000 are NOT a newly validated strict-BNS prior.",
    "Flat distance20–800Mpc and independent tc-trigger[-10,0] retained.",
    "128s/23Hz low-mass duration and release/conditioning concerns remain; check before science use.",
    "Map posterior is used once as sky prior, not an extra GRB likelihood or calibrated association odds.",
    "Convergence/PSD superiority/detection is not established by this workflow's unit tests or stopping criterion.",
]


def versions() -> dict[str, str]:
    """Read package metadata without initializing any numerical backend."""
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def positive_integer(value, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sampling_settings(raw: dict) -> dict:
    """Validate numerical settings without permitting scientific prior changes."""
    settings = dict(SAMPLING_DEFAULTS)
    given = raw.get("sampling", {})
    if not isinstance(given, dict) or not isinstance(raw.get("slurm", {}), dict):
        raise ValueError("sampling and slurm must be JSON objects")
    if set(given) - set(settings):
        raise ValueError("Unknown sampling option; the preserved engine uses fixed -3 stopping")
    settings.update(given)
    for key, value in settings.items():
        positive_integer(value, key, 4 if key == "n_live" else 1)
    if settings["num_delete"] >= settings["n_live"]:
        raise ValueError("num_delete must be smaller than n_live; choose deletion fraction explicitly")
    return settings


def _slurm_settings(given: dict) -> dict:
    """Validate allocation names and caps; never submit or infer resources."""
    slurm = dict(SLURM_DEFAULTS)
    if set(given) - set(slurm):
        raise ValueError("Unknown Slurm option")
    slurm.update(given)
    for key in ("cpus", "hours", "max_concurrent"):
        positive_integer(slurm[key], key)
    for key in ("account", "partition", "qos"):
        value = slurm[key]
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not isinstance(value, str) or any(c not in allowed for c in value):
            raise ValueError("Slurm names must contain only alphanumeric, hyphen or underscore characters")
    return slurm


def _event_and_task_records(entries, base: Path) -> tuple[list[dict], list[dict]]:
    """Resolve input locations and enumerate every declared whole/map seed pair."""
    events = []
    cells = []
    used = set()
    if not isinstance(entries, list):
        raise ValueError("events must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each event must be an object")
        if set(entry) - {"event", "data", "sky", "psd_source", "psd", "pairs"}:
            raise ValueError("Unknown event configuration option")
        event = event_name(entry["event"])
        if event in {e["event"] for e in events}:
            raise ValueError("Duplicate event")
        record = {
            "event": event,
            "data": str(resolve_path(entry["data"], base)),
            "sky": str(resolve_path(entry["sky"], base)),
            "psd_source": entry.get("psd_source", "bayesline"),
        }
        if record["psd_source"] not in ("bayesline", "welch"):
            raise ValueError("Unknown PSD source")
        if record["psd_source"] == "bayesline":
            record["psd"] = str(resolve_path(entry["psd"], base))
        pairs = entry.get("pairs", [])
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Each event requires explicit whole_seed/map_seed pairs")
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or set(pair) != {"whole_seed", "map_seed"}:
                raise ValueError("Every pair needs whole_seed and map_seed")
            for prior, key in (("whole_sky", "whole_seed"), ("grb_map", "map_seed")):
                seed = positive_integer(pair[key], key, 0)
                if seed >= 2**32 or seed in used:
                    raise ValueError("Seeds must be distinct unsigned 32-bit integers")
                used.add(seed)
                cells.append({
                    "task": len(cells),
                    "event": event,
                    "pair": pair_index,
                    "prior": prior,
                    "seed": seed,
                    "output": f"runs/{event}/{prior}_seed_{seed}",
                })
        events.append(record)
    if "GRB170817529" not in {e["event"] for e in events}:
        raise ValueError("Include a GW170817 whole/map calibration pair in every new workflow route")
    if len({e["psd_source"] for e in events}) != 1:
        raise ValueError("Use the same PSD route for candidates and GW170817; put Welch comparisons in a separate bundle")
    return events, cells


def normalize_config(path: Path) -> dict:
    """Convert the user JSON into the frozen, explicit fixed-model task plan."""
    path = path.resolve()
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Search configuration must be a JSON object")
    if set(raw) - {"schema_version", "model", "sampling", "slurm", "events"}:
        raise ValueError("Unknown top-level configuration key; scientific prior overrides are not supported by this fixed engine")
    if raw.get("schema_version") != 1 or raw.get("model") != "current_fixed":
        raise ValueError("Require schema_version=1 and model=current_fixed; revised BNS priors are not silently enabled")

    settings = _sampling_settings(raw)
    slurm = _slurm_settings(raw.get("slurm", {}))
    events, cells = _event_and_task_records(raw.get("events"), path.parent)
    return {
        "schema_version": 1,
        "model": "current_fixed",
        "created_utc": now(),
        "sampling": settings,
        "slurm": slurm,
        "events": events,
        "cells": cells,
        "source_config_sha256": sha256(path),
        "stop_logZ_live_minus_logZ": -3.,
        "limitations": list(MODEL_LIMITATIONS),
    }


def validate_sky(package: Path, event: str, gps: float) -> dict:
    """Bind prepared pixels and timing to the retained original released FITS."""
    import numpy as np
    from .sky import validate_release
    record = read_json(package.with_name("grb_inputs.json"))
    if record.get("event") != event or record.get("probability_usable") is not True or sha256(package) != record["artifact_sha256"]:
        raise ValueError("Sky package identity, usability or hash mismatch")
    if (record["timing"]["tc_minus_trigger_support_s"] != [-10., 0.]
            or record["timing"].get("density_per_s") != .1):
        raise ValueError("Sky-package time support differs from this engine")
    if not math.isfinite(float(record["gamma_trigger_gps"])) or not math.isfinite(gps) or abs(float(record["gamma_trigger_gps"])-gps) > .001:
        raise ValueError("Sky and strain trigger times disagree")
    released = package.with_name("released_probability.fit")
    original, _ = validate_release(released, event, gps, record["released_probability_source"])
    with np.load(package, allow_pickle=False) as z:
        p = z["probability_per_native_pixel"]
        n = int(z["nside"])
        if n <= 0 or n & (n-1) or p.shape != (12*n*n,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1., rtol=1e-6, atol=1e-10):
            raise ValueError("Invalid full-sky differential probability array")
        if str(z["ordering"]) not in ("nested", "ring") or str(z["native_frame"]) not in ("icrs", "fk5"):
            raise ValueError("Unsupported sky pixel ordering/frame")
        if (not np.isfinite(float(z["equinox"])) or not np.isfinite(float(z["gamma_trigger_gps"]))
            or abs(float(z["gamma_trigger_gps"])-gps) > .001):
            raise ValueError("Invalid package equinox or trigger")
        if ((n, str(z["ordering"]), str(z["native_frame"]), float(z["equinox"]))
            != (original.nside, original.order, original.native_frame, original.equinox)
            or p.shape != original.probability.shape or not np.allclose(p, original.probability, rtol=0, atol=1e-15)):
            raise ValueError("Prepared probability pixels or coordinates differ from the original released FITS")
    return record


def render_slurm(bundle, config, python):
    s = config["slurm"]
    optional = "\n".join(f"#SBATCH --{key}={s[key]}" for key in ("account", "partition", "qos") if s[key])
    return f"""#!/bin/bash
#SBATCH --job-name=subthreshold-search
{optional}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task={s['cpus']}
#SBATCH --time={s['hours']:02d}:00:00
#SBATCH --array=0-{len(config['cells'])-1}%{s['max_concurrent']}
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue
#SBATCH --chdir={bundle}
#SBATCH --output={bundle}/slurm-%A_%a.out
#SBATCH --error={bundle}/slurm-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec {shlex.quote(python)} -u {shlex.quote(str(bundle / 'source/run.py'))} search run --bundle {shlex.quote(str(bundle))} --task "${{SLURM_ARRAY_TASK_ID:?}}" "$@"
"""


def prepare(args) -> int:
    """Validate inputs, snapshot a new bundle and verify it; never submit it."""
    from .data import inspect
    from .psd import raw_psd
    config = normalize_config(args.config)
    if any(c.isspace() or c in "#'\"" for c in str(args.output.resolve())):
        raise ValueError("Use a bundle path without whitespace, quotes or # so Slurm directives are unambiguous")
    if sha256(PIPELINE / "sample_grb_sequential.py") != SAMPLER_SHA:
        raise ValueError("Current engine differs from the audited source; review before snapshotting")
    # Bind validation to the exact bytes later copied, not a second unreviewed read.
    validated_hashes, psd_profiles = {}, set()
    for entry in config["events"]:
        directory = Path(entry["data"])
        if directory.name != entry["event"]:
            raise ValueError("Each data directory must end in its event name")
        meta = metadata(directory)
        sky_path = Path(entry["sky"])
        if sky_path.name != "grb_probability_inputs.npz":
            raise ValueError("Use the standard acquired-package filename")
        paths = [directory / f"{entry['event']}_metadata.pkl"]
        paths += [directory / f"{entry['event']}_{d}.hdf5" for d in meta["detectors"]]
        paths += [sky_path, sky_path.with_name("grb_inputs.json"), sky_path.with_name("released_probability.fit")]
        if entry["psd_source"] == "bayesline":
            paths += [Path(entry["psd"]) / "psd.json"]
            paths += [Path(entry["psd"]) / f"glitch_median_PSD_{d}.dat" for d in meta["detectors"]]
        validated_hashes.update({str(p): sha256(p) for p in paths})
        meta, _ = inspect(directory)
        validate_sky(Path(entry["sky"]), entry["event"], float(meta["GPS"]))
        if entry["psd_source"] == "bayesline":
            psd = Path(entry["psd"])
            record = read_json(psd / "psd.json")
            if record.get("event") != entry["event"] or record.get("convention") != "raw_bayeswave_power_2_over_128":
                raise ValueError("PSD event/units mismatch; use psd import for legacy raw files, not physical PSD or ASD files")
            if record["metadata_sha256"] != sha256(directory / f"{entry['event']}_metadata.pkl"):
                raise ValueError("PSD and strain metadata mismatch")
            check_hashes(psd, record["files"])
            required_psds = {f"glitch_median_PSD_{d}.dat" for d in meta["detectors"]}
            if not required_psds.issubset(record["files"]):
                raise ValueError("PSD manifest omits a required detector product")
            psd_profiles.add(record.get("estimator", "legacy_import_unspecified"))
            if record.get("bundle_result"):
                result = read_json((psd / record["bundle_result"]).resolve())
                if result.get("status") != "completed_with_input_and_unit_checks":
                    raise ValueError("BayesWave products do not have a successful completed input/unit-check result")
            if record.get("strain"):
                for detector in meta["detectors"]:
                    if record["strain"][detector]["sha256"] != sha256(directory / f"{entry['event']}_{detector}.hdf5"):
                        raise ValueError("PSD was estimated from different strain bytes")
            for detector in meta["detectors"]:
                raw_psd(psd / f"glitch_median_PSD_{detector}.dat")
    if len(psd_profiles) > 1:
        raise ValueError("Candidates and GW170817 must use the same PSD estimator route; do not mix archived and new noise-only estimates")
    for path, digest in validated_hashes.items():
        if sha256(path) != digest:
            raise ValueError("Input changed while being validated")
    output = new_directory(args.output)
    source = output / "source"
    source.mkdir()
    input_hashes = {}

    def copy_input(original, target):
        before = validated_hashes[str(original)]
        shutil.copy2(original, target)
        if sha256(original) != before or sha256(target) != before:
            raise ValueError("Source input changed during snapshot")
        input_hashes[str(original)] = before

    for name in (*SCIENCE, "run.py"):
        shutil.copy2(PIPELINE / name, source / name)
    shutil.copytree(PIPELINE / "lc_pipeline", source / "lc_pipeline", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for entry in config["events"]:
        event, directory = entry["event"], Path(entry["data"])
        meta = metadata(directory)
        dest = output / "inputs/strain" / event
        dest.mkdir(parents=True)
        for name in [f"{event}_metadata.pkl"]+[f"{event}_{d}.hdf5" for d in meta["detectors"]]:
            copy_input(directory / name, dest / name)
        sky = output / "inputs/sky" / event
        sky.mkdir(parents=True)
        for original in [Path(entry["sky"]), Path(entry["sky"]).with_name("grb_inputs.json")]:
            copy_input(original, sky / original.name)
        original = Path(entry["sky"]).with_name("released_probability.fit")
        copy_input(original, sky / original.name)
        if entry["psd_source"] == "bayesline":
            dest = output / "inputs/psd" / event
            dest.mkdir(parents=True)
            for name in ["psd.json"]+[f"glitch_median_PSD_{d}.dat" for d in meta["detectors"]]:
                copy_input(Path(entry["psd"]) / name, dest / name)
    write_json(output / "config.json", config)
    (output / "search.slurm").write_text(render_slurm(output, config, sys.executable))
    subprocess.run(["bash", "-n", str(output / "search.slurm")], check=True)
    write_json(output / "manifest.json", {"created_utc": now(), "files": snapshot_files(output),
               "versions": versions(), "python": str(Path(sys.executable).absolute()),
               "environment_prefix": str(Path(sys.prefix).resolve()), "python_version": sys.version,
               "original_input_hashes_at_snapshot": input_hashes,
               "scope": "Unified Subthreshold workflow; no changes to historical bundles"})
    verify(output)
    print("Prepared search (not submitted):", output)
    return 0


def verify(bundle: Path) -> tuple[dict, dict]:
    """Check frozen bytes, environment and original absolute bundle location."""
    bundle = bundle.resolve()
    manifest = read_json(bundle / "manifest.json")
    check_hashes(bundle, manifest["files"])
    if (versions() != manifest["versions"] or Path(sys.executable).resolve() != Path(manifest["python"]).resolve()
            or str(Path(sys.prefix).resolve()) != manifest["environment_prefix"] or sys.version != manifest["python_version"]):
        raise ValueError("Analysis environment changed; use the Python recorded in manifest.json")
    if sha256(bundle / "source/sample_grb_sequential.py") != SAMPLER_SHA:
        raise ValueError("Frozen scientific engine changed")
    config = read_json(bundle / "config.json")
    if (bundle / "search.slurm").read_text() != render_slurm(bundle, config, manifest["python"]):
        raise ValueError("Slurm payload or bundle location differs; frozen bundles must not be moved for resume")
    return config, manifest


def command(
    bundle: Path,
    config: dict,
    task: int,
    resume: bool = False,
    preflight: bool = False,
) -> tuple[list[str], Path]:
    """Build sampler arguments; preflight uses a separate output directory."""
    if type(task) is not int or not 0 <= task < len(config["cells"]) or resume and preflight:
        raise ValueError("Invalid task or conflicting resume/preflight")
    cell = config["cells"][task]
    entry = next(e for e in config["events"] if e["event"] == cell["event"])
    output = bundle / (f"preflight/task_{task}" if preflight else cell["output"])
    if not output.resolve().is_relative_to((bundle / ("preflight" if preflight else "runs")).resolve()):
        raise ValueError("Unsafe task output path")
    argv = [str(bundle / "source/sample_grb_sequential.py"), "--event", cell["event"],
            "--sky-package", str(bundle / "inputs/sky" / cell["event"] / "grb_probability_inputs.npz"),
            "--data-dir", str(bundle / "inputs/strain"), "--psd-source", entry["psd_source"],
            "--prior", cell["prior"], "--seed", str(cell["seed"]),
            "--output-root", str(bundle), "--output-dir", str(output)]
    if entry["psd_source"] == "bayesline":
        argv += ["--psd-dir", str(bundle / "inputs/psd" / cell["event"])]
    for key, flag in (("n_live", "--n-live"), ("n_mcmc", "--n-mcmc"), ("num_delete", "--num-delete"), ("checkpoint_every", "--checkpoint-every")):
        argv += [flag, str(config["sampling"][key])]
    if resume:
        argv.append("--resume")
    if preflight:
        argv.append("--preflight")
    return argv, output


def check_output(output: Path, resume: bool) -> None:
    """Refuse overwrites and any resume of final or partly finalized outputs."""
    if not resume:
        if output.exists():
            raise ValueError("Output already exists; choose a fresh task/bundle, not an overwrite")
    elif not (output / "checkpoint.pkl").is_file() or not (output / "run_metadata.json").is_file():
        raise ValueError("Resume requires an existing checkpoint and metadata")
    elif (output / "posterior.csv").exists() or (output / "evidence_summary.json").exists():
        raise ValueError("Final or partially finalized output requires review, not automatic resume")


def run(args) -> int:
    """Enter the unmodified engine only inside an explicitly selected GPU run."""
    if not os.environ.get("SLURM_JOB_ID") and not args.local_gpu:
        raise ValueError("Real-data JAX requires a compute allocation; use --local-gpu only on your own GPU workstation")
    if os.environ.get("JAX_PLATFORMS") != "cuda":
        raise ValueError("Set JAX_PLATFORMS=cuda; CPU fallback is refused")
    bundle = args.bundle.resolve()
    if Path(__file__).resolve() != bundle / "source/lc_pipeline/search.py":
        raise ValueError("Execute this bundle's frozen source/run.py, not the mutable checkout")
    config, _ = verify(bundle)
    argv, output = command(bundle, config, args.task, args.resume, args.preflight)
    check_output(output, args.resume)
    import jax
    if not jax.devices() or any(device.platform != "gpu" for device in jax.devices()):
        raise ValueError("No CUDA GPU device")
    sys.path.insert(0, str(bundle / "source"))
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")
    return 0


def submit(args) -> int:
    """Make one explicit, claimed Slurm request; ambiguous receipts never retry."""
    bundle = args.bundle.resolve()
    config, _ = verify(bundle)
    task = args.resume_task
    selected = range(len(config["cells"])) if task is None else [task]
    for index in selected:
        _, output = command(bundle, config, index, resume=task is not None)
        check_output(output, task is not None)
    queue = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%Z"], text=True, capture_output=True, check=True).stdout
    if any(str(bundle) in line for line in queue.splitlines()):
        raise ValueError("This bundle already has active jobs")
    prefix = "submission"
    options = []
    if task is not None:
        if not (bundle / "submission/receipt.json").exists():
            raise ValueError("Missing original submission receipt")
        receipt = read_json(bundle / "submission/receipt.json")
        if receipt.get("exit_code") != 0 or not str(receipt.get("job_id", "")).isdigit():
            raise ValueError("Original submission was failed/ambiguous; reconcile it before any resume")
        _, output = command(bundle, config, task, resume=True)
        prefix = f"resume/task_{task}/checkpoint_{sha256(output / 'checkpoint.pkl')}"
        options = [f"--array={task}"]
    write_json(bundle / prefix / "claim.json", {"created_utc": now(), "purpose": "Exactly once; never retry an ambiguous submission without reconciliation"})
    cmd = ["sbatch", "--parsable", *options, str(bundle / "search.slurm"), *(["--resume"] if task is not None else [])]
    test = subprocess.run(["sbatch", "--test-only", *cmd[2:]], text=True, capture_output=True)
    write_json(bundle / prefix / "attempt.json", {"command": cmd, "test_exit": test.returncode, "stdout": test.stdout, "stderr": test.stderr})
    if test.returncode:
        raise ValueError("Slurm test-only rejected request; nothing submitted. Review the saved attempt before retrying.")
    result = subprocess.run(cmd, text=True, capture_output=True)
    job = result.stdout.strip().split(";")[0]
    record = {"created_utc": now(), "command": cmd, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "job_id": job}
    write_json(bundle / prefix / "receipt.json", record)
    if result.returncode or not job.isdigit():
        raise ValueError("Submission failed/ambiguous; reconcile receipt with scheduler, do not blindly retry")
    print("Submitted array:", job)
    return 0


def _completed_summary(directory: Path, cell: dict, config: dict) -> dict:
    """Check a final summary's identity and scalar fields without loading JAX.

    This is not a full checkpoint/posterior integrity audit. That stronger
    acceptance step is separate from the human-readable status summary.
    """
    required = ("run_metadata.json", "posterior.csv", "checkpoint.pkl")
    if not all((directory / name).is_file() for name in required):
        raise ValueError("Final summary is missing a required output artifact")
    result = read_json(directory / "evidence_summary.json")
    meta = read_json(directory / "run_metadata.json")
    identity_matches = all(meta.get(key) == cell[key] for key in ("event", "prior", "seed"))
    if not identity_matches or result["remaining_logZ_gap"] >= -3:
        raise ValueError("Final output identity/stopping mismatch")

    settings = ("n_live", "n_mcmc", "num_delete")
    if any(meta.get(key) != config["sampling"][key] for key in settings):
        raise ValueError("Final sampler settings differ from the frozen configuration")
    expected_psd = next(
        entry["psd_source"] for entry in config["events"] if entry["event"] == cell["event"]
    )
    routes_match = (
        meta.get("psd_source") == expected_psd
        and result.get("psd_source") == expected_psd
        and result.get("prior") == cell["prior"]
    )
    if not routes_match:
        raise ValueError("Final summary/metadata prior or PSD route mismatch")

    fields = (
        "log_bayes_factor_cbc_noise", "log_bayes_factor_std",
        "noise_log_density", "remaining_logZ_gap",
    )
    if (
        not all(math.isfinite(result[key]) for key in fields)
        or result["log_bayes_factor_std"] < 0
    ):
        raise ValueError("Nonfinite evidence/stopping value or negative internal uncertainty")
    return result


def summarize(args) -> int:
    """Report every declared task and completed pair, not just favorable seeds."""
    bundle = args.bundle.resolve()
    config, _ = verify(bundle)
    runs, lookup = [], {}
    for cell in config["cells"]:
        directory = bundle / cell["output"]
        summary = directory / "evidence_summary.json"
        status = "not_started" if not directory.exists() else "incomplete"
        record = {**cell, "status": status}
        if summary.exists():
            result = _completed_summary(directory, cell, config)
            record.update(
                status="completed",
                logB=result["log_bayes_factor_cbc_noise"],
                internal_std=result["log_bayes_factor_std"],
                noise_log_density=result["noise_log_density"],
            )
        elif (directory / "checkpoint.pkl").exists():
            record["status"] = "checkpointed_or_running"
        runs.append(record)
        lookup[cell["task"]] = record
    pairs = []
    for cell in config["cells"]:
        if cell["prior"] != "whole_sky":
            continue
        a, b = lookup[cell["task"]], lookup[cell["task"]+1]
        if a["status"] == b["status"] == "completed":
            if abs(a["noise_log_density"]-b["noise_log_density"]) > 3e-8:
                raise ValueError("Whole/map noise normalization differs")
            pairs.append({"event": cell["event"], "whole_seed": a["seed"], "map_seed": b["seed"],
                          "delta_logZ": b["logB"]-a["logB"], "internal_std_only": math.hypot(a["internal_std"], b["internal_std"])})
    output = new_directory(args.output)
    write_json(output / "summary.json", {"created_utc": now(), "runs": runs, "completed_pairs": pairs,
               "caveat": "Conditional fixed-PSD CBC/noise and map-minus-whole evidences; no convergence/detection/association claim; internal errors omit cross-seed exploration error"})
    text = ["# Subthreshold search status", "", "| Event | Prior | Seed | Status | log B(CBC/noise) |", "|---|---|---:|---|---:|"]
    for record in runs:
        text.append(f"| {record['event']} | {record['prior']} | {record['seed']} | {record['status']} | {record.get('logB', '—')} |")
    text += ["", "All completed predeclared map-minus-whole pairs:", ""]
    text += [f"- {p['event']} ({p['whole_seed']}/{p['map_seed']}): {p['delta_logZ']:.6f} ± {p['internal_std_only']:.6f} (internal only)." for p in pairs]
    text += ["", "Incomplete checkpoints are not evidence results. Similar posteriors and internal errors alone do not establish convergence.", ""]
    (output / "summary.md").write_text("\n".join(text))
    print("Summary:", output / "summary.md")
    return 0