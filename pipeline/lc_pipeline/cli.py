"""Unified workflow CLI; expensive scientific imports happen only when needed."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from . import __version__


def parser() -> argparse.ArgumentParser:
    """Build the public interface; parsing never imports a numerical backend."""
    p = argparse.ArgumentParser(
        description=(
            "Subthreshold: released GRB sky → GW strain → BayesWave PSD → paired GW search. "
            "No command submits jobs implicitly."
        )
    )
    p.add_argument("--version", action="version", version=__version__)
    groups = p.add_subparsers(dest="group", required=True)
    groups.add_parser("doctor", help="Print host environment versions without importing JAX")
    _add_sky_commands(groups)
    _add_data_commands(groups)
    _add_psd_commands(groups)
    _add_search_commands(groups)
    return p


def _add_sky_commands(groups) -> None:
    """Released probability-map discovery, validation and local import."""
    sky = groups.add_parser("sky", help="Acquire strict released differential GBM probability maps").add_subparsers(dest="action", required=True)
    a = sky.add_parser("acquire", help="Discover/import, validate and package maps; always include GW170817")
    a.add_argument("--catalogue", required=True, type=Path, help="CSV with GRB,GPS (or event,gamma_trigger_gps)")
    selection = a.add_mutually_exclusive_group(required=True)
    selection.add_argument("--events", nargs="+")
    selection.add_argument("--all", action="store_true", help="Explicitly acquire all catalogue entries; does not launch searches")
    a.add_argument("--sources", type=Path, help="Optional JSON mapping event → pinned URL/local path and provenance")
    a.add_argument("--offline", action="store_true", help="Require local source entries; no network")
    a.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    a.add_argument("--samples", type=int, default=20000)
    a.add_argument("--seed", type=int, default=170817)


def _add_data_commands(groups) -> None:
    """Separate reviewed calibration acquisition from generic run-data fetching."""
    data = groups.add_parser("data", help="Import reviewed caches or fetch modern public strain").add_subparsers(dest="action", required=True)
    a = data.add_parser("import", help="Copy an existing event directory after validating sample coverage")
    a.add_argument("--source", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path, help="New directory ending in GRB identifier")
    a = data.add_parser("fetch", help="Fetch modern public GWOSC strain; release/DQ review remains required")
    a.add_argument("--event", required=True)
    a.add_argument("--gps", type=float, required=True)
    a.add_argument("--detectors", choices=("H1", "L1", "V1"), nargs="+", required=True)
    a.add_argument("--output", type=Path, required=True)
    a = data.add_parser("gw170817", help="Rebuild the reviewed calibration cache from checksum-pinned public v2 cleaned strain")
    a.add_argument("--source-dir", type=Path, help="Optional offline directory of the three exact released HDF files, not a legacy cache")
    a.add_argument("--output", type=Path, required=True, help="Fresh output directory ending in GRB170817529")


def _add_psd_commands(groups) -> None:
    """Explicit PSD-unit import or separate-environment BayesWave execution."""
    psd = groups.add_parser("psd", help="Import raw PSDs or prepare/run exact-input BayesWave").add_subparsers(dest="action", required=True)
    a = psd.add_parser("import", help="Register existing raw glitch_median_PSD_IFO.dat products")
    a.add_argument("--source", required=True, type=Path)
    a.add_argument("--data", required=True, type=Path, help="Event data directory")
    a.add_argument("--convention", required=True, choices=("raw-2-over-128",), help="Explicit unit assertion; do not pass physical PSD/ASD")
    a.add_argument("--output", required=True, type=Path)
    a = psd.add_parser("prepare", help="Write exact-input frames and commands in the BayesWave Python environment")
    a.add_argument("--data", required=True, type=Path)
    a.add_argument("--prefix", required=True, type=Path, help="Conda environment containing bin/BayesWave and bin/BayesWavePost")
    a.add_argument("--iterations", type=int, default=5000)
    a.add_argument("--burnin", type=int, default=1000)
    a.add_argument("--output", required=True, type=Path)
    a = psd.add_parser("run", help="Run both CPU BayesWave stages and check input FFTs/PSD units")
    a.add_argument("--bundle", required=True, type=Path)
    a.add_argument("--local", action="store_true", help="Explicit workstation execution instead of Slurm compute allocation")


def _add_search_commands(groups) -> None:
    """Preparation and submission are deliberately different user actions."""
    search = groups.add_parser("search", help="Freeze, run, resume and summarize whole/map searches").add_subparsers(dest="action", required=True)
    a = search.add_parser("prepare", help="Validate inputs and snapshot a new paired-search bundle (no submission)")
    a.add_argument("--config", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = search.add_parser("verify", help="Verify frozen files and analysis environment")
    a.add_argument("--bundle", required=True, type=Path)
    a = search.add_parser("run", help="Run one task inside a GPU allocation using frozen sources")
    a.add_argument("--bundle", required=True, type=Path)
    a.add_argument("--task", type=int, required=True)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--preflight", action="store_true", help="Use a SEPARATE output directory for likelihood preflight")
    a.add_argument("--local-gpu", action="store_true", help="Explicit own-workstation GPU use; never run on an HPC login node")
    a = search.add_parser("submit", help="Explicit Slurm submission with test-only and duplicate guards")
    a.add_argument("--bundle", required=True, type=Path)
    a.add_argument("--resume-task", type=int, help="Resume one unfinished task under the same allocation cap")
    a = search.add_parser("summarize", help="Report all tasks, incomplete states and completed evidence pairs")
    a.add_argument("--bundle", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)


def doctor() -> int:
    """Read package metadata without importing JAX, CUDA or the likelihood."""
    from .search import PACKAGES

    result = {
        "python": sys.version,
        "executable": sys.executable,
        "workflow": "subthreshold",
        "workflow_version": __version__,
    }
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "NOT INSTALLED"
    result["jax_imported"] = "jax" in sys.modules
    result["required_nss_commit"] = "2180e29ffb645b2c46b76c768229c7c24212446c"
    result["nss_fork_version_matches_reference"] = (
        result["blackjax"] == "0.1.0b1.dev194+g2180e29ff"
    )
    result["dependency_metadata_note"] = (
        "JimGW requires blackjax>=1.6; this NSS fork reports 0.1.0. "
        "Install it separately with --no-deps. pip check reports that known "
        "version mismatch; do not auto-upgrade it."
    )
    print(json.dumps(result, indent=2))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    """Import only the selected command's implementation after parsing."""
    if args.group == "doctor":
        return doctor()
    if args.group == "sky":
        if args.samples < 1 or args.seed < 0:
            raise ValueError("Positive sample count and nonnegative seed required")
        from .sky import acquire

        return acquire(args)
    if args.group == "data":
        if args.action == "gw170817":
            from .calibration import acquire

            return acquire(args)
        from . import data

        handlers = {"import": data.import_data, "fetch": data.fetch}
        return handlers[args.action](args)
    if args.group == "psd":
        from . import psd

        if args.action == "prepare" and not 0 <= args.burnin < args.iterations:
            raise ValueError("Require iterations > burnin >= 0")
        handlers = {"import": psd.import_psd, "prepare": psd.prepare, "run": psd.run}
        return handlers[args.action](args)

    from . import search

    if args.action == "verify":
        config, manifest = search.verify(args.bundle)
        print(
            f"Verified {len(manifest['files'])} frozen files; "
            f"{len(config['cells'])} tasks; no likelihood executed"
        )
        return 0
    handlers = {
        "prepare": search.prepare,
        "run": search.run,
        "submit": search.submit,
        "summarize": search.summarize,
    }
    return handlers[args.action](args)


def main(argv: list[str] | None = None) -> int:
    """Translate expected input/command failures to a short user-facing error."""
    p = parser()
    args = p.parse_args(argv)
    try:
        return dispatch(args)
    except (ValueError, KeyError, TypeError, OSError, ImportError, AssertionError, subprocess.CalledProcessError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())