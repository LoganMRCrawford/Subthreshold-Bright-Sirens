"""Explicit raw-PSD import and exact-input BayesWave 1.1.3 noise-only workflow."""

import math
import os
import shutil
import subprocess
import sys
import importlib.metadata
from pathlib import Path

from .common import PIPELINE, check_hashes, metadata, new_directory, now, read_json, sha256, snapshot_files, write_json


def raw_psd(path):
    import numpy as np
    raw = np.loadtxt(path, usecols=(0, 1))
    if raw.ndim != 2 or len(raw) < 2 or not np.isfinite(raw).all() or np.any(raw[:, 1] <= 0):
        raise ValueError(f"Expected finite positive raw BayesWave power bins: {path}")
    frequency, inverse, counts = np.unique(raw[:, 0], return_inverse=True, return_counts=True)
    if frequency[0] > 23 or frequency[-1] < 2048.-.02 or len(frequency) < 2:
        raise ValueError("Raw PSD must cover the complete 23–2048 Hz likelihood band")
    physical = np.bincount(inverse, weights=raw[:, 1]*2./128.)/counts
    return frequency, physical


def import_psd(args):
    meta = metadata(args.data)
    source = args.source.resolve()
    paths = {d: source / f"glitch_median_PSD_{d}.dat" for d in meta["detectors"]}
    for path in paths.values():
        raw_psd(path)
    output = new_directory(args.output)
    files = {}
    for detector, path in paths.items():
        target = output / path.name
        shutil.copy2(path, target)
        digest = sha256(target)
        if digest != sha256(path):
            raise ValueError("PSD changed during import")
        files[target.name] = digest
    write_json(output / "psd.json", {"event": meta["GRB"], "created_utc": now(), "files": files,
               "source_directory": str(source), "convention": "raw_bayeswave_power_2_over_128",
               "estimator": "legacy_import_unspecified",
               "strain": {d: {"path": str((args.data / f"{meta['GRB']}_{d}.hdf5").resolve()),
                              "sha256": sha256(args.data / f"{meta['GRB']}_{d}.hdf5")} for d in meta["detectors"]},
               "strain_binding_origin": "Caller-associated cache at import, not independent proof of the historical estimator's input release",
               "import_assertion": "Caller explicitly declared raw-2-over-128 units; units cannot be inferred from positive numbers",
               "metadata_sha256": sha256(args.data / f"{meta['GRB']}_metadata.pkl"),
               "note": "Not physical PSD/ASD input. Legacy source release, strain matching and estimator convergence require review."})
    print("Imported raw PSD products:", output)
    return 0


def validate_fft_grid(rate=4096, duration=128, fmin=23., fmax=2048.):
    if (rate, duration, fmin, fmax) != (4096, 128, 23., 2048.):
        raise ValueError("This adapter requires 4096Hz/128s and the FULL 23–2048Hz fitting grid; lowering fhigh can truncate/reinterpret BayesLine startup samples")
    count = int(rate*duration)
    if count & (count-1):
        raise ValueError("BayesLine startup sample count must be a power of two")
    return count


def environment():
    return {"python": sys.version, "prefix": str(Path(sys.prefix).resolve()),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "gwpy", "astropy")}}


def commands(bundle, config):
    validate_fft_grid()
    common = ["--psdlength", "1024", "--trigtime", str(config["trigger_gps"]),
              "--segment-start", str(config["fit_start_gps"]), "--psdstart", str(config["psd_start_gps"]),
              "--srate", "4096", "--seglen", "128", "--window", "1.0", "--padding", "0.4",
              "--dataseed", "1234", "--outputDir", str(bundle / "work")]
    for detector in config["detectors"]:
        common += ["--ifo", detector, f"--{detector}-cache", str(bundle / "inputs" / f"{detector}.cache"),
                   f"--{detector}-channel", f"{detector}:LC_EXACT_STRAIN", f"--{detector}-flow", "23",
                   f"--{detector}-fhigh", "2048", f"--{detector}-timeslide", "0.0"]
    prefix = Path(config["bayeswave_prefix"])
    return [[str(prefix / "bin/BayesWave"), *common, "--bayesLine", "--noiseOnly", "--noClean",
             "--noWaveletPrior", "--Niter", str(config["iterations"]), "--Nburnin", str(config["burnin"]),
             "--Nchain", "2", "--Nthreads", "2", "--chainseed", "170817"],
            [str(prefix / "bin/BayesWavePost"), *common, "--bayesLine", "--0noise"]]


def prepare(args):
    """Run in the separate BayesWave environment (LAL frame bindings required)."""
    import lal
    import lalframe
    import numpy as np
    from gwpy.timeseries import TimeSeries
    from scipy.signal import get_window
    meta = metadata(args.data)
    prefix = args.prefix.resolve()
    if Path(sys.prefix).resolve() != prefix:
        raise ValueError("Prepare with the Python inside --prefix so LAL, libraries and BayesWave use the same environment")
    if type(args.iterations) is not int or type(args.burnin) is not int or not 0 <= args.burnin < args.iterations:
        raise ValueError("Require integer iterations > burnin >= 0")
    metadata_path = args.data / f"{meta['GRB']}_metadata.pkl"
    metadata_hash = sha256(metadata_path)
    for name in ("BayesWave", "BayesWavePost"):
        if not os.access(prefix / "bin" / name, os.X_OK):
            raise ValueError(f"Missing executable: {prefix / 'bin' / name}")
    output = new_directory(args.output)
    (output / "inputs").mkdir()
    (output / "reference").mkdir()
    (output / "source").mkdir()
    shutil.copy2(metadata_path, output / "inputs" / metadata_path.name)
    shutil.copy2(PIPELINE / "run.py", output / "source/run.py")
    shutil.copytree(PIPELINE / "lc_pipeline", output / "source/lc_pipeline", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    starts = []
    sources = {}
    for detector in meta["detectors"]:
        source = args.data / f"{meta['GRB']}_{detector}.hdf5"
        source_hash = sha256(source)
        series = TimeSeries.read(source, format="hdf5")
        context = series.crop(float(meta["psd_start"]), float(meta["psd_end"]))
        if len(context) != 1024*4096 or float(context.sample_rate.value) != 4096 or not np.isfinite(context.value).all():
            raise ValueError("Incomplete finite 4096Hz/1024s off-source context")
        start = float(context.t0.value)
        if abs(start-meta["psd_start"])*4096 > 1.01:
            raise ValueError("PSD context start is not aligned with the requested sample grid")
        starts.append(start)
        channel = f"{detector}:LC_EXACT_STRAIN"
        epoch = lal.LIGOTimeGPS(start)
        data = lal.CreateREAL8TimeSeries(channel, epoch, 0, 1/4096, lal.StrainUnit, len(context))
        data.data.data[:] = context.value
        frame = lalframe.FrameNew(epoch, 1024., "LC_EXACT", 0, 0, 0)
        lalframe.FrameAddREAL8TimeSeriesProcData(frame, data)
        frame_path = output / "inputs" / f"{detector}.gwf"
        lalframe.FrameWrite(frame, str(frame_path))
        returned = TimeSeries.read(frame_path, channel=channel, format="gwf", backend="lalframe")
        np.testing.assert_array_equal(returned.value, context.value)
        if float(returned.t0.value) != start:
            raise ValueError("Frame write changed sample epoch")
        cache_start, cache_end = math.floor(start), math.ceil(start+1024)
        cache_path = output / "inputs" / f"{detector}.cache"
        cache_path.write_text(f"{detector[0]} LC_EXACT {cache_start} {cache_end-cache_start} file://localhost{frame_path}\n")
        cache = lal.CacheImport(str(cache_path))
        stream = lalframe.FrStreamCacheOpen(cache)
        checked = lalframe.FrStreamReadREAL8TimeSeries(stream, channel, epoch, 1024., 0)
        np.testing.assert_array_equal(checked.data.data, context.value)
        fit = context.value[-128*4096:]
        window = get_window(("tukey", 2*.4/128), len(fit), fftbins=False)
        np.savez_compressed(output / "reference" / f"{detector}.npz", frequency=np.fft.rfftfreq(len(fit), 1/4096),
                            fft=np.fft.rfft(fit*window)/4096, window_energy=np.mean(window**2))
        if sha256(source) != source_hash:
            raise ValueError("Strain changed while preparing frames")
        sources[detector] = {"path": str(source.resolve()), "sha256": source_hash}
    if len(set(starts)) != 1:
        raise ValueError("Detector PSD context epochs differ")
    if sha256(metadata_path) != metadata_hash or sha256(output / "inputs" / metadata_path.name) != metadata_hash:
        raise ValueError("Metadata changed during PSD preparation")
    config = {"event": meta["GRB"], "created_utc": now(), "detectors": meta["detectors"],
              "bayeswave_prefix": str(prefix), "trigger_gps": meta["GPS"], "psd_start_gps": starts[0],
              "fit_start_gps": starts[0]+1024-128, "iterations": args.iterations, "burnin": args.burnin,
              "metadata_sha256": metadata_hash, "strain": sources,
              "status": "NEW noiseOnly/noClean estimator route, not identical to historical cleanOnly PSDs or a validated production replacement",
              "training": "1024s initialization context; final128s noise-only fit; sixteen-second gap to on-source"}
    write_json(output / "psd_config.json", config)
    write_json(output / "commands.json", {"commands": commands(output, config)})
    files = snapshot_files(output)
    write_json(output / "psd_manifest.json", {"files": files, "environment": environment(),
               "executables": {name: sha256(prefix / "bin" / name) for name in ("BayesWave", "BayesWavePost")}})
    print("Prepared exact-input BayesWave bundle:", output)
    print("Run psd run in the BayesWave environment inside a compute allocation. This command submitted nothing.")
    return 0


def validate_fft_export(path, reference):
    import numpy as np
    raw = np.loadtxt(path, usecols=(0, 1, 2))
    if len(raw) not in (len(reference["frequency"]), len(reference["frequency"])-1):
        raise ValueError("BayesWave Fourier export has an unexpected number of bins")
    f = reference["frequency"][:len(raw)]
    np.testing.assert_allclose(raw[:, 0], f, rtol=0, atol=.0051)
    band = (f >= 23) & (f <= 1600)
    observed = raw[:, 1]+1j*raw[:, 2]
    expected = reference["fft"][:len(raw)]
    scale = np.vdot(expected[band], observed[band])/np.vdot(expected[band], expected[band])
    residual = np.linalg.norm(observed[band]-scale*expected[band])/np.linalg.norm(observed[band])
    energy = float(reference["window_energy"])
    if not np.isfinite(residual) or residual > .005 or min(abs(scale-x) for x in (1., np.sqrt(energy), 1/np.sqrt(energy))) > .005:
        raise ValueError("BayesWave Fourier export differs from the exact input; do not use PSD products")
    return {"relative_residual": float(residual), "scale_real": float(scale.real), "scale_imag": float(scale.imag)}


def stage_products(bundle, config):
    """Validate the entire network before exposing any sampler-ready directory."""
    import numpy as np
    pending = bundle / ".products.pending"
    pending.mkdir(exist_ok=False)
    files = {}
    for detector in config["detectors"]:
        raw = bundle / "work/post/noise" / f"noise_median_PSD_{detector}.dat"
        physical = bundle / "work/post/noise" / f"noise_median_PSD_forLI_{detector}.dat"
        a, b = np.loadtxt(raw, usecols=(0, 1)), np.loadtxt(physical, usecols=(0, 1))
        with np.load(bundle / "reference" / f"{detector}.npz", allow_pickle=False) as reference:
            frequency = reference["frequency"]
            if a.ndim != 2 or b.shape != a.shape or len(a) not in (len(frequency), len(frequency)-1):
                raise ValueError("Posterior PSD row count/shape differs from the original FFT grid")
            np.testing.assert_allclose(a[:, 0], frequency[:len(a)], rtol=0, atol=.0051)
        np.testing.assert_allclose(a[:, 0], b[:, 0], rtol=0, atol=0)
        np.testing.assert_allclose(a[:, 1]*2/128, b[:, 1], rtol=1e-5, atol=0)
        raw_psd(raw)
        target = pending / f"glitch_median_PSD_{detector}.dat"
        expected = sha256(raw)
        shutil.copy2(raw, target)
        if sha256(target) != expected or sha256(raw) != expected:
            raise ValueError("PSD changed during product staging")
        files[target.name] = expected
    write_json(pending / "psd.json", {"event": config["event"], "files": files,
               "convention": "raw_bayeswave_power_2_over_128", "estimator": "BayesWave1.1.3 noiseOnly/noClean",
               "config_sha256": sha256(bundle / "psd_config.json"), "metadata_sha256": config["metadata_sha256"],
               "strain": config["strain"], "bundle_result": "../run_result.json",
               "note": "glitch_median filename is only the preserved sampler API, not a glitch-model fit or PSD marginalization"})
    return pending


def run(args):
    import numpy as np
    if not os.environ.get("SLURM_JOB_ID") and not args.local:
        raise ValueError("Run CPU BayesWave on a compute node, or explicitly select --local on your own workstation")
    bundle = args.bundle.resolve()
    if Path(__file__).resolve() != bundle / "source/lc_pipeline/psd.py":
        raise ValueError("Execute this bundle's frozen source/run.py for the PSD run, not the mutable checkout")
    config, manifest = read_json(bundle / "psd_config.json"), read_json(bundle / "psd_manifest.json")
    check_hashes(bundle, manifest["files"])
    if environment() != manifest["environment"]:
        raise ValueError("BayesWave Python/library environment changed since preparation")
    if read_json(bundle / "commands.json")["commands"] != commands(bundle, config):
        raise ValueError("Command list or bundle location changed; do not move a prepared BayesWave bundle")
    for name, digest in manifest["executables"].items():
        if sha256(Path(config["bayeswave_prefix"]) / "bin" / name) != digest:
            raise ValueError("BayesWave executable changed")
    for item in config["strain"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Input strain changed since PSD preparation")
    write_json(bundle / "run_claim.json", {"created_utc": now(), "purpose": "No duplicate fit/postprocessing"})
    logs = bundle / "logs"
    stages = []
    try:
        logs.mkdir()
        # BayesWave's chdir(outputDir) does not create it or reliably fail if absent.
        # Create it explicitly BEFORE either binary can write relative outputs.
        (bundle / "work").mkdir(exist_ok=False)
        for index, command in enumerate(commands(bundle, config)):
            with (logs / f"stage_{index}.log").open("x") as stream:
                result = subprocess.run(command, cwd=bundle, stdout=stream, stderr=subprocess.STDOUT)
            stages.append({"command": command, "exit_code": result.returncode})
            if result.returncode:
                raise ValueError(f"BayesWave stage {index} failed; see {logs}")
            checks = {}
            for detector in config["detectors"]:
                export = bundle / "work/waveforms" / f"fourier_domain_data_{detector}.dat"
                shutil.copy2(export, logs / f"stage_{index}_{detector}_fourier_data.dat")
                with np.load(bundle / "reference" / f"{detector}.npz", allow_pickle=False) as reference:
                    checks[detector] = validate_fft_export(export, reference)
            write_json(bundle / f"stage_{index}_input_check.json", checks)
        pending = stage_products(bundle, config)
        check_hashes(bundle, manifest["files"])
        products = bundle / "products"
        if products.exists():
            raise ValueError("Refusing to overwrite an existing products directory")
        pending.rename(products)
        write_json(bundle / "run_result.json", {"status": "completed_with_input_and_unit_checks", "stages": stages,
                   "production_validated": False, "caveat": "Estimator/conditioning differs from historical PSDs. Calibrate with GW170817 and repeatability/held-out checks before scientific use."})
    except Exception as error:
        write_json(bundle / "run_result.json", {"status": "failed", "stages": stages, "error": str(error)})
        raise
    print("Raw sampler-compatible products:", products)
    return 0