"""Import existing event caches or explicitly fetch modern public GW strain."""

import math
import pickle
import shutil
from pathlib import Path

from .common import event_name, metadata, new_directory, now, sha256, write_json


def inspect(directory: Path) -> tuple[dict, dict]:
    """Validate finite strain, cadence and each exact fixed-model data window."""
    import numpy as np
    from gwpy.timeseries import TimeSeries
    meta = metadata(directory)
    channels = {}
    for detector in meta["detectors"]:
        path = directory / f"{meta['GRB']}_{detector}.hdf5"
        series = TimeSeries.read(path, format="hdf5")
        if float(series.sample_rate.value) != 4096 or not np.isfinite(series.value).all():
            raise ValueError(f"Require finite 4096-Hz strain: {path}")
        windows = (
            (meta["start"], meta["end"], 524288),
            (meta["psd_start"], meta["psd_end"], 4194304),
        )
        for start, end, samples in windows:
            cropped = series.crop(float(start), float(end))
            start_offset_samples = abs(float(cropped.t0.value) - float(start)) * 4096
            if len(cropped) != samples or start_offset_samples > 1.01:
                raise ValueError(f"Incomplete or misaligned required data window: {path}")
        channels[detector] = {
            "sha256": sha256(path),
            "sample_rate_hz": 4096,
            "start_gps": float(series.t0.value),
            "samples": len(series),
        }
    return meta, channels


def import_data(args) -> int:
    """Copy reviewed inputs; this does not promote them to a new data release."""
    source = args.source.resolve()
    meta, channels = inspect(source)
    if args.output.name != meta["GRB"]:
        raise ValueError("Output event directory must be named after the GRB")
    output = new_directory(args.output)
    names = [f"{meta['GRB']}_metadata.pkl"] + [f"{meta['GRB']}_{d}.hdf5" for d in meta["detectors"]]
    for name in names:
        shutil.copy2(source / name, output / name)
        if sha256(source / name) != sha256(output / name):
            raise ValueError("Data changed during import")
    write_json(output / "data.json", {"created_utc": now(), "source_directory": str(source),
               "metadata_sha256": sha256(output / names[0]), "channels": channels,
               "note": "Byte-identical import; no new release, DQ, cleaning or conditioning validation"})
    print("Imported event cache:", output)
    return 0


def fetch(args) -> int:
    """Fetch modern run data only; the calibration anchor has a pinned route."""
    import numpy as np
    from gwpy.timeseries import TimeSeries
    event = event_name(args.event)
    if event == "GRB170817529":
        raise ValueError("GW170817 must use 'data gw170817' for the pinned reviewed v2 cleaned release, or 'data import' for a reviewed cache; default run-release substitution is refused.")
    gps = float(args.gps)
    if not math.isfinite(gps) or gps <= 0 or args.output.name != event:
        raise ValueError("Positive GPS and output directory ending in the event name are required")
    network = list(dict.fromkeys(args.detectors))
    output = new_directory(args.output)
    meta = {"GRB": event, "GPS": gps, "start": gps-128., "end": gps,
            "psd_start": gps-1168., "psd_end": gps-144., "detectors": network,
            "duration": 128, "psd_duration": 1024, "psd_pad": 16}
    for detector in network:
        series = TimeSeries.fetch_open_data(detector, math.floor(gps-1168.), math.ceil(gps),
                                            sample_rate=4096, cache=True, verbose=True)
        if not np.isfinite(series.value).all() or float(series.sample_rate.value) != 4096:
            raise ValueError(f"Nonfinite or wrong-cadence data for {detector}; no complete metadata will be written")
        series.write(output / f"{event}_{detector}.hdf5", format="hdf5")
    # Primitive metadata remains compatible with the preserved sampler.
    with (output / f"{event}_metadata.pkl").open("xb") as stream:
        pickle.dump(meta, stream)
    _, channels = inspect(output)
    write_json(output / "data.json", {"created_utc": now(), "acquisition": "GWpy fetch_open_data default public run release",
               "metadata_sha256": sha256(output / f"{event}_metadata.pkl"), "channels": channels,
               "status": "Data acquisition only: review release version, DQ, hardware injections, gaps, calibration and filtering before science use"})
    print("Fetched event cache (scientific input review still required):", output)
    return 0