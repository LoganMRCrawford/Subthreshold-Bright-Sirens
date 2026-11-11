"""Released GBM differential probability acquisition, with explicit blockers."""

import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, urlsplit

from .common import event_name, new_directory, now, read_json, resolve_path, sha256, write_json

S3 = "https://nasa-heasarc.s3.amazonaws.com/"
ANCHOR = "GRB170817529"
ANCHOR_GPS = 1187008884.474624
ANCHOR_SOURCE = {
    "url": "https://gammaray.nsstc.nasa.gov/gbm/science/grbs/grb170817a/gbuts_healpix_systematic.fit",
    "sha256": "fa07a0a83641b2c33198544f785766df037d54ef390ceba310ca5c1371990f4e",
    "method": "GBM targeted search, systematic-error product",
    "event_binding": "https://gammaray.nsstc.nasa.gov/gbm/science/grbs/grb170817a.html",
}
MAX_BYTES = 64 * 1024 * 1024


def catalogue(path):
    """Accept SFM's CSV schema without importing or modifying its code."""
    rows = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            event = event_name(row.get("GRB", row.get("event")))
            gps = float(row.get("GPS", row.get("gamma_trigger_gps", "nan")))
            if event in rows or not math.isfinite(gps) or gps <= 0:
                raise ValueError("Duplicate event or invalid trigger GPS in catalogue")
            rows[event] = {"event": event, "gamma_trigger_gps": gps, "run": row.get("LIGO_run", "unspecified")}
    if not rows:
        raise ValueError("Empty catalogue")
    return rows


def discover(session, event):
    """Choose the latest current released HEALPix file ONCE; no numeric fallback."""
    trigger = event_name(event)[3:]
    prefix = f"fermi/data/gbm/triggers/20{trigger[:2]}/bn{trigger}/current/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = session.get(S3, params=params, timeout=(10, 40))
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        keys += [node.text for node in tree.findall(".//{*}Contents/{*}Key")]
        truncated = tree.findtext("{*}IsTruncated") == "true"
        if not truncated:
            break
        token = tree.findtext("{*}NextContinuationToken")
        if not token:
            raise ValueError("Truncated archive listing without continuation token")
    pattern = re.compile(rf"glg_healpix_all_bn{trigger}_v(\d+)\.fits?")
    candidates = [(int(m.group(1)), key) for key in keys
                  if (m := pattern.fullmatch(Path(key).name)) and key.startswith(prefix)]
    if not candidates:
        raise ValueError("No released differential HEALPix product in the mission current directory; PMAP/contours are not a substitute")
    _, key = max(candidates)
    return {"url": S3 + quote(key, safe="/"), "method": "GBM mission localization",
            "event_binding": f"Mission archive trigger bn{trigger}",
            "selection": "Latest version in current; invalid latest products are blocked, not replaced by older versions"}


def download(session, source, destination, base):
    """Bounded transfer or explicit local import; raw downloaded bytes retained."""
    if "path" in source:
        local = resolve_path(source["path"], base)
        if local.stat().st_size > MAX_BYTES:
            raise ValueError("Localization exceeds size limit")
        with local.open("rb") as incoming, destination.open("xb") as outgoing:
            import shutil
            shutil.copyfileobj(incoming, outgoing)
    else:
        if urlsplit(source["url"]).scheme != "https":
            raise ValueError("Only HTTPS downloads or explicit local paths are accepted")
        with session.get(source["url"], stream=True, timeout=(10, 90)) as response:
            response.raise_for_status()
            if int(response.headers.get("Content-Length", 0)) > MAX_BYTES:
                raise ValueError("Localization exceeds size limit")
            count = 0
            with destination.open("xb") as stream:
                for block in response.iter_content(1024 * 1024):
                    count += len(block)
                    if count > MAX_BYTES:
                        raise ValueError("Localization exceeds size limit")
                    stream.write(block)
    digest = sha256(destination)
    if source.get("sha256") and digest != source["sha256"]:
        raise ValueError("Localization source checksum differs from pinned release")
    return digest


def validate_release(path, event, gps, source):
    """Recheck original released bytes, their binding and the strict map reader."""
    from astropy.io import fits
    from astropy.time import Time
    from grb_localization import HealpixLocalization
    event_name(event)
    if not math.isfinite(float(gps)) or gps <= 0:
        raise ValueError("Trigger GPS must be finite and positive")
    if source.get("sha256") and sha256(path) != source["sha256"]:
        raise ValueError("Released FITS checksum differs from its source record")
    localization = HealpixLocalization.from_fits(path)
    if localization.nside <= 0 or localization.nside & (localization.nside-1):
        raise ValueError("HEALPix NSIDE must be a positive power of two")
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        if "GBM" not in str(header.get("LOC_SRC", "")).upper():
            raise ValueError("Source does not identify a GBM localization")
        if event == ANCHOR:
            if (sha256(path) != ANCHOR_SOURCE["sha256"] or source.get("event_binding") != ANCHOR_SOURCE["event_binding"]
                    or abs(gps-ANCHOR_GPS) > .001):
                raise ValueError("GW170817 requires the pinned mission-targeted-search product and event-binding reference")
        else:
            if header.get("OBJECT") != event:
                raise ValueError("Released FITS OBJECT differs from requested event")
            met = float(header.get("TRIGTIME", "nan"))
            if not math.isfinite(met) or abs(float(Time("2001-01-01T00:00:00", scale="utc").gps)+met-gps) > .001:
                raise ValueError("Released FITS trigger differs from catalogue GPS by more than 1 ms")
        headers = {k: header.get(k, "") for k in ("OBJECT", "CREATOR", "LOC_SRC", "FILENAME", "TRIGTIME", "RADECSYS", "EQUINOX")}
    return localization, headers


def prepare_product(path, event, gps, source, directory, samples=20000, seed=170817):
    import numpy as np
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive integer sample count and nonnegative integer seed required")
    artifact = directory / "grb_probability_inputs.npz"
    if artifact.exists():
        raise FileExistsError("Refusing to overwrite an existing probability package")
    localization, headers = validate_release(path, event, gps, source)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(event[3:])]))
    points = localization.sample_sky(samples, rng)
    np.savez_compressed(artifact, probability_per_native_pixel=localization.probability,
                        nside=localization.nside, ordering=localization.order,
                        native_frame=localization.native_frame, equinox=localization.equinox,
                        ra_icrs_rad=points[:, 0], dec_icrs_rad=points[:, 1],
                        tc_minus_trigger_s=rng.uniform(-10., 0., samples), gamma_trigger_gps=gps)
    regions = []
    for level in (.5, .9, .95):
        mask = localization.credible_region(level)
        regions.append({"level": level, "enclosed_probability": float(localization.probability[mask].sum()),
                        "area_deg2": float(mask.sum()*localization.pixel_area_sr*(180/np.pi)**2)})
    return {"event": event, "probability_usable": True, "gamma_trigger_gps": gps,
            "coordinate_api": "ICRS radians; native-pixel frame retained",
            "timing": {"tc_minus_trigger_support_s": [-10., 0.], "density_per_s": .1,
                       "status": "Assumed independent uniform delay; not measured; no geocentre correction"},
            "artifact": artifact.name, "artifact_sha256": sha256(artifact),
            "released_probability_source": {**source, "cache_path": path.name, "sha256": sha256(path), "headers": headers},
            "native_frame": localization.native_frame, "equinox": localization.equinox,
            "nside": localization.nside, "ordering": localization.order,
            "probability_sum": localization.probability_sum, "credible_regions": regions,
            "sample_count": samples, "seed": seed, "prior_usage": "sequential_posterior_once",
            "limitation": "Released PROBABILITY only; no PMAP/SIGNIFICANCE conversion, clipping, de-prioring or mission-systematics validation. GW170817 targeted search differs from routine mission localizations."}


def acquire(args):
    import requests
    rows = catalogue(args.catalogue)
    if not args.events and not args.all:
        raise ValueError("Select --events or explicitly request --all")
    selected = list(rows) if args.all else list(dict.fromkeys(args.events))
    for event in selected:
        if event not in rows:
            raise ValueError(f"Unknown catalogue event: {event}")
    # Every collection includes the calibration anchor; no catalogue-wide search is submitted.
    if ANCHOR not in selected:
        selected.insert(0, ANCHOR)
    if ANCHOR in rows and abs(rows[ANCHOR]["gamma_trigger_gps"]-ANCHOR_GPS) > .001:
        raise ValueError("Catalogue GW170817 trigger mismatch")
    rows.setdefault(ANCHOR, {"event": ANCHOR, "gamma_trigger_gps": ANCHOR_GPS, "run": "O2"})
    overrides = read_json(args.sources) if args.sources else {}
    if not isinstance(overrides, dict) or set(overrides)-set(rows):
        raise ValueError("Source overrides must map known catalogue event identifiers to objects")
    if any(not isinstance(value, dict) for value in overrides.values()):
        raise ValueError("Each source override must be an object")
    base = args.sources.resolve().parent if args.sources else Path.cwd()
    output = new_directory(args.output)
    records = []
    with requests.Session() as session:
        for event in selected:
            directory = output / event
            directory.mkdir()
            source = overrides.get(event)
            record = {**rows[event], "probability_usable": False}
            try:
                if source is None:
                    if args.offline:
                        raise ValueError("Offline acquisition requires an explicit local source entry")
                    source = dict(ANCHOR_SOURCE) if event == ANCHOR else discover(session, event)
                if args.offline and "path" not in source:
                    raise ValueError("Offline source must contain a local path")
                raw = directory / "released_probability.fit"
                digest = download(session, source, raw, base)
                write_json(directory / "source.json", {**source, "sha256": digest, "acquired_utc": now()})
                record = {**rows[event], **prepare_product(raw, event, rows[event]["gamma_trigger_gps"], source,
                                                          directory, args.samples, args.seed)}
            except (OSError, ValueError, KeyError, TypeError, ET.ParseError, requests.RequestException) as error:
                record["probability_blocker"] = str(error)
                record["attempted_source"] = source
            write_json(directory / "grb_inputs.json", record)
            records.append(record)
            print(event, "READY" if record["probability_usable"] else "BLOCKED: " + record["probability_blocker"], flush=True)
    write_json(output / "acquisition.json", {"created_utc": now(), "catalogue_sha256": sha256(args.catalogue),
               "usable": sum(r["probability_usable"] for r in records), "requested": len(records), "events": records})
    with (output / "acquisition.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["event", "probability_usable", "probability_blocker"])
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in writer.fieldnames})
    print("Acquisition index:", output / "acquisition.json")
    return 0 if all(r["probability_usable"] for r in records) else 2