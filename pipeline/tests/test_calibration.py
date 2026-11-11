"""Pinned calibration acquisition tests; synthetic files, no public downloads."""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc_pipeline import calibration, cli, common, data


def fixture(directory: Path) -> dict:
    """Use small deterministic public-format files for the exact extraction path."""
    release = copy.deepcopy(calibration.load_release())
    release.update(
        sample_rate_hz=4,
        release_start_gps=100,
        release_duration_s=8,
        cache_start_gps=102,
        cache_samples=16,
    )
    for offset, detector in enumerate(calibration.NETWORK):
        source = release["detectors"][detector]
        values = np.arange(32, dtype=np.float64) + offset / 10
        path = directory / source["filename"]
        with h5py.File(path, "w") as stream:
            strain = stream.create_dataset("strain/Strain", data=values)
            strain.attrs.update(Xstart=100, Xspacing=0.25)
        source.update(
            bytes=path.stat().st_size,
            sha256=common.sha256(path),
            sample_sha256=calibration.sample_digest(values[8:24]),
        )
    return release


class CalibrationTests(unittest.TestCase):
    def test_pinned_release_scope_and_cli(self):
        release = calibration.load_release()
        self.assertEqual(release["event"], "GRB170817529")
        self.assertEqual(release["sample_rate_hz"], 4096)
        self.assertEqual(release["cache_samples"], 4784128)
        self.assertEqual(set(release["detectors"]), {"H1", "L1", "V1"})
        for source in release["detectors"].values():
            self.assertIn("O1_O2-Preliminary/GW170817/v2/", source["url"])
            self.assertIn("LOSC_CLN_4_V1", source["filename"])
            self.assertEqual(len(source["sha256"]), 64)
        args = cli.parser().parse_args(["data", "gw170817", "--output", "new/GRB170817529"])
        self.assertIsNone(args.source_dir)
        with patch.object(calibration, "acquire", return_value=0) as acquire:
            self.assertEqual(cli.main(["data", "gw170817", "--output", "new/GRB170817529"]), 0)
            acquire.assert_called_once()
        self.assertNotIn("jax", sys.modules)

    def test_generic_fetch_still_refuses_anchor(self):
        with self.assertRaisesRegex(ValueError, "data gw170817"):
            data.fetch(SimpleNamespace(event="GRB170817529"))

    def test_metadata_satisfies_fixed_contract(self):
        release = calibration.load_release()
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / release["event"]
            directory.mkdir()
            import pickle
            path = directory / f"{release['event']}_metadata.pkl"
            path.write_bytes(pickle.dumps(calibration.event_metadata(release)))
            meta = common.metadata(directory)
            self.assertEqual(meta["detectors"], ["H1", "L1", "V1"])
            self.assertEqual(meta["GPS"], release["gamma_trigger_gps"])

    def test_exact_local_bytes_and_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            release = fixture(root)
            source = release["detectors"]["H1"]
            with patch("requests.Session.get", side_effect=AssertionError("No network")):
                calibration.acquire_release_file(None, source, root / "copy.hdf5", root)
            self.assertEqual(common.sha256(root / "copy.hdf5"), source["sha256"])
            source["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                calibration.acquire_release_file(None, source, root / "bad.hdf5", root)
            self.assertFalse((root / "bad.hdf5").exists())

    def test_exact_extraction_and_sample_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            release = fixture(root)
            source = release["detectors"]["H1"]
            record = calibration.extract_cache(root / source["filename"], release, source, root / "cache.hdf5")
            actual = TimeSeries.read(root / "cache.hdf5", format="hdf5")
            np.testing.assert_array_equal(actual.value, np.arange(8, 24, dtype=float))
            self.assertEqual(float(actual.t0.value), 102)
            self.assertTrue(record["matches_historical_reviewed_samples"])
            source["sample_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                calibration.extract_cache(root / source["filename"], release, source, root / "bad.hdf5")
            self.assertFalse((root / "bad.hdf5").exists())

    def test_wrong_grid_and_nonfinite_samples_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            release = fixture(root)
            source = release["detectors"]["H1"]
            path = root / source["filename"]
            with h5py.File(path, "a") as stream:
                stream["strain/Strain"].attrs["Xspacing"] = 0.5
            with self.assertRaises(ValueError):
                calibration.extract_cache(path, release, source, root / "wrong-grid.hdf5")
            with h5py.File(path, "a") as stream:
                stream["strain/Strain"].attrs["Xspacing"] = 0.25
                stream["strain/Strain"][10] = np.nan
            with self.assertRaises(ValueError):
                calibration.extract_cache(path, release, source, root / "nan.hdf5")

    def test_download_size_checksum_and_https_guards(self):
        payload = b"reviewed release bytes"
        source = {"url": "https://gwosc.org/pinned.hdf5", "bytes": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest()}

        class Response:
            url = source["url"]
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, size):
                return iter([payload])

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            response = Response()
            session = SimpleNamespace(get=lambda *args, **kwargs: response)
            calibration.acquire_release_file(session, source, root / "correct", None)
            for name, changes in (("too-large", {"bytes": len(payload) - 1}),
                                  ("truncated", {"bytes": len(payload) + 1}),
                                  ("wrong-hash", {"sha256": "0" * 64})):
                with self.assertRaises(ValueError):
                    calibration.acquire_release_file(session, {**source, **changes}, root / name, None)
            response.url = "http://unsafe.example/file"
            with self.assertRaises(ValueError):
                calibration.acquire_release_file(session, source, root / "insecure", None)

    def test_full_network_published_only_after_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = root / "raw"
            inputs.mkdir()
            release = fixture(inputs)
            output = root / "prepared/GRB170817529"
            args = SimpleNamespace(output=output, source_dir=inputs)
            with patch.object(calibration, "load_release", return_value=release), \
                 patch.object(data, "inspect", return_value=({}, {})), \
                 patch("requests.Session.get", side_effect=AssertionError("No network")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(calibration.acquire(args), 0)
            record = json.loads((output / "data.json").read_text())
            self.assertEqual(len(record["channels"]), 3)
            self.assertEqual(record["source_mode"], "pinned_local_files")
            self.assertTrue((output / "release.json").exists())
            for detector in calibration.NETWORK:
                self.assertTrue((output / f"GRB170817529_{detector}.hdf5").exists())
            with self.assertRaises(FileExistsError):
                calibration.acquire(args)

    def test_failed_network_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = root / "GRB170817529"
            args = SimpleNamespace(output=output, source_dir=None)
            with patch.object(calibration, "prepare_cache", side_effect=ValueError("bad L1 release")):
                with self.assertRaisesRegex(ValueError, "no cache published"):
                    calibration.acquire(args)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(root.glob(".GRB170817529.pending-*/failure.json"))), 1)
            with self.assertRaises(FileExistsError):
                calibration.acquire(args)


if __name__ == "__main__":
    unittest.main()