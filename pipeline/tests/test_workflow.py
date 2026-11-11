"""Offline unified-workflow tests. No JAX, likelihood evaluation, Slurm or network."""

import copy
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.io import fits
from astropy.time import Time

from lc_pipeline import cli, common, psd, search, sky


def write_fixture(path, event="GRB190728271", probability=None):
    primary = fits.PrimaryHDU()
    primary.header["LOC_SRC"] = "Fermi, GBM"
    primary.header["OBJECT"] = event
    primary.header["TRIGTIME"] = 1248330654.8524225-float(Time("2001-01-01T00:00:00", scale="utc").gps)
    primary.header["RADECSYS"] = "ICRS"
    extension = fits.BinTableHDU.from_columns([fits.Column(name="PROBABILITY", format="D",
                          array=np.ones(12)/12 if probability is None else probability)], name="HEALPIX")
    for key, value in {"INDXSCHM": "IMPLICIT", "COORDSYS": "C", "ORDERING": "NESTED", "NSIDE": 1}.items():
        extension.header[key] = value
    fits.HDUList([primary, extension]).writeto(path)


def config_file(directory):
    config = common.read_json(common.PIPELINE / "examples/search.json")
    path = directory / "search.json"
    path.write_text(json.dumps(config))
    return path, config


class WorkflowTests(unittest.TestCase):
    def test_imports_and_doctor_do_not_load_jax(self):
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(cli.main(["doctor"]), 0)
        self.assertFalse(json.loads(stream.getvalue())["jax_imported"])
        self.assertNotIn("jax", sys.modules)

    def test_exclusive_json_and_safe_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common.write_json(root / "record.json", {"a": 1})
            with self.assertRaises(FileExistsError):
                common.write_json(root / "record.json", {})
            common.check_hashes(root, {"record.json": common.sha256(root / "record.json")})
            with self.assertRaises(ValueError):
                common.check_hashes(root, {"../outside": "bad"})
            with self.assertRaises(ValueError):
                common.check_hashes(root, {})
            with self.assertRaises(ValueError):
                common.write_json(root / "nonfinite.json", {"gps": float("nan")})
            self.assertFalse((root / "nonfinite.json").exists())
            (root / "nonfinite.json").write_text('{"gps": NaN}')
            with self.assertRaises(ValueError):
                common.read_json(root / "nonfinite.json")

    def test_metadata_refuses_executable_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "GRB190728271"; directory.mkdir()
            p = directory / "GRB190728271_metadata.pkl"
            p.write_bytes(pickle.dumps(Path("/not-primitive")))
            with self.assertRaises(ValueError):
                common.metadata(directory)
            p.write_bytes(b"not a pickle")
            with self.assertRaises(ValueError):
                common.metadata(directory)

    def test_metadata_timing_contract(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "GRB190728271"; directory.mkdir()
            gps = 1248330654.8524225
            m = {"GRB": directory.name, "GPS": gps, "start": gps-128., "end": gps,
                 "psd_start": gps-1168., "psd_end": gps-144., "detectors": ["H1", "L1"]}
            p = directory / f"{directory.name}_metadata.pkl"
            p.write_bytes(pickle.dumps(m))
            self.assertEqual(common.metadata(directory), m)
            m["start"] -= 128
            p.write_bytes(pickle.dumps(m))
            with self.assertRaises(ValueError):
                common.metadata(directory)
            m["start"] += 128
            m["GPS"] = float("nan")
            p.write_bytes(pickle.dumps(m))
            with self.assertRaises(ValueError):
                common.metadata(directory)

    def test_catalogue_explicit_events_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.csv"
            p.write_text("GRB,GPS\nGRB190728271,1248330654.8524225\n")
            self.assertIn("GRB190728271", sky.catalogue(p))
            p.write_text(p.read_text()+"GRB190728271,1248330654.8524225\n")
            with self.assertRaises(ValueError):
                sky.catalogue(p)

    def test_discovery_uses_latest_current_probability_not_pmap(self):
        prefix = "fermi/data/gbm/triggers/2019/bn190728271/current/"
        keys = [prefix+f"glg_healpix_all_bn190728271_v{v:02d}.fit" for v in (0, 3, 1)]
        keys += [prefix+"glg_locprob_all_bn190728271_v99.fit"]
        xml = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
        xml += "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)+"</ListBucketResult>"
        session = NS(get=lambda *a, **k: NS(content=xml.encode(), raise_for_status=lambda: None))
        self.assertTrue(sky.discover(session, "GRB190728271")["url"].endswith("v03.fit"))
        empty = NS(get=lambda *a, **k: NS(content=b"<ListBucketResult/>", raise_for_status=lambda: None))
        with self.assertRaises(ValueError):
            sky.discover(empty, "GRB190728271")

    def test_local_download_is_pinned_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "input.fit"; p.write_bytes(b"abc")
            source = {"path": "input.fit", "sha256": common.sha256(p)}
            target = root / "copied.fit"
            self.assertEqual(sky.download(None, source, target, root), source["sha256"])
            with self.assertRaises(FileExistsError):
                sky.download(None, source, target, root)
            source["sha256"] = "wrong"
            with self.assertRaises(ValueError):
                sky.download(None, source, root / "bad.fit", root)

    def test_probability_product_and_coordinate_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "released_probability.fit"; write_fixture(p)
            record = sky.prepare_product(p, "GRB190728271", 1248330654.8524225,
                          {"method": "fixture", "event_binding": "fixture"}, root, samples=100)
            common.write_json(root / "grb_inputs.json", record)
            checked = search.validate_sky(root / "grb_probability_inputs.npz", record["event"], record["gamma_trigger_gps"])
            self.assertTrue(checked["probability_usable"])
            with np.load(root / "grb_probability_inputs.npz") as z:
                np.testing.assert_allclose(z["probability_per_native_pixel"].sum(), 1.)
                self.assertEqual(z["ra_icrs_rad"].shape, (100,))
                self.assertTrue(np.all((z["tc_minus_trigger_s"] >= -10) & (z["tc_minus_trigger_s"] <= 0)))
            with self.assertRaises(ValueError):
                search.validate_sky(root / "grb_probability_inputs.npz", record["event"], record["gamma_trigger_gps"]+1.)
            with self.assertRaises(FileExistsError):
                sky.prepare_product(p, record["event"], record["gamma_trigger_gps"], {}, root, samples=100)

    def test_invalid_probability_is_never_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "negative.fit"
            probability = np.ones(12)/12; probability[0] = -1e-12; probability[1] += 1/12+1e-12
            write_fixture(p, probability=probability)
            digest = common.sha256(p)
            with self.assertRaises(ValueError):
                sky.prepare_product(p, "GRB190728271", 1248330654.8524225, {}, root)
            self.assertEqual(common.sha256(p), digest)
            self.assertFalse((root / "grb_probability_inputs.npz").exists())

    def test_wrong_event_and_time_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "map.fit"; write_fixture(p)
            with self.assertRaises(ValueError):
                sky.prepare_product(p, "GRB240715239", 1248330654.8524225, {}, root)
            with self.assertRaises(ValueError):
                sky.prepare_product(p, "GRB190728271", 1248330655.8524225, {}, root)

    def test_acquisition_blocks_missing_anchor_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = root / "map.fit"; write_fixture(raw)
            cat = root / "events.csv"; cat.write_text("GRB,GPS\nGRB190728271,1248330654.8524225\n")
            sources = root / "sources.json"
            sources.write_text(json.dumps({"GRB190728271": {"path": "map.fit", "sha256": common.sha256(raw)}}))
            args = NS(catalogue=cat, events=["GRB190728271"], all=False, sources=sources, offline=True,
                      output=root / "prepared", samples=10, seed=3)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(sky.acquire(args), 2)
            index = common.read_json(args.output / "acquisition.json")
            self.assertEqual(index["requested"], 2)
            self.assertEqual(index["usable"], 1)
            self.assertFalse(index["events"][0]["probability_usable"])

    def test_relative_config_and_required_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, c = config_file(root)
            normalized = search.normalize_config(path)
            self.assertEqual(len(normalized["cells"]), 6)
            self.assertTrue(Path(normalized["events"][0]["data"]).is_absolute())
            self.assertEqual(normalized["sampling"]["n_live"], 500)
            for entry in c["events"]:
                entry["pairs"][0]["whole_seed"] = 42
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                search.normalize_config(path)
            _, c = config_file(root)
            c["events"] = c["events"][1:]; path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                search.normalize_config(path)
            _, c = config_file(root)
            c["distance_prior"] = [0., 800.]; path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                search.normalize_config(path)
            _, c = config_file(root)
            c["events"][0]["psd_source"] = "welch"; path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                search.normalize_config(path)
            path.write_text("[]")
            with self.assertRaises(ValueError):
                search.normalize_config(path)

    def test_command_preflight_separate_and_resume_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, _ = config_file(root); config = search.normalize_config(path)
            bundle = root / "bundle"
            normal, output = search.command(bundle, config, 0)
            resumed, _ = search.command(bundle, config, 0, resume=True)
            preflight, preoutput = search.command(bundle, config, 0, preflight=True)
            self.assertEqual(resumed, normal+["--resume"])
            self.assertNotEqual(output, preoutput)
            self.assertIn("--preflight", preflight)
            with self.assertRaises(ValueError):
                search.command(bundle, config, True)
            text = search.render_slurm(bundle, config, "/safe/python")
            for part in ("--signal=B:USR1@600", "--no-requeue", "exec /safe/python", "--time=12:00:00"):
                self.assertIn(part, text)

    def test_output_resume_guard_and_host_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            search.check_output(output, False)
            with self.assertRaises(ValueError):
                search.check_output(output, True)
            output.mkdir(); (output / "checkpoint.pkl").write_bytes(b"not unpickled")
            (output / "run_metadata.json").write_text("{}")
            search.check_output(output, True)
            (output / "posterior.csv").write_text("partial")
            with self.assertRaises(ValueError):
                search.check_output(output, True)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                search.run(NS(local_gpu=False))
            with self.assertRaises(ValueError):
                psd.run(NS(local=False))

    def test_bayeswave_grid_and_exact_command_inputs(self):
        self.assertEqual(psd.validate_fft_grid(), 524288)
        for high in (1600., 1024.):
            with self.assertRaises(ValueError):
                psd.validate_fft_grid(fmax=high)
        config = {"trigger_gps": 1000., "fit_start_gps": -40., "psd_start_gps": -936.,
                  "detectors": ["H1", "L1"], "bayeswave_prefix": "/env", "iterations": 5000, "burnin": 1000}
        bw, post = psd.commands(Path("/bundle"), config)
        self.assertIn("--noiseOnly", bw)
        self.assertIn("--noClean", bw)
        self.assertIn("--0noise", post)
        self.assertNotIn("--lite", post)
        for cmd in (bw, post):
            self.assertEqual(cmd[cmd.index("--H1-channel")+1], "H1:LC_EXACT_STRAIN")
            self.assertEqual(cmd[cmd.index("--H1-fhigh")+1], "2048")

    def test_raw_units_and_duplicate_frequencies(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.dat"
            np.savetxt(p, [[0., 64.], [23., 128.], [23., 256.], [2048., 64.]])
            f, s = psd.raw_psd(p)
            np.testing.assert_array_equal(f, [0., 23., 2048.])
            np.testing.assert_allclose(s, [1., 3., 1.])
            np.savetxt(p, [[23., 1.], [1600., 2.]])
            with self.assertRaises(ValueError):
                psd.raw_psd(p)

    def test_fft_export_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fft.dat"
            frequency = np.arange(2049.)
            expected = np.exp(1j*frequency/200.)
            ref = {"frequency": frequency, "fft": expected, "window_energy": np.array(1.)}
            np.savetxt(p, np.c_[frequency, expected.real, expected.imag])
            self.assertLess(psd.validate_fft_export(p, ref)["relative_residual"], 1e-12)
            changed = expected*np.exp(1j*frequency/100.)
            np.savetxt(p, np.c_[frequency, changed.real, changed.imag])
            with self.assertRaises(ValueError):
                psd.validate_fft_export(p, ref)

    def test_search_submission_ambiguity_cannot_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, _ = config_file(root); config = search.normalize_config(path)
            bundle = root / "bundle"; bundle.mkdir()
            args = NS(bundle=bundle, resume_task=None)
            with patch.object(search, "verify", return_value=(config, {})), patch.object(search.subprocess, "run") as run:
                run.side_effect = [NS(stdout="", returncode=0), NS(stdout="ok", stderr="", returncode=0),
                                   NS(stdout="ambiguous", stderr="", returncode=0)]
                with self.assertRaises(ValueError):
                    search.submit(args)
                self.assertTrue((bundle / "submission/receipt.json").exists())
                run.side_effect = None; run.return_value = NS(stdout="", returncode=0)
                with self.assertRaises(FileExistsError):
                    search.submit(args)

    def test_summary_keeps_incomplete_runs_and_all_completed_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, _ = config_file(root); config = search.normalize_config(path)
            bundle = root / "bundle"; bundle.mkdir()
            for cell, logb in zip(config["cells"][:2], [2., 3.5]):
                output = bundle / cell["output"]; output.mkdir(parents=True)
                common.write_json(output / "run_metadata.json", {**cell, **config["sampling"], "psd_source": "bayesline"})
                common.write_json(output / "evidence_summary.json", {"remaining_logZ_gap": -3.1,
                    "log_bayes_factor_cbc_noise": logb, "log_bayes_factor_std": .2, "noise_log_density": 100.,
                    "prior": cell["prior"], "psd_source": "bayesline"})
                (output / "posterior.csv").write_text("stub,not inference\n")
                (output / "checkpoint.pkl").write_bytes(b"not decoded")
            args = NS(bundle=bundle, output=root / "summary")
            with patch.object(search, "verify", return_value=(config, {})), redirect_stdout(io.StringIO()):
                self.assertEqual(search.summarize(args), 0)
            report = common.read_json(args.output / "summary.json")
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(len(report["completed_pairs"]), 1)
            self.assertAlmostEqual(report["completed_pairs"][0]["delta_logZ"], 1.5)
            self.assertEqual(sum(r["status"] == "not_started" for r in report["runs"]), 4)
            (bundle / config["cells"][0]["output"] / "posterior.csv").unlink()
            with patch.object(search, "verify", return_value=(config, {})):
                with self.assertRaises(ValueError):
                    search.summarize(NS(bundle=bundle, output=root / "bad-summary"))

    def test_sky_original_bytes_and_finite_trigger_are_rechecked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = root / "released_probability.fit"; write_fixture(raw)
            event, gps = "GRB190728271", 1248330654.8524225
            record = sky.prepare_product(raw, event, gps, {}, root, samples=20)
            package = root / "grb_probability_inputs.npz"
            common.write_json(root / "grb_inputs.json", record)
            with self.assertRaises(ValueError):
                search.validate_sky(package, event, float("nan"))
            with np.load(package, allow_pickle=False) as z:
                values = {key: np.array(z[key]) for key in z.files}
            values["probability_per_native_pixel"][0] += .01
            values["probability_per_native_pixel"][1] -= .01
            np.savez_compressed(package, **values)
            record["artifact_sha256"] = common.sha256(package)
            (root / "grb_inputs.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                search.validate_sky(package, event, gps)
            values["probability_per_native_pixel"] = np.ones(12)/12
            values["gamma_trigger_gps"] = float("nan")
            np.savez_compressed(package, **values)
            record["artifact_sha256"] = common.sha256(package)
            (root / "grb_inputs.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                search.validate_sky(package, event, gps)

    def test_search_run_refuses_mutable_checkout_before_jax(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "test", "JAX_PLATFORMS": "cuda"}, clear=True):
            with self.assertRaisesRegex(ValueError, "frozen"):
                search.run(NS(bundle=Path("/unused/bundle"), local_gpu=False))
        self.assertNotIn("jax", sys.modules)

    def test_ambiguous_initial_receipt_cannot_authorize_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, _ = config_file(root); config = search.normalize_config(path)
            bundle = root / "bundle"; bundle.mkdir()
            output = bundle / config["cells"][0]["output"]; output.mkdir(parents=True)
            (output / "checkpoint.pkl").write_bytes(b"trusted file is not decoded here")
            (output / "run_metadata.json").write_text("{}")
            common.write_json(bundle / "submission/receipt.json", {"exit_code": 0, "job_id": "ambiguous"})
            with patch.object(search, "verify", return_value=(config, {})), patch.object(search.subprocess, "run", return_value=NS(stdout="")) as run:
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    search.submit(NS(bundle=bundle, resume_task=0))
                self.assertEqual(run.call_count, 1)  # Only squeue, never sbatch.
                self.assertFalse((bundle / "resume").exists())

    def test_no_partial_psd_products_on_network_validation_failure(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d); work = bundle / "work/post/noise"; work.mkdir(parents=True)
            (bundle / "reference").mkdir()
            frequency = np.arange(2049.)
            for detector in ("H1", "L1"):
                np.savez(bundle / "reference" / f"{detector}.npz", frequency=frequency)
                power = np.ones_like(frequency)*64.
                np.savetxt(work / f"noise_median_PSD_{detector}.dat", np.c_[frequency, power])
                physical = power*2/128 if detector == "H1" else power
                np.savetxt(work / f"noise_median_PSD_forLI_{detector}.dat", np.c_[frequency, physical])
            config = {"detectors": ["H1", "L1"]}
            with self.assertRaises(AssertionError):
                psd.stage_products(bundle, config)
            self.assertFalse((bundle / "products").exists())
            self.assertFalse((bundle / ".products.pending/psd.json").exists())

    def test_bayeswave_products_grid_must_match_reference(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d); work = bundle / "work/post/noise"; work.mkdir(parents=True)
            (bundle / "reference").mkdir()
            frequency = np.arange(2049.)
            np.savez(bundle / "reference/H1.npz", frequency=frequency)
            shifted = frequency+.01
            np.savetxt(work / "noise_median_PSD_H1.dat", np.c_[shifted, np.ones_like(frequency)*64.])
            np.savetxt(work / "noise_median_PSD_forLI_H1.dat", np.c_[shifted, np.ones_like(frequency)])
            with self.assertRaises(AssertionError):
                psd.stage_products(bundle, {"detectors": ["H1"]})
            self.assertFalse((bundle / "products").exists())

    def test_valid_psd_network_is_staged_with_explicit_completion_link(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d); work = bundle / "work/post/noise"; work.mkdir(parents=True)
            (bundle / "reference").mkdir()
            frequency = np.arange(2049.)
            for detector in ("H1", "L1"):
                np.savez(bundle / "reference" / f"{detector}.npz", frequency=frequency)
                np.savetxt(work / f"noise_median_PSD_{detector}.dat", np.c_[frequency, np.ones_like(frequency)*64.])
                np.savetxt(work / f"noise_median_PSD_forLI_{detector}.dat", np.c_[frequency, np.ones_like(frequency)])
            config = {"detectors": ["H1", "L1"], "event": "GRB190728271", "metadata_sha256": "fixture", "strain": {}}
            common.write_json(bundle / "psd_config.json", config)
            pending = psd.stage_products(bundle, config)
            record = common.read_json(pending / "psd.json")
            self.assertEqual(len(record["files"]), 2)
            self.assertEqual(record["bundle_result"], "../run_result.json")
            common.check_hashes(pending, record["files"])
            self.assertFalse((bundle / "products").exists())

    def test_stage_one_requirements_do_not_request_incompatible_fork(self):
        text = (common.PIPELINE / "requirements-search.txt").read_text()
        actual = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertFalse(any("blackjax" in line.lower() for line in actual))
        self.assertIn("--no-deps", text)
        self.assertIn("2180e29ffb645b2c46b76c768229c7c24212446c", text)

    def test_bayeswave_working_directory_exists_before_binary_launch(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            config = {"trigger_gps": 1000., "fit_start_gps": -40., "psd_start_gps": -936.,
                      "detectors": ["H1"], "bayeswave_prefix": "/env", "iterations": 1000,
                      "burnin": 200, "strain": {}}
            common.write_json(bundle / "psd_config.json", config)
            common.write_json(bundle / "commands.json", {"commands": psd.commands(bundle, config)})
            manifest = {"files": common.snapshot_files(bundle), "environment": {}, "executables": {}}
            common.write_json(bundle / "psd_manifest.json", manifest)

            def binary_stub(command, **kwargs):
                self.assertTrue((bundle / "work").is_dir())
                self.assertEqual(command[command.index("--outputDir")+1], str(bundle / "work"))
                return NS(returncode=1)  # Stop before any scientific execution or exports.

            with patch.object(psd, "__file__", str(bundle / "source/lc_pipeline/psd.py")), \
                    patch.object(psd, "environment", return_value={}), \
                    patch.object(psd.subprocess, "run", side_effect=binary_stub) as run, \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "test"}, clear=True):
                with self.assertRaisesRegex(ValueError, "stage 0 failed"):
                    psd.run(NS(bundle=bundle, local=False))
                self.assertEqual(run.call_count, 1)
            self.assertEqual(common.read_json(bundle / "run_result.json")["status"], "failed")
            self.assertFalse((bundle / "products").exists())

    def test_sampler_is_unchanged(self):
        self.assertEqual(common.sha256(common.PIPELINE / "sample_grb_sequential.py"), search.SAMPLER_SHA)

    def test_unified_entrypoint_is_frozen_without_legacy_launcher(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, _ = config_file(root)
            text = search.render_slurm(root / "bundle", search.normalize_config(path), "/safe/python")
            self.assertIn("source/run.py", text)
            self.assertNotIn("source/lc.py", text)
            self.assertNotIn("LC_grb_informed_pipeline", text)
        self.assertTrue((common.PIPELINE / "run.py").is_file())

    def test_config_uses_new_work_root_not_historical_outputs(self):
        raw = common.read_json(common.PIPELINE / "examples/search.json")
        for event in raw["events"]:
            for key in ("data", "sky", "psd"):
                self.assertTrue(event[key].startswith("../work/"))
                self.assertNotIn("LC_", event[key])


if __name__ == "__main__":
    unittest.main()