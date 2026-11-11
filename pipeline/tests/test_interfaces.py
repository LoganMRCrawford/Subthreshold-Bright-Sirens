"""Readable-interface regression tests; parsing and validation are host-only."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc_pipeline import cli, common, psd, search


class InterfaceTests(unittest.TestCase):
    def test_all_command_groups_keep_public_argument_contracts(self):
        cases = [
            (["sky", "acquire", "--catalogue", "events.csv", "--all", "--output", "sky"], "sky", "acquire"),
            (["data", "import", "--source", "source", "--output", "output"], "data", "import"),
            (["data", "fetch", "--event", "GRB190728271", "--gps", "1248330654", "--detectors", "H1", "L1", "--output", "output"], "data", "fetch"),
            (["data", "gw170817", "--source-dir", "raw", "--output", "GRB170817529"], "data", "gw170817"),
            (["psd", "import", "--source", "raw", "--data", "data", "--convention", "raw-2-over-128", "--output", "psd"], "psd", "import"),
            (["psd", "prepare", "--data", "data", "--prefix", "env", "--output", "bundle"], "psd", "prepare"),
            (["psd", "run", "--bundle", "bundle"], "psd", "run"),
            (["search", "prepare", "--config", "config.json", "--output", "bundle"], "search", "prepare"),
            (["search", "verify", "--bundle", "bundle"], "search", "verify"),
            (["search", "run", "--bundle", "bundle", "--task", "0", "--resume"], "search", "run"),
            (["search", "submit", "--bundle", "bundle", "--resume-task", "0"], "search", "submit"),
            (["search", "summarize", "--bundle", "bundle", "--output", "report"], "search", "summarize"),
        ]
        for tokens, group, action in cases:
            with self.subTest(command=tokens[:2]):
                args = cli.parser().parse_args(tokens)
                self.assertEqual((args.group, args.action), (group, action))
        self.assertNotIn("jax", sys.modules)

    def test_normalization_does_not_mutate_shared_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "search.json"
            raw = common.read_json(common.PIPELINE / "examples/search.json")
            path.write_text(json.dumps(raw))
            first = search.normalize_config(path)
            first["sampling"]["n_live"] = 9999
            first["slurm"]["hours"] = 99
            first["limitations"].clear()
            second = search.normalize_config(path)
            self.assertEqual(second["sampling"], raw["sampling"])
            self.assertEqual(second["slurm"]["hours"], raw["slurm"]["hours"])
            self.assertEqual(len(second["limitations"]), 5)
            self.assertEqual([cell["task"] for cell in second["cells"]], list(range(6)))

    def test_named_config_validators_retain_rejection_boundaries(self):
        for raw in ({"sampling": []}, {"slurm": []}, {"sampling": {"n_live": True}},
                    {"sampling": {"n_live": 50, "num_delete": 50}},
                    {"sampling": {"stop": -6.9}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                search._sampling_settings(raw)
        for given in ({"cpus": 0}, {"hours": False}, {"account": "bad\nname"}, {"extra": 1}):
            with self.subTest(given=given), self.assertRaises(ValueError):
                search._slurm_settings(given)

    def test_psd_dispatch_rejects_bad_lengths_before_implementation(self):
        args = SimpleNamespace(group="psd", action="prepare", iterations=0, burnin=0)
        with patch.object(psd, "prepare", side_effect=AssertionError("Never prepare")):
            with self.assertRaises(ValueError):
                cli.dispatch(args)

    def test_completed_summary_helper_keeps_identity_and_numeric_guards(self):
        config = {"sampling": dict(search.SAMPLING_DEFAULTS),
                  "events": [{"event": "GRB170817529", "psd_source": "bayesline"}]}
        cell = {"event": "GRB170817529", "prior": "whole_sky", "seed": 1}
        result = {"prior": "whole_sky", "psd_source": "bayesline", "remaining_logZ_gap": -3.1,
                  "log_bayes_factor_cbc_noise": 2.0, "log_bayes_factor_std": 0.2,
                  "noise_log_density": 100.0}
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            meta = {**cell, **config["sampling"], "psd_source": "bayesline"}
            (directory / "run_metadata.json").write_text(json.dumps(meta))
            (directory / "posterior.csv").write_text("synthetic fixture")
            (directory / "checkpoint.pkl").write_bytes(b"not decoded by scalar summary")
            path = directory / "evidence_summary.json"
            path.write_text(json.dumps(result))
            self.assertEqual(search._completed_summary(directory, cell, config), result)
            for key, value in (("prior", "grb_map"), ("psd_source", "welch"),
                               ("remaining_logZ_gap", -2.0), ("log_bayes_factor_std", -1.0)):
                changed = copy.deepcopy(result)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    search._completed_summary(directory, cell, config)


if __name__ == "__main__":
    unittest.main()