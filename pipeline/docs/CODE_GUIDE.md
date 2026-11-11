# Reading and extending the maintained pipeline

## Start here, not in an old sampler

Use [../run.py](../run.py) for current commands and
[../README.md](../README.md) for the workflow. The SFM/LC root trees are
local historical material, sometimes restored for exact resume. Their
presence does not make them maintained alternatives.

The public command groups follow the data flow:

1. `sky acquire`: released probability map → validated probability package.
2. `data gw170817`, `data import`, or `data fetch`: reviewed strain → event cache.
3. `psd prepare` then `psd run`, or `psd import`: explicit raw PSD products.
4. `search prepare`: input checks → a new frozen bundle, **no submission**.
5. Frozen `search run` / `search submit`: GPU evaluation / explicit allocation.
6. Frozen `search summarize`: all tasks and completed predeclared pairs.

Never edit a frozen bundle or an environment used by active/resumable jobs
to pick up a source-code improvement. New source changes affect new bundles
only; the unmodified numerical engine is hash-checked before preparation.

## Module map

| File | Read this for | Main entry points / invariants |
|---|---|---|
| [../lc_pipeline/cli.py](../lc_pipeline/cli.py) | Public arguments and lazy command dispatch | Command groups are built separately. `dispatch()` imports only the chosen implementation; `doctor()` reads package metadata without JAX. |
| [../lc_pipeline/common.py](../lc_pipeline/common.py) | Paths, hashes, primitive metadata and JSON | Exclusive writes; nonfinite JSON refused; metadata enforces the fixed timing/network contract. Config paths are relative to the config file, not the shell. |
| [../lc_pipeline/calibration.py](../lc_pipeline/calibration.py) | Archive-independent GW170817 preparation | `load_release()` → bounded `acquire_release_file()` → exact `extract_cache()` → full-network publication. Original-file and canonical sample hashes are different checks. |
| [../lc_pipeline/releases/gw170817-v2.json](../lc_pipeline/releases/gw170817-v2.json) | Reviewed calibration release identity | Public URLs, byte sizes, file/sample hashes and exact sample interval; changing this is a release change, not a cosmetic edit. |
| [../lc_pipeline/data.py](../lc_pipeline/data.py) | Existing-cache import and modern run data | `inspect()` checks cadence, finite values and both requested windows. Generic fetching still refuses GW170817 substitution. |
| [../lc_pipeline/sky.py](../lc_pipeline/sky.py) | GBM discovery, FITS validation and packaging | Use actual released probabilities; failed latest releases stay blocked. Include GW170817 in each collection. |
| [../lc_pipeline/psd.py](../lc_pipeline/psd.py) | Exact-input BayesWave and raw units | Keep the full fitting grid; validate fit/Post Fourier inputs; publish only a complete checked detector network. |
| [../lc_pipeline/search.py](../lc_pipeline/search.py) | Fixed-model task plans, freeze/submit/resume/report | Configuration is split into sampling, Slurm and event/pair helpers. `_completed_summary()` validates final summary fields separately from report rendering. Only `run()` enters JAX after GPU/frozen-source guards. |
| [../sample_grb_sequential.py](../sample_grb_sequential.py) | Protected likelihood/prior/nested engine | Byte-preserved research engine. Do not reformat or change its inherited defaults casually; call it through frozen CLI arguments. |
| [../sequential_sky_prior.py](../sequential_sky_prior.py), [../grb_localization.py](../grb_localization.py) | Numerical map density/sampling and coordinate handling | Continuous within-pixel sampling, solid-angle normalization and native-coordinate interpretation. These scientific sources remain unchanged. |

## State and provenance vocabulary

- **Source checkout:** human-editable maintained code. It is not the launcher
  for an already prepared search or BayesWave run.
- **Frozen bundle:** copied code, scientific inputs, configuration and a
  file-hash manifest. Use its own entrypoint at its original location.
- **Claim:** an exclusive record preventing duplicate acquisition/submission.
  A claim is not evidence that a job was accepted or completed.
- **Receipt:** the scheduler response to a submission. Reconcile ambiguous
  receipts before any retry; do not remove claims blindly.
- **Checkpoint:** completed-batch state, random key and accumulated dead
  history. Its existence alone says nothing about current queue state.
- **Summary:** finished-run scalar bookkeeping or a human-readable report.
  The report is not an independent checkpoint/weight integrity audit.
- **READY / BLOCKED:** probability-package acceptance / explicit release
  failure, not a claim about source association or scientific convergence.

Exit 0 means the command completed its intended software action. Acquisition
exit 2 can mean an incomplete collection; runtime exit 75 means deliberately
checkpointed and unfinished. A Slurm TIMEOUT is unfinished regardless of its
exit-code field. Check both accounting and artifacts, not just filenames.

## Safe readability changes

Prefer one operation per line, meaningful intermediate predicates, small
validation helpers, typed public boundaries and docstrings that state side
effects. Keep mathematical formulas and unit conversions explicit. Avoid
mixing a readability refactor with changes to priors, stopping, release
selection or tolerances. The scientific source hash is an intentional boundary.

The current refactor splits CLI groups/dispatch, search configuration and
summary validation, and expands shared metadata/data-window checks. Some
PSD/map orchestration still has long provenance dictionaries and can be
improved separately; this is not a claim that every historical script is now
idiomatic or runnable. Empty historical helper files are preserved user
state, not an instruction to regenerate them.

## Test before preparing a new bundle

```bash
python -m unittest discover -s pipeline/tests -v
python pipeline/run.py doctor
python pipeline/run.py data gw170817 --help
```

The tests use synthetic files and mocked scheduling/downloads; no JAX
likelihood or network transfer is needed. The real calibration acquisition
has additionally been checked against the public release and every reviewed
historical sample. Neither exercise validates a new scientific model.