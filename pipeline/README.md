# Unified Subthreshold pipeline

**This is the single maintained workflow.** Use [run.py](run.py) for new work.
The three older SFM/LC trees are [local-only historical archives](../archive/README.md),
not competing production entry points or files shipped with a fresh cloud
checkout. Their original root paths are temporarily restored for legacy
checkpoint continuation, with archive references pointing to them. Use the
read-only layout-status command in the archive guide to inspect either
supported layout. Do not move queued/running jobs' paths or a prepared new
workflow bundle. This is a launch/provenance compatibility restriction,
not an intrinsic property of the saved numerical state.

Run the examples from the **repository root**. Use explicit input paths and
fresh output directories; new work belongs under `pipeline/work/`, not inside
an archived result tree.

## Layout

| Location | Purpose |
|---|---|
| [run.py](run.py) | One public command interface. |
| [lc_pipeline/](lc_pipeline/) | Maintained acquisition, data, PSD and search implementation; the internal package name retains its LC origin. |
| [examples/](examples/) | Catalogue and relative-path search configuration. |
| [docs/BAYESWAVE.md](docs/BAYESWAVE.md) | BayesWave installation, fitting, postprocessing and PSD units. |
| [docs/GW170817.md](docs/GW170817.md) | Archive-independent acquisition of the pinned, reviewed calibration strain release. |
| [docs/SMOKE_TEST.md](docs/SMOKE_TEST.md) | Recorded short-run test and its limitations. |
| [docs/REPOSITORY_REVIEW.md](docs/REPOSITORY_REVIEW.md) | Consistency review, onboarding difficulties and unresolved software/scientific boundaries. |
| [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) | Module/function navigation and safe extension/testing conventions. |
| [docs/SCIENTIFIC_LIMITS.md](docs/SCIENTIFIC_LIMITS.md) | Detailed explanation of convergence, physical priors, finite segments and catalogue availability. |
| [docs/PUBLICATION.md](docs/PUBLICATION.md) | Curated release contents, local exclusions and unresolved licence/history decisions. |
| [tests/](tests/) | Offline numerical/interface/safety tests. |
| [sample_grb_sequential.py](sample_grb_sequential.py) and sky helpers | Byte-identical copies of the audited science engine. Use the public CLI, not this engine's historical defaults. |

## 1. Install

Use a **new Python 3.11 environment**. Never upgrade an environment needed by
an existing run or checkpoint. From the repository root:

```bash
python3.11 -m venv .venv-pipeline
source .venv-pipeline/bin/activate
python -m pip install -r pipeline/requirements-search.txt
python -m pip install --no-deps "blackjax @ git+https://github.com/handley-lab/blackjax.git@2180e29ffb645b2c46b76c768229c7c24212446c"
python pipeline/run.py doctor
python pipeline/run.py --help
```

The **second install is required**: JimGW metadata requires `blackjax>=1.6`,
but the NSS fork needed for `stepper_fn` reports `0.1.0b1.dev194+g2180e29ff`.
Resolve the base dependencies first, then install the exact fork with
`--no-deps`. `pip check` reports that known version mismatch; do not repair
it by replacing the NSS fork. This is an intentional research-stack override,
not a resolver-clean package set. A fresh install has not been executed as
part of this consolidation. CUDA12-compatible NVIDIA drivers are needed for
the search; follow JAX's driver requirements.

For host acquisition/tests only, install [requirements-host.txt](requirements-host.txt)
instead. Host commands never evaluate the GW likelihood or import JAX.

```bash
python -m unittest discover -s pipeline/tests -v
```

Install BayesWave in its **separate** CPU/LAL environment; see step 4.

## 2. Acquire real released sky distributions

The supplied [event catalogue](examples/events.csv) includes GW170817 and the
two prime candidates. Larger CSVs use `GRB,GPS,LIGO_run` (or `event,gamma_trigger_gps`).
This command acquires maps; it does not select a new GRB catalogue or validate
GW data quality.

```bash
python pipeline/run.py sky acquire \
  --catalogue pipeline/examples/events.csv \
  --events GRB170817529 GRB190728271 GRB240715239 \
  --output pipeline/work/sky
```

Use `--all` explicitly for every entry in a supplied catalogue. GW170817 is
always included. There is no automatic search or guarantee of 50 usable maps.

- Latest released differential HEALPix file in the mission current directory;
  the pinned targeted-search systematic product is used for GW170817.
- Original FITS, source URL/hash, frame/order, event/time checks and a prepared
  NPZ/JSON package are retained.
- No PMAP/contour/`SIGNIFICANCE` conversion, clipping of negative probabilities
  or fallback to an older version to bypass failed numeric checks.
- An invalid/missing release is **BLOCKED** in the collection JSON/CSV; exit2
  means the collection is incomplete, not that every source failed.

Routine and targeted-search localizations differ; acceptance is not a claim
of equal mission-systematic calibration. The assumed independent uniform
merger-to-GRB delay is 0–10 s, not a measured timing posterior.

For pinned URLs/local caches use `--sources sources.json` mapping each event
to `path` or HTTPS `url`, optional `sha256`, `method` and `event_binding`.
Local paths resolve relative to the source JSON. `--offline` requires a local
entry for every event including GW170817. Its exact source/hash/binding are
listed in [the acquisition implementation](lc_pipeline/sky.py), because its
FITS header does not independently identify the event.

## 3. Import or fetch strain

The current engine uses finite **4096 Hz** data: **128 s** on-source ending
at the trigger, **16 s** gap and **1024 s** earlier PSD context.

For GW170817, acquire the **pinned reviewed v2 cleaned release** without
requiring any local SFM/LC archive:

```bash
python pipeline/run.py data gw170817 \
  --output pipeline/work/strain/GRB170817529
```

This downloads about 185 MB, checks original-file and extracted-sample
hashes for H1/L1/V1, checks the sample grid and publishes the event cache
only after all detectors pass. An offline `--source-dir` accepts those same
full public files. See [the calibration guide](docs/GW170817.md) for exact
release identity, fresh-directory rules and failure diagnostics.

Alternatively import an existing reviewed cache, here through an optional
local SFM archive reference:

```bash
python pipeline/run.py data import \
  --source archive/SFM_baseline_GW170817_GRB_extended/data/GRB170817529 \
  --output pipeline/work/strain/GRB170817529
```

That source is optional legacy data, not a runtime dependency: any reviewed
event cache with the same HDF5/primitive-metadata contract may be supplied.
Large strain files are not included in Git. For fresh modern public data:

```bash
python pipeline/run.py data fetch \
  --event GRB190728271 --gps 1248330654.8524225 --detectors H1 L1 \
  --output pipeline/work/strain/GRB190728271
```

Choose the network explicitly; GRB240715239 currently uses L1/V1. Review
release/version, science segments, calibration, hardware injections, gaps
and filtering before science use. Default run-data fetching is refused for
GW170817: use `data gw170817` or import a reviewed cache instead of silently
substituting another public release. No extra L1 glitch subtraction is
applied here. The pinned sample values match the audited historical cache;
new file serialization is not a replacement for existing frozen inputs.

## 4. Obtain PSD estimates

```bash
conda env create -f pipeline/environment-bayeswave.yml
conda run -n subthreshold-bayeswave BayesWave --help
conda run -n subthreshold-bayeswave BayesWavePost --help
```

[The BayesWave guide](docs/BAYESWAVE.md) covers exact-input frame preparation,
CPU compute-node fitting, postprocessing, input FFT checks and physical/raw
unit validation. It also covers import of archived estimates. The new
noiseOnly/noClean estimator is not identical to historical cleanOnly PSDs.
Use one estimator route for candidates and GW170817 in each search bundle.

The unchanged sampler expects **raw power**, converting by **2/128 exactly
once**. Never pass physical `_forLI` PSDs or ASDs to that loader.

## 5. Configure and freeze whole-sky/GRB-map runs

Edit [examples/search.json](examples/search.json). Input paths are relative
to the **config file**, not the shell. Its account/partition are Tursa examples.
Include GW170817; specify distinct whole/map seeds, data, complete sky package
(FITS/NPZ/JSON) and PSD directories. New BayesWave uses its `products` directory;
registered archived PSDs use the directory produced by `psd import`.

For Welch choose `psd_source: "welch"` and omit `psd`, in a separate matched
bundle. The example 500live/192MCMC/150delete is diagnostic, not established
convergence. At 5000 live points, 1500 deletions retains the 30% fraction.
Set Slurm account, partition, QOS, hours and concurrency explicitly.

```bash
python pipeline/run.py search prepare \
  --config pipeline/examples/search.json --output pipeline/work/search/test01
python pipeline/work/search/test01/source/run.py search verify \
  --bundle pipeline/work/search/test01
```

This copies scientific inputs, original FITS, code, versions and hashes;
preparation submits nothing. Use bundle paths without whitespace/quotes/#.
After preparation, use the **bundle's frozen source/run.py** for everything.
Bundles must stay at the prepared absolute location; paths and environment
identity are part of resume compatibility.

## 6. Preflight, submit and resume

On a GPU compute node (never a shared login node):

```bash
JAX_PLATFORMS=cuda python pipeline/work/search/test01/source/run.py search run \
  --bundle pipeline/work/search/test01 --task 0 --preflight
```

Preflight has a separate output directory. On your own GPU workstation only,
add `--local-gpu` explicitly. Inspect the complete task mapping before submission.

```bash
python pipeline/work/search/test01/source/run.py search submit \
  --bundle pipeline/work/search/test01
```

Submission uses test-only validation and exclusive claims/receipts. Reconcile
ambiguous receipts with Slurm; never blindly remove a claim and retry.

Atomic checkpoints retain state, PRNG key, dead history, iteration, accumulated
time and source/input identity. The example saves each completed iteration;
USR1 is requested ten minutes before timeout. Active GPU steps are not
interrupted, so a long step can overrun that lead. Exit75 means incomplete.
No automatic requeue or extra allocation is enabled.

Once this bundle has no active jobs, resume an unfinished task explicitly:

```bash
python pipeline/work/search/test01/source/run.py search submit \
  --bundle pipeline/work/search/test01 --resume-task 0
```

Do not change source, environment, seed, sampler settings or scientific inputs.
Finished/partially finalized CSVs are refused. Old campaigns must use their
own historical frozen launchers at their original paths.

## 7. Summarize

```bash
python pipeline/work/search/test01/source/run.py search summarize \
  --bundle pipeline/work/search/test01 --output pipeline/work/reports/test01
```

Every task and completed pair is reported. Use CSV posterior **weights**,
not equal-weight rows. Internal uncertainties omit cross-seed exploration error.
Incomplete checkpoints are not final evidences.

## Scientific boundary

The released sky distribution is used **once as a sequential prior**, not
an extra GRB likelihood or catalogue-centred Gaussian. This produces
conditional GW evidence, not calibrated detection/association odds.

The science engine's bytes and existing results are unchanged. Its inherited
Mc1–2/component0.5–7.7 prior is **not a revised strict-BNS prior**; distance
remains20–800Mpc. BNS-only prior revision and the 128s/23Hz low-mass duration
concern remain separate scientific work. Fixed PSD medians do not marginalize
PSD/glitch uncertainty or establish BayesLine superiority over Welch.

See [archive policy](../archive/README.md) for historical attribution,
recovery and how to restore original paths before using a legacy frozen run.
See [the scientific explanation](docs/SCIENTIFIC_LIMITS.md) for why these
limits are not fixed by input reproducibility or software tests, and
[the code guide](docs/CODE_GUIDE.md) for reading and extending the maintained
implementation without changing frozen campaigns.