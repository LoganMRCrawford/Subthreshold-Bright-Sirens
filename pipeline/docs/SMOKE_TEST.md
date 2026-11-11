# Recorded execution test and relocation boundary

## Unified-workflow test after consolidation

A fresh real GW170817 H1/L1/V1 test completed on 10 November 2026 using
the unified frozen entrypoint, newly estimated PSDs and inputs below the
maintained pipeline's local work directory.

| Stage | Job | Elapsed | Result |
|---|---:|---:|---|
| BayesWave fit + Post + checks + search preparation | 402260 | 24m09s | COMPLETED, exit 0:0 |
| Whole-sky GW run | 402262_0 | 16m18s | COMPLETED, exit 0:0 |
| GRB-map GW run | 402262_1 | 17m03s | COMPLETED, exit 0:0 |

BayesWave used 1000 iterations, 200 burn-in and two chains/threads. GW
sampling used 50 live points, 24 MCMC steps, 15 deletions and a checkpoint
every batch, with unchanged −3 stopping. The runs completed 34 and 144
batches; recorded sampling times were approximately 79 s and 192 s.
The Slurm elapsed times above also include startup, imports and compilation;
they are not sampling-only performance measurements.

Acceptance checks passed for identical fit/Post Fourier inputs, PSD units
and hashes, use of the new PSDs by both searches, CUDA/x64 execution,
posterior weights, checkpoint decoding and protected historical files.
**This is execution/interface validation, not scientific convergence.**
GPU interruption/resume was not exercised; the short PSD chains and 50-live
evidences are not production estimates. This test is separate from the
unfinished 5000-live candidate continuations.

Detailed human-readable acceptance and machine-readable checks are retained
in the ignored local smoke-test work directory, not shipped as release inputs.
The recorded summary above is portable documentation; a source-only reader
does not need local filesystem receipts to understand its scope.

## Earlier pre-consolidation test

The maintained workflow was exercised on real reviewed GW170817 H1/L1/V1
inputs before consolidation. Its files are preserved byte-for-byte in the
local historical archive. Recorded absolute paths describe the original
layout; restore that layout before legacy verification/resume. This is
execution/interface validation, **not scientific convergence**.

| Corrected stage | Job | Elapsed | Result |
|---|---:|---:|---|
| BayesWave + Post + checks + search preparation | 400969 | 13m45s | COMPLETED0:0 |
| Whole-sky GW run | 400971_0 | 5m41s | COMPLETED0:0 |
| GRB-map GW run | 400971_1 | 5m41s | COMPLETED0:0 |

BayesWave used1000iterations/200burn-in/twochains. GW sampling used50live,
24MCMC,15deletions, checkpoint1 and unchanged−3 stopping. Both GPU runs used
the newly estimated PSDs; raw/physical units, FFT inputs, GPU/x64 provenance,
weights and saved checkpoints passed. GPU interruption/resume was not tested.

The first attempt exposed an uncreated BayesWave output-directory bug. That
attempt is preserved as failed; the fix and regression test are included in
the unified implementation. No numerical/input gates were relaxed.

Consolidation changes the public entry name to `run.py` and generated bundle
paths, not the waveform/prior/evidence implementation. Offline tests and
source/input preparation are rerun in the new folder. The historical GPU jobs
are not claimed to have run from this new path, and no new allocation is
automatically submitted by consolidation.

- Earlier detailed acceptance artifacts remain in the local historical
	archive and are not dependencies of the current source checkout.
- [Archive policy and original path constraints](../../archive/README.md)