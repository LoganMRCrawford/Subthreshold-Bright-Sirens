# BayesWave/BayesLine PSDs for the unified pipeline

Run paths below from the repository root. BayesWave is compiled **CPU**
software with LAL frame dependencies; keep it separate from JAX/CUDA.

## Install

```bash
conda env create -f pipeline/environment-bayeswave.yml
conda run -n subthreshold-bayeswave BayesWave --help
conda run -n subthreshold-bayeswave BayesWavePost --help
conda list -n subthreshold-bayeswave --explicit > bayeswave-explicit-spec.txt
```

The recipe requests conda-forge BayesWave **1.1.3** and Python 3.12. The
previous local tested Linux build was `1.1.3-h8e39d12_0`; the YAML is not a
cross-platform binary lock. Archive the explicit spec for your actual
environment. If the platform lacks a package, use the official BayesWave
source-build instructions, not an arbitrary downloaded binary.

Find the actual prefix with `conda env list` and pass it to `--prefix`.
Existing environments are not modified by the workflow commands.

## Prepare exact inputs

First create the event cache using `pipeline/run.py data import` or `data fetch`.
The fixed-model adapter requires 4096 Hz finite strain, a 128 s GW segment,
1024 s earlier PSD context and 16 s gap. It writes the off-source context
to exact-sample GWF frames, checks frame and LAL-cache read-back equality,
and records input/binary hashes and reference Fourier data.

```bash
conda run -n subthreshold-bayeswave python pipeline/run.py psd prepare \
  --data pipeline/work/strain/GRB190728271 \
  --prefix /path/to/conda/envs/subthreshold-bayeswave \
  --iterations 5000 --burnin 1000 \
  --output pipeline/work/bayeswave/GRB190728271
```

This command submits nothing and performs no sampling or additional cleaning.
The bundle includes frozen CLI sources. Exact sample epochs are retained;
no interpolation/padding is used to conceal fractional-GPS offsets.
Preparation must use the Python inside `--prefix`; Python/package versions
are recorded and checked again at execution. Keep the bundle at its prepared
absolute path and preserve that environment for its later execution.

### Which data are fitted?

The new **noiseOnly/noClean** route uses the **final 128 seconds** of the
off-source context as its fit segment, with **1024 seconds for initialization**.
It does not fit every second of the 1024 s context. The 16 s gap remains
outside both intervals. This is **not the same estimator** as historical
cleanOnly runs producing archived glitch PSDs; do not silently replace those
PSDs in a current-model comparison.

### Keep the full fitting grid

For BayesWave 1.1.3 the tested startup path needs the full Nyquist **2048 Hz**
fitting cutoff with 4096 Hz/128 s input (524288 samples). A 1600 Hz cutoff
does not yield the required radix-2 startup; a 1024 Hz workaround can
truncate/reinterpret the input cadence. The adapter rejects those grids.

Fourier checks use 23–1600 Hz, but the **fit still extends to 2048 Hz**.
This does not remove upper-band influence or certify the 4 kHz public
release's high-frequency conditioning.

## Fit and postprocess on a compute node

Use the site's supported CPU allocation if available. A historical Tursa
GPU-node workaround is not a general requirement. Do not run fits on shared
login nodes. Inside a compute allocation:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n subthreshold-bayeswave python \
  pipeline/work/bayeswave/GRB190728271/source/run.py psd run \
  --bundle pipeline/work/bayeswave/GRB190728271
```

On your own workstation add `--local` explicitly. Run the **frozen entry
point**, not the mutable checkout. The adapter executes:

- **BayesWave** with `--bayesLine --noiseOnly --noClean --noWaveletPrior`,
  explicit iterations/burn-in, 2 chains and 2 OpenMP threads.
- **BayesWavePost** with `--bayesLine --0noise`, without `--lite`.
- Both stages with identical real frames/channels, trigger/context/fit times,
  `--srate 4096 --seglen 128 --psdlength 1024 --window 1.0 --padding 0.4`
  and detector-specific flow/fhigh of 23/2048 Hz.

Inspect the generated `commands.json` for complete flags. Never replace the
Post inputs with simulated/`interp:` data. Default 5000 iterations / 1000
burn-in is a **pilot**, not demonstrated PSD convergence. No BayesWave
checkpoint/resume is implemented here; that is separate from the CBC
nested sampler's checkpoint mechanism.

Claims prevent duplicate runs; each stage has a durable log and a retained
Fourier export checked against the exact reference with stated window-scale
tolerances. A failed stage produces a failed result record, not valid PSDs.
Inspect failures and make a reviewed new bundle rather than overwriting
products or rerunning Post on different data.

## Products and normalization

Only after both input checks, raw/physical frequency-grid agreement,
positive active-band values and the raw-to-physical unit check do products
appear in `…/products`. The complete detector network is staged privately,
checked against the original FFT grid and then published together. A failed
check may leave a diagnostic `.products.pending` directory, which must not
be passed to inference. Search preparation also requires the successful
bundle result record for newly estimated products.
The published products have the legacy sampler-compatible names:

```
products/
  glitch_median_PSD_H1.dat
  glitch_median_PSD_L1.dat
  psd.json
```

**These names are compatibility aliases only**: the new estimator fitted
noise, not glitch wavelets. The JSON states this explicitly and binds source
strain and metadata hashes. Set the search config's `psd` path to this directory.

The unchanged sampler expects **raw BayesWave power bins**, applying

$$S_n(f)=P(f)\frac{2}{128}$$

exactly once. The adapter checks agreement with `_forLI` physical PSDs.
Never supply `_forLI`, already converted physical PSDs, or ASDs to the raw
loader. Do not double-convert physical products used by another API.
Positive numbers cannot establish which units a file uses.

## Import historical PSDs instead

To reproduce the current fixed-PSD model, register its existing raw archive:

```bash
python pipeline/run.py psd import \
  --data pipeline/work/strain/GRB190728271 \
  --source /path/to/original/run/trigtime_EVENT/post/clean \
  --convention raw-2-over-128 \
  --output pipeline/work/psd/GRB190728271
```

The source must contain `glitch_median_PSD_IFO.dat` for each selected detector.
The import records your **explicit unit assertion** and checks hashes/grid
coverage, but cannot infer historical units, matching strain release,
estimator convergence or scientific cleanliness from filenames. Inspect the
archived configuration. The workspace's archives/`assist_bayeswave` copies
remain untouched outside this repository.
The imported record binds the chosen current strain bytes, so later
substitution is detected; that association is not independent proof of which
release the historical fit originally used.

## Validation boundary

Fixed median PSDs do not propagate PSD uncertainty or perform joint
astrophysical-versus-glitch inference. A completed estimate does not show
BayesLine outperforms Welch. Keep training/held-out roles distinct, validate
GW170817 whole/map recovery and repeatability, and use matched background
and injection studies for detection-performance claims.

The new noise-only interface is not automatically promoted to the active
campaign. Keep scientific changes in separately named bundles.