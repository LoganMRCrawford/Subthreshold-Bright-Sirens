# Summer Work Report: Subthreshold Bright Sirens

**Author:** [Logan Crawford]  
**Project partner:** [Sienna Folkes-Miller]  
**Project period:** Summer 2026  
**Project:** Subthreshold Bright Sirens

> **Scope:** This is a retrospective of the summer's Gaussian-prior work.
> Its historical results and proposed next steps are not the current
> released-map campaign status, a convergence certificate, or evidence that
> BayesLine outperforms Welch. Current operations and unresolved limitations
> are documented in [pipeline/README.md](pipeline/README.md) and
> [pipeline/docs/REPOSITORY_REVIEW.md](pipeline/docs/REPOSITORY_REVIEW.md).

## Abstract

This summer project developed and extended a Bayesian search for subthreshold gravitational-wave (GW) signals associated with short gamma-ray bursts (sGRBs). The search uses the GRB trigger time and localisation to identify possible compact-binary mergers that may be too weak to appear in the standard GW detection catalogues, but for which electromagnetic information can provide a useful prior constraint. The existing pipeline was reproduced and assessed, then extended with improved detector-noise modelling using BayesLine. The principal outcome was a scalable, GPU-based nested-sampling workflow that compares an unconditioned sky prior against a GRB-conditioned sky prior while holding the data segment, waveform, and other inference choices fixed. The original Welch-based pipeline was validated on GRB170817529/GW170817 and used to identify a small set of high-priority candidates. A catalogue-scale BayesLine reanalysis was then prepared and run using posterior-median PSDs derived from BayesWave/BayesLine. The work establishes a controlled basis for determining whether apparent candidate support is robust to the noise model.

## 1. Scientific Motivation

Gravitational-wave detections from compact binaries are generally identified through searches that must contend with a very large parameter space and non-ideal detector noise. A coincident short GRB supplies independent information: a narrow time window and a localisation on the sky. These constraints can increase the relative support for an astrophysical signal associated with the GRB, even when the GW signal is below the normal detection threshold.

The project investigates these possible "subthreshold bright sirens" by comparing two Bayesian signal hypotheses for each GRB:

- an **unconditioned** analysis with a full-sky location prior;
- a **GRB-conditioned** analysis with Gaussian priors centred on the GRB sky position.

The comparison statistic is

$$
\Delta \log Z = \log Z_{\mathrm{conditioned}} - \log Z_{\mathrm{unconditioned}},
$$

where $Z$ is the Bayesian evidence. Positive values favour the analysis in which the GW source is spatially associated with the GRB, while negative values can indicate that structure in the strain data is better accommodated away from the GRB location. This is a model comparison statistic, not a detection claim by itself.

## 2. Starting Point and Project Contributions

The project began from the previous intern's JAX/BlackJAX implementation, which established the GRB-driven analysis idea and a heterodyned-likelihood prototype. The main working pipeline in this project instead uses a full frequency-domain transient likelihood for greater reliability in the broad catalogue analysis.

My work, undertaken alongside my project partner, focused on the following:

1. Understanding, documenting, and validating the existing GRB-triggered nested-sampling workflow.
2. Maintaining a like-for-like comparison between full-sky and GRB-localised priors.
3. Replacing the fixed median-Welch PSD input with BayesLine posterior-median PSDs while preserving the signal-analysis configuration.
4. Building robust Slurm array workflows to process the BayesLine products over the candidate catalogue.
5. Adding diagnostic tooling to inspect strain, time-frequency structure, periodograms, and coalescence-time posteriors for the most interesting events.
6. Defining a realistic next stage in which non-stationary noise and glitches are treated more directly.

The repository records the implementation in `LC_noise_improved/`, while the original data collection, baseline sampling, and catalogue analyses are retained in `SFM_baseline_GW170817_GRB_extended/`.

## 3. Baseline Analysis Pipeline

### 3.1 Candidate selection and data

Candidates were selected from the Fermi GBM Burst Catalogue. The baseline catalogue criteria required GRB times within LIGO observing runs, $T_{90} < 3\,\mathrm{s}$, and localisation uncertainty below $10^\circ$. This produced 84 candidate events including GRB170817529, the GRB associated with GW170817.

For each selected event, publicly available strain data were downloaded from the available LIGO Hanford (H1), LIGO Livingston (L1), and Virgo (V1) instruments. An event-specific metadata file records the GRB GPS time, detector set, likelihood segment, PSD segment, and sky location.

### 3.2 Signal model and inference

The likelihood analysis uses a $128\,\mathrm{s}$ strain segment and the IMRPhenomD_NRTidalv2 binary-neutron-star waveform. The inference band is $23$--$2048\,\mathrm{Hz}$ and phase marginalisation is used. Nested sampling is performed with BlackJAX on GPU hardware.

The sampled parameters include chirp mass, mass ratio, aligned spins, inclination, luminosity distance, coalescence time, polarisation, right ascension, declination, and tidal deformabilities. The physical priors are deliberately broad, including a coalescence-time interval of $[-10,0]$ seconds relative to the GRB trigger, because the GW--GRB delay is not known precisely.

The two analyses differ only in their sky priors. The unconditioned case uses the full sky, while the conditioned case uses Gaussian priors around the GRB coordinates with standard deviations of $0.2$ rad in right ascension and $0.1$ rad in declination. Keeping the remaining choices identical makes the evidence comparison interpretable as the effect of the GRB localisation information under a fixed signal and noise model.

### 3.3 Welch PSD baseline

The original workflow estimated detector noise using a median Welch PSD from off-source data. This is a reasonable and computationally inexpensive baseline, but it treats the PSD as known once estimated. In marginal events, spectral lines, slow non-stationarity, or transient contamination of the PSD-estimation region can affect the likelihood and hence the evidence difference.

## 4. Baseline Results and Candidate Selection

The pipeline was validated using GRB170817529, associated with the established bright siren GW170817. It recovered physically plausible source parameters, including a chirp mass near $1.20\,M_\odot$ and a luminosity distance near $42\,\mathrm{Mpc}$, and produced

$$
\Delta \log Z = 5.233 \pm 0.453.
$$

This serves as an important end-to-end validation of the data handling, prior comparison, nested sampler, and evidence calculation.

For the wider Welch-based catalogue, most events showed no meaningful distinction between the conditioned and unconditioned models: 77 of the 84 initial candidates had $-1 < \Delta\log Z < 1$. Two previously noted candidates showed modest positive support for the GRB-conditioned model:

| Event | $\Delta\log Z$ | Interpretation at baseline |
|---|---:|---|
| GRB190728271 | $1.04 \pm 0.19$ | Mild preference for the GRB-conditioned model; prioritised for follow-up. |
| GRB240715239 | $1.93 \pm 0.33$ | Strongest positive subthreshold candidate in the baseline set; prioritised for follow-up. |
| GRB170817529 / GW170817 | $5.23 \pm 0.45$ | Confirmed reference event and pipeline validation. |

Several events had large absolute evidence values but negative $\Delta\log Z$, including GRB240603102, GRB190810675, GRB170403583, and GRB240824549. Such cases are useful diagnostics, but they cannot be interpreted as signals associated with their target GRBs without examining detector data quality and posterior structure. They motivated the noise-modelling work described below.

## 5. Improved Noise Modelling with BayesLine

### 5.1 Rationale

The central technical aim of the summer work was to assess candidate robustness under a more flexible noise model. BayesLine, run as part of BayesWave/BayesLine, represents the broad-band PSD and spectral lines jointly and returns posterior PSD products. It therefore offers a better-motivated PSD estimate than a fixed Welch estimate where instrumental spectral features or imperfect stationarity matter.

The improved analysis deliberately retains the $128\,\mathrm{s}$ likelihood interval, frequency band, detectors, waveform, and prior comparison used in the Welch baseline. The only intended analysis change is the PSD supplied to the likelihood. This separation is essential: differences in $\Delta\log Z$ should be attributable to the noise model rather than to a change in the astrophysical hypothesis.

### 5.2 BayesLine PSD integration

BayesWave/BayesLine was run on a $1024\,\mathrm{s}$ off-source context ending $16\,\mathrm{s}$ before the start of the likelihood segment. For each available detector, the pipeline loads the raw posterior-median PSD files

```
glitch_median_PSD_<IFO>.dat
```

from the BayesWave post-processing directory. These spectra are filtered for finite, positive values, consolidated at repeated frequencies, checked for coverage of the likelihood band, and interpolated onto the likelihood frequency grid.

The raw BayesWave PSD uses a different normalisation convention from the likelihood. The loader therefore applies

$$
S_n(f) = P(f)\frac{2}{128},
$$

where $P(f)$ is the raw BayesWave PSD product and $128\,\mathrm{s}$ is the likelihood-segment duration. An early implementation issue arose because another BayesWave output, `bayesline_median_psd_physical_<IFO>.dat`, already included this conversion. Loading that product through the raw-product conversion path would normalise the PSD twice. The production workflow now explicitly uses the raw `glitch_median_PSD` products, avoiding that error.

For every detector-event pair, the sampler writes a BayesLine-versus-Welch diagnostic plot. These plots provide a direct check that the PSD has physically sensible values and reveal frequency regions where the two estimates differ appreciably.

### 5.3 Catalogue-scale execution

Two Slurm array submission scripts were created for the two BayesLine PSD uploads:

- `submit_bulk_bayesline_psd.slurm` for `GRB_bayesline_psd_runs_31082026`;
- `submit_bulk_bayesline_psd_01092026.slurm` for `GRB_bayesline_psd_runs_01092026`.

Each task selects one GRB, verifies that exactly one `trigtime_*/post/clean` directory is present, confirms the raw PSD files exist, exports the local data and catalogue paths, and launches the GPU-based sampler. The second upload included a trailing underscore in each event directory name; the script safely removes only this archival suffix before looking up the GRB in the catalogue and writing results.

The BayesLine batches completed successfully. Their products are kept separately in `LC_noise_improved/results/`, preventing accidental replacement of the Welch baseline and allowing event-by-event comparison once evidence values are collated into a dedicated summary table.

## 6. Diagnostics and Interpretation Strategy

Evidence values alone are not sufficient to establish a candidate. I added a diagnostic script for the leading candidates that generates:

- strain time series around the GRB trigger;
- whitened Q-transforms over $20$--$1024\,\mathrm{Hz}$;
- on-source detector periodograms;
- weighted conditioned and unconditioned coalescence-time posterior plots.

The current priority list includes GRB240824549, GRB170403583, GRB240817057, GRB240615744, GRB170121067, GRB170817529, and GRB240715239. The purpose is to identify obvious glitch morphology, line contamination, or posterior support concentrated at the edge of the $t_c$ prior before interpreting an evidence shift physically.

The most useful comparison is not simply whether BayesLine increases a candidate's evidence. A promising candidate should exhibit stable or strengthened conditioned preference, compatible posterior structure, no obvious single-detector artefact, and reasonable behaviour under repeat sampling. Conversely, a large shift between Welch and BayesLine identifies sensitivity to the noise treatment and should trigger closer investigation, not a claim of astrophysical support.

## 7. Challenges and Lessons Learned

The project involved several practical and methodological challenges:

- **Heterodyned likelihood reliability:** the earlier heterodyned prototype was computationally attractive but returned implausibly high evidences for some non-reference GRBs. The transient likelihood was therefore used for the main comparison despite its higher computational cost.
- **PSD convention matching:** correct normalisation and selection of BayesWave products was crucial. This was resolved by tracing the PSD units through the likelihood and using only the intended raw median PSD files.
- **Cluster execution:** initial array tasks could cancel when required data were not yet available. The production scripts now validate the expected PSD structure before launching the expensive sampler.
- **Evidence interpretation:** nested-sampling evidence is a Bayesian quantity. It should be compared with its sampling uncertainty and with posterior diagnostics, rather than treated as a frequentist detection statistic or an isolated ranking number.

## 8. Future Work

### 8.1 BayesLine result collation and controlled repeats

The immediate next step is to extract the BayesLine conditioned and unconditioned evidences into a separate results table and compare them with the Welch baseline for every event. The highest-priority events should be rerun with multiple seeds and, where justified, increased live-point settings to assess nested-sampling stability. GRB170817529 should continue to be used as a reference case for every major pipeline modification.

### 8.2 ALCS noise modelling

An adaptive local covariance or locally adaptive covariance-spectrum (ALCS) approach could provide another PSD model, particularly where no single stationary PSD adequately describes a long off-source interval. Any ALCS prototype should be benchmarked against both Welch and BayesLine using the same data segments, frequency range, detector set, priors, waveform, and sampler settings. Useful validation tests include PSD comparisons, software injections, evidence repeatability, and control-event distributions of $\Delta\log Z$.

### 8.3 Joint signal, noise, and glitch inference

The more ambitious long-term direction is to model the astrophysical signal, detector PSD, and transient glitches simultaneously. BayesWave-style coherent signal components and detector-specific glitch wavelets offer one route. In such a model, uncertainty in both the PSD and glitch content would be propagated into the evidence, rather than being fixed in preprocessing. This is especially relevant for low-SNR candidates, where an unmodelled transient can mimic a change in signal support.

JAXLine is a proposed accelerator-friendly reconstruction of the BayesLine PSD model using JAX/BlackJAX methods, including reversible-jump spline and spectral-line updates. It can provide an independently testable noise component before extending to a JAXWave-style joint model. This work should begin with controlled injections and known events before any new catalogue-scale campaign.

## 9. Conclusion

This summer project advanced a GRB-triggered Bayesian search for subthreshold bright sirens from a validated Welch-PSD baseline to a controlled BayesLine improved-noise workflow. The baseline recovered the known GW170817/GRB170817529 association and identified a small set of events worth further investigation, particularly GRB240715239 and GRB190728271. The BayesLine implementation, diagnostic plots, and Slurm arrays provide the technical foundation to test whether these candidates remain interesting when the detector noise model is made more flexible. The responsible conclusion at this stage is methodological: the pipeline is now equipped to distinguish candidate behaviour that is robust to the PSD model from behaviour that is likely driven by noise assumptions.

## Repository Pointers

### Current maintained workflow

New work uses the unified pipeline. Its released-map sequential sky prior is
distinct from the historical Gaussian-prior analysis described in this report.

- Project overview: [README.md](README.md).
- Installation and end-to-end usage: [pipeline/README.md](pipeline/README.md).
- Public command-line entry point: [pipeline/run.py](pipeline/run.py).
- Released GRB sky-distribution acquisition: [pipeline/lc_pipeline/sky.py](pipeline/lc_pipeline/sky.py).
- BayesWave installation, fitting, postprocessing and PSD units: [pipeline/docs/BAYESWAVE.md](pipeline/docs/BAYESWAVE.md); implementation: [pipeline/lc_pipeline/psd.py](pipeline/lc_pipeline/psd.py).
- Paired whole-sky/GRB-map searches, frozen bundles, submission, resume and summaries: [pipeline/lc_pipeline/search.py](pipeline/lc_pipeline/search.py).
- Example event catalogue and search configuration: [pipeline/examples/events.csv](pipeline/examples/events.csv) and [pipeline/examples/search.json](pipeline/examples/search.json).
- Recorded execution-test scope and offline regression tests: [pipeline/docs/SMOKE_TEST.md](pipeline/docs/SMOKE_TEST.md) and [pipeline/tests/test_workflow.py](pipeline/tests/test_workflow.py).
- Historical Welch baseline evidence table: [grb_results.csv](grb_results.csv).

### Historical summer-work implementation — local archives

The following portable policy/method references describe the original work,
not competing maintained entry points. The historical source trees are
local-only and excluded from future Git snapshots; they are not required to
read this report or use the maintained pipeline. Existing Git history has
not been rewritten. Original root paths may be restored for legacy resume.

- Archive policy, attribution and recovery/layout instructions: [archive/README.md](archive/README.md).
- Original project documentation, including SFM's method notes: [archive/README.before-consolidation.md](archive/README.before-consolidation.md).
- SFM Gaussian-prior baseline and local implementation provenance: [SFM archive description](archive/README.md#sfm-baseline).
- LC Gaussian-prior BayesLine sampler, original batch submissions and candidate diagnostics: [LC noise-study archive description](archive/README.md#lc-noise-studies).
- Curated publication contents and the distinction between local exclusions and old history: [pipeline/docs/PUBLICATION.md](pipeline/docs/PUBLICATION.md).