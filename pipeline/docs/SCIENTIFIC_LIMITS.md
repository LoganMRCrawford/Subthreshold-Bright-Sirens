# What remains scientifically unresolved

Working software, reproducible inputs and stable inference are different
requirements. Acquisition and readability fixes do not alter the inference
target or make a candidate astrophysical. The points below explain the
scientific limits flagged in the repository review.

## 1. Convergence: did sampling explore the fixed target reliably?

For fixed data, likelihood and normalized priors there is one evidence,

$$ Z = \int \mathcal{L}(d\mid\theta)\,\pi(\theta)\,d\theta. $$

Nested sampling estimates it from a finite live population and constrained
replacement draws. More live points can help preserve small modes, while
longer inner MCMC can reduce correlations. Neither guarantees independent
or complete exploration. Missing modes and correlated constrained draws can
produce errors not represented by the run's internal uncertainty estimate.

Earlier 500-live candidate repeats changed both evidence and posterior
structure. For GRB240715239, the two predeclared map-minus-whole comparisons
were approximately −0.47 and +1.52: even the sign changed. This is not a
stable association preference. GW170817's tighter stopping continuation
changed little but did not remove the earlier cross-seed discrepancies.

The eight 5000-live searches are current-model continuation tests, not
completed new evidence results. Their first twelve-hour allocations did not
finish. Completion of a continuation would be necessary before comparing
all declared seeds, absolute evidences and mass/time/sky/distance structure;
even two seeds per prior give only a limited measure of repeatability.

The stopping gap below −3 controls an **estimated remaining-evidence**
criterion within a sampled trajectory. It does not show that an unexplored
mode is absent. Similar corner plots can coexist with different evidence
normalizations. Successful unit tests and checkpoint loading are not tests
of constrained-draw mixing or complete scientific exploration.

## 2. BNS prior: which physical population is the model describing?

The inherited detector-frame chirp-mass range is 1–2 solar masses, but the
component support is 0.5–7.7 solar masses with mass ratio 0.125–1. Using an
NS-tidal waveform does not, by itself, restrict both objects to a
conventional neutron-star population. Unequal-mass solutions can receive
weight in regions that a stricter BNS hypothesis would exclude. Earlier
candidate posteriors placed substantial weight near very low secondary
masses, so this is not just unused prior volume.

A scientifically motivated BNS prior requires an explicit minimum mass,
maximum mass/EOS treatment, spin assumptions and a decision about source-
versus detector-frame masses. The relation is

$$ m_{\rm detector}=(1+z)\,m_{\rm source}. $$

There is no universally justified cutoff that can be silently inserted as
a numerical repair. A stricter prior changes both the allowed likelihood
structure and the evidence normalization. Its results describe a **new
model** and need new calibrated searches, not relabelled checkpoints or
simple comparison with old evidences as though only convergence improved.

The current 5000-live campaign deliberately retains the original target so
repeatability can be tested without also changing the physical hypothesis.

## 3. Duration: does the selected segment accommodate the waveform?

The engine analyzes 128 seconds ending at the GRB trigger, with coalescence
time between 10 seconds before the trigger and the trigger itself. Thus it
contains only about **118–128 seconds before coalescence**, before allowing
for taper and detector-arrival margins.

A leading-order inspiral duration estimate from starting frequency $f$ is

$$ t(f)\simeq\frac{5}{256}
\left(\frac{G\mathcal{M}_{\rm detector}}{c^3}\right)^{-5/3}
(\pi f)^{-8/3}. $$

At 23 Hz this is approximately **151 s for chirp mass 1 solar mass**,
**112 s at 1.198 solar masses**, and **191 s at 0.871 solar masses**.
The low-mass end can therefore require more pre-merger time than the chosen
segment supplies. Allowing still lower chirp mass without a duration review
would make the mismatch worse.

This estimate is a screening warning, **not proof** of an existing
wrap-around/likelihood bug or the cause of the candidate modes. The next
validation should examine the actual waveform, time shifts, finite FFT
representation, taper and injected recovery. If a longer on-source segment
is adopted, the PSD context must also move to retain the gap; the FFT grid,
raw PSD conversion and full analysis contract need explicit review. Such
changes do not belong inside a resume of the current frozen target.

## 4. Fifty maps: what constitutes a usable analysis sample?

The example catalogue contains three events. A catalogue entry, error
radius, contour plot or successful URL request is not a usable released
probability distribution. Each additional event needs a real differential
map with valid normalization, supported coordinate conventions, correct
event/trigger binding and retained provenance. The search also needs
appropriate finite strain, a reviewed detector network and consistent PSDs.

The acquisition pipeline reports missing/invalid products as BLOCKED rather
than manufacturing densities or silently choosing a numerically convenient
older release. The target of at least 50 usable distributions, including
GW170817 and the prime candidates, therefore remains an acquisition and
selection task. A catalogue-wide command does not establish that it has
been achieved. Availability and selection effects must also be documented.

## 5. Noise and association: what would a positive result mean?

Welch and a BayesLine posterior median are alternative **fixed PSD inputs**
in the current likelihood. Neither propagates the full PSD posterior or
jointly fits transient glitches and the CBC signal. A flexible PSD model
does not automatically have better evidence calibration or detection
performance. Earlier comparisons also revealed release/conditioning
differences; those cannot be attributed solely to the PSD estimator.

Matched input checks and GW170817 calibration are required for new routes,
but demonstrating performance requires controlled injections, appropriate
held-out/background comparisons and independent repeats. A larger
map-prior boost alone is not evidence that one PSD is more accurate.

Finally, using the released GRB sky distribution **once as a sequential
prior** yields conditional GW evidence. It does not supply search trials,
the astrophysical association prior, background coincidence rates or a
complete detection/association probability. The uniform 0–10 s delay and
flat 20–800 Mpc distance prior remain modelling assumptions.

## Sensible order of work

1. Finish and audit the authorized unchanged-target convergence experiment.
2. Validate finite-segment/waveform behavior and define the intended BNS
   population; document any revised target separately.
3. Run a separately authorized GW170817 calibration and independent repeats
   for that target with consistent PSD routes and reviewed inputs.
4. Expand and audit released-map availability before a large catalogue
   campaign, then address background/selection and injection performance.

This list describes follow-up requirements, not jobs submitted by this
documentation change. See [GW170817.md](GW170817.md) for the now-reproducible
calibration input route and [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md) for
the distinct software/usability issues.