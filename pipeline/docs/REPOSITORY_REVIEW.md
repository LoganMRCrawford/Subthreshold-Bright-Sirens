# Repository consistency and human-use review

Reviewed 10 November 2026. This is a software/documentation review, not a
certificate of scientific convergence, PSD superiority or installation on
every machine. It concerns this repository, not unrelated workspace projects.

## What is maintained

- Start with [../README.md](../README.md) and [../run.py](../run.py).
- Current inference uses released GRB probability maps once as sequential
  sky priors. Historical Gaussian-prior catalogue results are a different
  analysis and must remain labelled as such.
- Legacy SFM/LC trees and generated work are local-only. Archive references
  may therefore be absent in a fresh cloud checkout. Their old Git history
  has not been rewritten.
- Scientific engines, frozen campaign inputs/source, completed historical
  results and running-search settings were not changed by this review.

## Inconsistencies corrected

1. Archive pages and inventory described only the physical-archive layout
   even after original paths were restored for continuation. They now
   describe both supported layouts and point to a read-only status command.
2. The pipeline overview implied the local archives ship with a checkout.
   It now distinguishes maintained code from optional local historical data.
3. Smoke-test notes described only the pre-consolidation run. They now also
    record the successful unified-workflow jobs 402260/402262 and their limits.
4. The summer report did not clearly distinguish retrospective Gaussian-prior
   results from current released-map work. A scope notice was added without
   changing its historical numbers or narrative.

## Follow-up fixes: calibration access and readability

- **The archive-dependent calibration onboarding gap is fixed.**
  `data gw170817` now acquires the checksum-pinned reviewed public v2
  cleaned H1/L1/V1 files, checks the exact extracted sample hashes and
  publishes a new cache only after full-network validation. The real public
  route passed and every sample/epoch matched the historical cache. See
  [GW170817.md](GW170817.md) for normal/offline commands and failure handling.
  This reproduces reviewed inputs; it is not new cleaning or scientific
  validation, and no existing frozen input is replaced.
- **The maintained navigation/validation bottleneck is improved.** CLI
  construction and lazy dispatch are separated into named functions;
  search configuration and final-summary validation have focused helpers,
  typed interfaces and invariant docstrings. Shared network and strain-window
  checks use readable blocks. See [CODE_GUIDE.md](CODE_GUIDE.md). Scientific
  engine bytes, historical files and active/frozen runs are not refactored.
- [SCIENTIFIC_LIMITS.md](SCIENTIFIC_LIMITS.md) explains why convergence,
  physical priors, waveform duration and catalogue availability remain
  distinct from these software fixes.

Follow-up validation passed **54 offline tests**, plus before/after parity
checks for 21 configuration cases, existing task arguments, rendered Slurm
and eight existing CLI argument namespaces. A real, new two-task calibration
bundle was prepared and verified from the acquired data without submission
or likelihood evaluation. Actual GPU inference after this refactor has not
been claimed; existing jobs continue using their unchanged frozen sources.

## Remaining difficulties and risks

| Priority | Issue | Practical implication / recommended next action |
|---|---|---|
| High | Fresh setup is not a one-command reproduction. | The pinned NSS BlackJAX fork conflicts with JimGW's version metadata. Use the documented two-stage install; the sole known local `pip check` mismatch is intentional. A clean-environment installation has not been certified. Do not upgrade environments used by resumable jobs. |
| High | Numerical/model validity remains unsettled. | Higher-live convergence is unfinished. The inherited mass prior is not a revised strict-BNS prior; low-mass duration versus the 128 s / 23 Hz segment needs validation. These are scientific questions, not documentation fixes. The 50-map availability target is unmet, and BayesLine superiority is not established. |
| Medium | Frozen bundles are location-bound. | Absolute launcher paths and input-path equality are part of existing validation. Do not move a prepared search/BayesWave bundle or change active legacy paths. A content-identity/relocation adapter is future work, not an intrinsic BlackJAX requirement. |
| Medium | Batch stopping is not immediate. | A GPU batch can exceed the ten-minute USR1 lead. Per-batch atomic checkpoints protect earlier progress, but the active batch can be lost at timeout. A signal exit 75 is incomplete, not evidence; a Slurm TIMEOUT is not success even with exit 0:0. |
| Medium | BayesWave scheduling and environments remain manual. | Prepare/run require a separate LAL/BayesWave environment and an explicit compute allocation; preparation itself submits nothing. Account/partition settings are site-specific. BayesWave resume is not implemented, and default chain lengths are pilots rather than proven PSD convergence. |
| Medium | Summary output is not a full integrity or scheduler audit. | The summary checks required artifacts, metadata and finite evidence fields, but does not independently decode checkpoints or validate all posterior weights. A checkpoint file alone cannot distinguish running, timed-out and stopped jobs. Use scheduler accounting and a dedicated acceptance audit for stronger claims. |
| Medium | Configuration requires careful path/route choices. | Input paths are relative to the configuration file, not the shell. New BayesWave products and registered legacy PSDs use different output directories and estimators. Edit examples explicitly; do not mix routes between candidates and GW170817. Existing output directories are intentionally refused. |
| Medium | Historical material is easy to mistake for current guidance. | Root historical notes, the summer report, fixed-PSD audit fragment and archived scripts preserve earlier interpretations. Use the maintained guide for operations and do not pool Gaussian/map evidences or treat old PSD rankings as present conclusions. |
| Low | Remaining research-code complexity. | The maintained CLI/configuration/shared validation paths have been refactored and documented. Some PSD/map provenance construction and the byte-protected numerical engine remain dense; further cleanup must preserve tests and scientific identities. Historical scripts are not promoted to current entrypoints. |
| Low | No maintained candidate-plot command. | Plotting tools exist in local historical work, but the public CLI currently covers acquisition, PSDs, searches and summaries, not publication-ready corners or an interactive dashboard. |

## Operational reminders

- Publication preparation now excludes machine-specific layout receipts and
  full local inventories. A source-only layout status reports `not_installed`
  without querying Slurm or creating directories. See
  [PUBLICATION.md](PUBLICATION.md) for the curated set and remaining decisions.
- A host-only CI recipe is configured, not yet claimed to have run on GitHub.
  Licensing, a clean dependency installation and any public-history cleanup
  still require separate review/approval; no history is rewritten by staging.
- [../../requirements.txt](../../requirements.txt) is the older environment
  dump, not the maintained installation recipe. Use
  [../requirements-host.txt](../requirements-host.txt) or
  [../requirements-search.txt](../requirements-search.txt) plus the separate
  NSS-fork step.
- Fifteen empty historical Python helpers were found. They are preserved
  user state, not missing code to silently regenerate. Archived frozen
  snapshots can contain working versions; new work should use the maintained
  CLI instead of guessing which historical script is runnable.
- The local-only policy staged 444 index deletions while retaining all files
  on disk. Review that intended Git change before a later commit; this audit
  performs no staging, commit, push or history rewrite.
- The layout utility's `status` action is read-only. Applied archive/restore
  actions require an empty user queue and are not part of this review.
- Short smoke success validates execution/interfaces only. It does not
  promote the new noise-only PSD estimator to scientific production or test
  GPU interruption/resume.

## Review coverage and limits

The completed scan parsed **89 maintained/legacy Python files** with no
syntax errors, syntax-checked **38 legacy Slurm scripts**, checked **74 local
documentation links** and parsed **17 documented CLI examples**. All **40
maintained offline tests** passed, including ten new layout-safety tests.
The installed dependency check reported only the known JimGW/NSS-fork
metadata conflict described above.

Preservation checks passed for **527 protected historical file hashes**, all
**295 legacy inventory hashes**, the **16 saved pre-resume checkpoint/metadata
copies**, current run metadata, and the original frozen campaign's complete
parent/legacy verification chain. Maintained scientific/workflow implementation,
Git index and HEAD remained unchanged during this review; only the separate
archive-layout inspection utility, tests and documentation were edited.
Authorized sampling is allowed to advance its live checkpoints; that is not
treated as an audit-induced change. Results are recorded in local review
artifacts without JAX likelihood execution, downloads or new allocations.

The three historical trees were inventoried without importing their scripts;
their protected hashes and frozen manifests are checked separately. Bulk
posterior tables, binary data and every saved campaign log were not
individually reinterpreted, and historical source was not rewritten to make
all old instructions mutually current. External release URLs and a fresh
installation were not network-tested in this scan.

The subsequent onboarding fix **did** exercise the pinned public calibration
downloads and exact sample identity, unlike the earlier offline scan. It
does not certify a fresh package installation, a new PSD fit, or a full GPU
inference route after these source changes. New source changes apply only
to new bundles; prior frozen jobs retain their recorded implementation.

See [SMOKE_TEST.md](SMOKE_TEST.md), [BAYESWAVE.md](BAYESWAVE.md) and
[../../archive/README.md](../../archive/README.md) for the corresponding
execution, noise-model and preservation boundaries.