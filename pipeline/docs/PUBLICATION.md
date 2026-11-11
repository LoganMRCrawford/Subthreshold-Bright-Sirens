# Curated publication set and remaining decisions

This document describes release preparation, **not approval to commit or
push**, a scientific production release, or permission to rewrite shared
history. The maintained workflow is experimental research software.

## Proposed versioned contents

- Root overview, the retrospective summer report and qualified fixed-PSD
  findings, the existing summary table, illustration and candidate list.
- The complete maintained pipeline: public CLI, acquisition and orchestration
  code, pinned GW170817 release manifest, unchanged scientific engine,
  examples, installation recipes, tests and current documentation.
- Portable archive policy, the retained former root README for attribution,
  and the tested read-only/preview-first local layout utility.
- A host-only CI workflow. It installs host dependencies and runs offline
  tests; it neither runs JAX likelihoods nor downloads scientific data or
  submits Slurm jobs. Dependency installation itself requires network access.

The root dependency dump is retained as historical context, not the current
installation route. Its already-reviewed astropy-healpix addition is included;
the maintained host/search recipes remain the installation instructions.

## Intentionally local, not release inputs

The three historical SFM/LC trees are excluded at both root and archive
locations. Generated pipeline work, downloaded data, PSDs, checkpoints,
local recovery copies, full archive inventories and layout-operation
receipts are ignored. Original forensic records are preserved locally,
not edited in place to remove machine paths. The original legacy removals
are index-only; on-disk files and active jobs remain intact.

In a source-only checkout, `python archive/layout.py status` reports
`not_installed` and creates nothing. Applied archive/restore still requires
the corresponding complete populated layout and an empty user Slurm queue.
No relocation is required to use the maintained data-acquisition commands.

## Decisions that still require explicit approval

### Existing Git history

Ignoring or deleting paths in the next commit does **not** remove old blobs.
The reviewed branch reaches approximately **1.16 GB of uncompressed legacy
file content**, including old strain/posterior artifacts. This is object
content size, not a compressed clone-size estimate. Remote contents were
not changed or assumed absent.

If the requirement is "do not include these trees in future snapshots,"
the curated set implements that. If the requirement is "do not distribute
them anywhere in public history," publication requires a separately approved
clean-history export or coordinated history-removal plan. **Do not force-push,
rewrite history or delete forensic copies automatically.** A ZIP/source-only
export can omit old history, but does not erase copies already on a remote.

### Licence and attribution

No licence has been selected on contributors' behalf. Obtain agreement on
the intended licence and authorship/attribution, then add licence and citation
metadata. Preserved attribution alone is not a licence grant. Public reuse
rights must not be inferred from this preparation step.

### Fresh environment and CI

Offline tests pass in the existing tested environment and are re-run from
an isolated staged-source copy. That does not certify fresh dependency
resolution. The two-stage JimGW/NSS-fork installation remains an intentional
metadata override. The new CI workflow is configured but **not claimed to
have run on GitHub** until an actual workflow result exists. No current
HPC environment is upgraded for release preparation.

## Pre-commit review checklist

- Review additions **and** the 444 intended legacy index removals together;
  do not commit a deletion-only snapshot with no replacement pipeline.
- Confirm the staged source includes the public entrypoint, calibration
  release resource, all tests and the portable archive utility.
- Confirm ignored legacy trees, generated work, inventories and migration
  receipts are absent from the staged tree.
- Run tests and inspect documented examples in the staged source, not only
  in a working tree that contains private local dependencies.
- Review local-path/credential findings. A pattern scan finding no obvious
  token is not a formal guarantee that all historical objects are clean.
- Resolve the history and licence decisions before public publication.
- Commit/push only when explicitly requested. Staging is reversible and
  does not send files to a remote.

See [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md) for software/usability
boundaries and [SCIENTIFIC_LIMITS.md](SCIENTIFIC_LIMITS.md) for interpretation.