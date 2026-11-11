# Historical SFM and LC workflows

**New work uses [the unified pipeline](../pipeline/README.md).** The following
three trees are historical archives, not three supported search entry points.

| Archive reference | Preserved work |
|---|---|
| [SFM baseline](#sfm-baseline) | Sienna Folkes-Miller's original catalogue/strain collection, Gaussian-sky Welch searches and baseline results. |
| [LC noise studies](#lc-noise-studies) | LC's Gaussian-sky noise-model studies, BayesLine/Welch comparisons, experimental alternatives and legacy production results. |
| [LC released-map development](#lc-released-map-development) | LC's released-map development, audits, calibration, repeatability campaigns, frozen bundles and recorded smoke tests. |

### SFM baseline

The local tree is named **SFM_baseline_GW170817_GRB_extended**. It retains
Sienna Folkes-Miller's data collection, original Gaussian-prior sampler and
analysis scripts. These are historical references, not shipped source-only
checkout dependencies. The [former README](README.before-consolidation.md)
retains the original file-by-file method notes.

### LC noise studies

The local tree is named **LC_noise_improved**. It retains the original
Gaussian-prior BayesLine sampler, the two bulk PSD submission scripts and
candidate diagnostics. These local files preserve the implementation behind
the summer report; current commands use the maintained unified pipeline.

### LC released-map development

The local tree is named **LC_grb_informed_pipeline**. It preserves early
released-map implementation, source audits and frozen campaigns. Existing
campaigns use their own frozen launchers, not whichever mutable source
currently happens to be present in that historical tree.

## Layout and local-only Git policy

**The legacy trees are local-only.** Their root and archive locations are
ignored and untracked in the current Git index; they are not shipped with a
fresh cloud checkout. Previously committed copies remain in Git history.

There are two supported filesystem layouts:

- **Physically archived:** the three real directories live here and the old
	root names are absent.
- **Original paths restored:** the real directories live at their original
	root paths; the entries here are reference symlinks, not duplicate copies.

The original paths were temporarily restored for the authorized twelve-hour
continuation array **402487**. This dated record is not a live queue status.
Check the current layout without changing paths or contacting Slurm:

```bash
python archive/layout.py status
```

In a source-only checkout, status reports **not_installed** without creating
folders or querying Slurm. That is normal: local historical data were not
distributed. Applied relocation remains refused for missing or mixed trees.

**Never relocate the trees while any legacy task is queued or running.**
Same-filesystem metadata renames preserve file identities and contents.
The utility explicitly preserves moved-root directory timestamps on
filesystems that change them during rename. No bulk contents are deleted
or duplicated by relocation.

Frozen manifests, saved checkpoints and generated commands contain absolute
**resolved** paths. Those records retain their original values. They are
preserved historical evidence, **not directly relocatable runnable bundles**.
This restriction comes from path-dependent launch/validation code, not from
BlackJAX's numerical state. A validated relocation adapter is future work;
do not bypass the checks by editing historical identities.

Do not run new campaigns or edit code in these archives. Archival status does
not mean every job completed or evidence converged. The first eight
5000-live candidate allocations stopped unfinished; array 402487 continues
those saved trajectories. Working checkpoints may advance during authorized
sampling. Their pre-resume bytes are backed up under ignored local work;
metadata, frozen inputs and completed historical results remain protected.

## Before any legacy verification, resume or older rollback

Use the [layout utility](layout.py) from the repository root. It previews by
default and refuses an applied move unless the user Slurm queue is empty.
Run `status` first; restore **only if physically archived**, not if the paths
are already restored:

```bash
python archive/layout.py restore
python archive/layout.py restore --apply
```

The applied operation restores all three original root paths and creates
archive reference links, so the exact historical launchers can see the same
canonical paths again. Only then review job/checkpoint state and use the
run's **own frozen launcher**; restoration does not submit or resume anything.
Use the preserved analysis environment. Never change identity fields merely
to make a moved checkpoint load.

After all legacy jobs have stopped, the same utility can return the folders
to the physical archive:

```bash
python archive/layout.py archive
python archive/layout.py archive --apply
```

Both directions verify complete inode-level tree identity and save receipts
under `layout_operations`. Failure attempts rollback rather than deleting
contents. Archive/restore round-trip, active-queue refusal and injected-rename
failure rollback were tested in a disposable sandbox. Filesystem validation
alone does not demonstrate successful GPU checkpoint loading or convergence;
check the particular continuation's logs and artifacts separately.

The older pre-cleanup/pre-consolidation rollback tools also assume the original
layout. Restore it first, then preview those tools. Running an old rollback
directly against physical archive directories is intentionally not supported.

## Attribution and former documentation

[README.before-consolidation.md](README.before-consolidation.md) is a byte-for-byte
copy of the former root README, retaining SFM's complete original entries and
LC's historical noise notes. It is a historical document, not the current
install guide. Relative links/code paths inside that snapshot are interpreted
from the original repository layout (its former location), not the current
archive layout.

The active engine was copied from the tested LC released-map route; SFM and
legacy LC algorithms are **not silently merged as if statistically equivalent**.
Their distinct Gaussian/map priors, PSD estimators, settings and results remain
attributed in their original trees.

## Recovery and inventory

The full local inventories record preserved hashes and original-path mapping,
not live layout or sampling status. Both inventory records and layout-operation
receipts are ignored because they contain machine-specific forensic metadata;
they are not required by the maintained workflow or shipped in a source-only
checkout. Use `layout.py status` for the current layout. Frozen records retain
old absolute paths; the local mapping permits read-only hash checks without
changing them. See [the curated publication policy](../pipeline/docs/PUBLICATION.md)
for the distinction between excluded future snapshots and existing Git history.
Protected result hashes and all eight current high-budget checkpoints were
checked before/after relocation.

A separate pre-consolidation source/configuration snapshot is retained in the
workspace recovery area, alongside the earlier611-file pre-cleanup recovery.
The rollback utility defaults to preview and preserves displaced contents.
No Git reset/stash/commit, push or environment upgrade was performed. The
local-only policy intentionally staged index removals of 444 previously
tracked files without deleting their working copies; do not mistake those
staged deletions for data loss or blindly reset the preserved index state.