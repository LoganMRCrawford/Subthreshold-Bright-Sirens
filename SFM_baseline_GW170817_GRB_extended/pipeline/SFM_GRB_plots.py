import os
import numpy as np
import sys
from anesthetic import make_2d_axes
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pickle

from _plot_utils import (
    load_nested_csv,
    map_from_hist,
    compute_hpd_samples,
)

GRB_name = "GRB190525032"
run_corner = "conditioned"

dir_path = (
    f"SFM_baseline_GW170817_GRB_extended/"
    f"data/"
    f"{GRB_name}"
)

###################
##STRAIN PLOTTING##
###################


#reads in metadata and strain data
metadata_filename = f"{GRB_name}_metadata.pkl"
metadata_file = os.path.join(
    dir_path,
    metadata_filename
)

with open(metadata_file, "rb") as f:
    metadata = pickle.load(f)

gps = float(metadata["GPS"])
start = float(metadata["start"])
end = float(metadata["end"])
grb_ra = float(metadata["RA_rad"])
grb_dec = float(metadata["DEC_rad"])

GWOSC_LOCAL_FILES = {}
strains = []
labels = ["LIGO Hanford", "LIGO Livingston", "Virgo"]

for ifo_name in ["H1", "L1", "V1"]:
    strain_filename = f"{GRB_name}_{ifo_name}.hdf5"
    GWOSC_LOCAL_FILES[ifo_name] = os.path.join(dir_path, strain_filename)

    path = GWOSC_LOCAL_FILES[ifo_name]

    strain = TimeSeries.read(
        path,
        format="hdf5"
    ).crop(start, end)

    strains.append(strain)

#plots all three strains
fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 8),
    sharex=True
)

for ax, strain, label in zip(axes, strains, labels):
    ax.plot(
        strain.times.value - gps,
        strain.value,
        linewidth=0.5
    )

    ax.set_ylabel("Strain", fontsize=18)
    ax.set_title(label, fontsize=20)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel(
    "Time relative to GRB trigger [s]",
    fontsize=18
)

plt.savefig(
    os.path.join(dir_path, "strain_data.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()


#################
##LOGZ AND MAPS##
#################

#read in samples and calculate logZ
transient_path = os.path.join(dir_path, f"{GRB_name}_Transient.csv")
transient_samples = load_nested_csv(transient_path)
unconditioned_path = os.path.join(dir_path, f"unconditioned_Heterodyned.csv")
unconditioned_samples = load_nested_csv(unconditioned_path)
conditioned_path = os.path.join(dir_path, f"conditioned_Heterodyned.csv")
conditioned_samples = load_nested_csv(conditioned_path)

logZ_transient = transient_samples.logZ(100).mean()
logZ_transient_err = transient_samples.logZ(100).std()
logZ_unconditioned = unconditioned_samples.logZ(100).mean()
logZ_unconditioned_err = unconditioned_samples.logZ(100).std()
logZ_conditioned = conditioned_samples.logZ(100).mean()
logZ_conditioned_err = conditioned_samples.logZ(100).std()
delta_logZ = logZ_conditioned - logZ_unconditioned
delta_logZ_err = np.sqrt(logZ_unconditioned_err**2 + logZ_conditioned_err**2)
print(f"Delta logZ:    {delta_logZ:.2f} +/- {delta_logZ_err:.2f}")

#print MAPs
PARAM_NAMES = [
    "M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c",
    "psi", "ra", "dec", "lambda_1", "lambda_2",
]

runs = [
    ("Transient", transient_samples),
    ("Unconditioned", unconditioned_samples),
    ("Conditioned", conditioned_samples),
]


for run_name, samples in runs:

    weights = np.asarray(
        samples.get_weights(),
        dtype=float
    ).copy()

    weights /= weights.sum()

    print(f"\n===== {run_name.upper()} MAP ESTIMATES =====")

    for param in PARAM_NAMES:

        values = samples[param].to_numpy()

        bins = np.linspace(
            np.nanmin(values),
            np.nanmax(values),
            200
        )

        map_value = map_from_hist(
            values,
            weights,
            bins
        )

        print(
            f"{param:8s} : {map_value:.6g}"
        )

    print("=========================")



###################
##CORNER PLOTTING##
###################

#take from _plot_utils file
# --------------------------------------------------------------------------- #
# Corner plot helper
# --------------------------------------------------------------------------- #
def make_corner(datasets, params, out_dir, out_name, figsize=(10, 10), lims=None):
    """Create a corner plot from multiple datasets.

    Parameters
    ----------
    datasets : list of (MCMCSamples, label, color)
    params : list of str — column names to plot
    out_name : str — output filename stem
    figsize : tuple
    lims : dict, optional
        Mapping from plotted parameter label to ``(lo, hi)`` axis limits.
    """
    fig, axes = make_2d_axes(params=params, upper=False, figsize=figsize)

    for samples, label, color in datasets:
        samples.plot_2d(
            axes,
            kinds=dict(diagonal='hist_1d', lower='kde_2d'),
            diagonal_kwargs=dict(
                bins=35,
                histtype='step',
                linewidth=2.0,
                density=True,
            ),
            lower_kwargs=dict(levels=[0.99730, 0.95450, 0.68269]),
            color=color, alpha=0.75, label=label,
        )

    if lims:
        for y_param in axes.index:
            for x_param in axes.columns:
                ax = axes.loc[y_param, x_param]
                if ax is None:
                    continue
                is_diagonal = x_param == y_param
                if x_param in lims:
                    ax.set_xlim(*lims[x_param])
                if y_param in lims and not is_diagonal:
                    ax.set_ylim(*lims[y_param])
                if is_diagonal:
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
                    ax.set_ylabel('')

    for ax in fig.axes:
        if ax.get_ylabel():
            continue
        if ax.get_xlabel() not in params:
            continue
        if lims and ax.get_xlabel() in lims:
            ax.set_xlim(*lims[ax.get_xlabel()])
        y_max = 0.0
        for patch in ax.patches:
            if patch.__class__.__name__ == 'Rectangle':
                continue
            vertices = patch.get_path().vertices
            if len(vertices):
                y_max = max(y_max, float(np.nanmax(vertices[:, 1])))
        if y_max > 0:
            ax.set_ylim(0.0, y_max * 1.08)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.set_ylabel('')

    for ax in fig.axes:
        if ax is None:
            continue
        for artist in [*ax.lines, *ax.patches, *ax.collections]:
            artist.set_clip_on(True)
            artist.set_zorder(2)
        for spine in ax.spines.values():
            spine.set_zorder(10)

    axes.iloc[-1, 0].legend(
        bbox_to_anchor=(len(axes) * 0.85, len(axes) * 0.8),
        loc='lower center',
        fontsize=14,
    )
    fig.tight_layout()
    axes.tick_params(grid_alpha=0)

    path = os.path.join(out_dir, out_name)
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.png")
    plt.close(fig)


#plots corner plot for specified samples
samples_dict = {
    "conditioned": conditioned_samples,
    "unconditioned": unconditioned_samples,
    "transient": transient_samples,
}
samples = samples_dict[run_corner]
make_corner(
    [(samples,"IMRPhenomD NRTidalv2","C0")],
    params=[
        "M_c",
        "q",
        "iota",
        "d_L",
        "ra",
        "dec",
        "t_c"
    ],
    out_dir=dir_path,
    out_name=f"{GRB_name}_{run_corner}_corner",
)
