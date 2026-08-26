import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _plot_utils import (
    load_nested_csv,
    map_from_hist,
    compute_hpd_samples,
)


####################
##INITIAL SETTINGS##
####################

DATA_DIR = "../data"
RESULTS_FILE = os.path.join(DATA_DIR, "grb_results.csv")

# SELECTED_GRBS = []  #empty list = process ALL GRBs
SELECTED_GRBS = ["GRB190810675","GRB240603102"]


PARAM_NAMES = [
    "M_c",
    "q",
    "s1_z",
    "s2_z",
    "iota",
    "d_L",
    "t_c",
    "psi",
    "ra",
    "dec",
    "lambda_1",
    "lambda_2",
]

PARAM_LABELS = {
    "M_c": r"$\mathcal{M}_c$",
    "q": r"$q$",
    "s1_z": r"$s_{1,z}$",
    "s2_z": r"$s_{2,z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L$",
    "t_c": r"$t_c$",
    "psi": r"$\psi$",
    "ra": r"$\mathrm{RA}$",
    "dec": r"$\mathrm{Dec}$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
}


####################
## READ RESULTS FILE
####################

df = pd.read_csv(RESULTS_FILE)
print(f"Number of GRBs in results file: {len(df)}")
table = df[
    [
        "GRB",
        "logZ_Unconditioned_tran",
        "logZ_Unconditioned_err_tran",
        "logZ_Conditioned_tran",
        "logZ_Conditioned_err_tran",
        "delta_logZ_tran",
        "delta_logZ_err_tran",
    ]
].copy()

#sorts from highest to lowest Delta logZ
table = table.sort_values(
    "delta_logZ_tran",
    ascending=False
)
table["Unconditioned"] = (
    table["logZ_Unconditioned_tran"].map("{:.3f}".format)
    + " +/- "
    + table["logZ_Unconditioned_err_tran"].map("{:.3f}".format)
)
table["Conditioned"] = (
    table["logZ_Conditioned_tran"].map("{:.3f}".format)
    + " +/- "
    + table["logZ_Conditioned_err_tran"].map("{:.3f}".format)
)
table["Delta logZ"] = (
    table["delta_logZ_tran"].map("{:.3f}".format)
    + " +/- "
    + table["delta_logZ_err_tran"].map("{:.3f}".format)
)
table = table[
    [
        "GRB",
        "Unconditioned",
        "Conditioned",
        "Delta logZ",
    ]
]

print()
print("=" * 90)
print("TRANSIENT LOG EVIDENCE RESULTS")
print("=" * 90)
print(table.to_string(index=False))
print("=" * 90)
print(f"Number of GRBs: {len(df)}")


################
##SELECT GRBs##
###############

if len(SELECTED_GRBS) == 0:
    GRBS_TO_PROCESS = sorted(
        df["GRB"].dropna().unique()
    )
    print(
        f"\nPlotting ALL {len(GRBS_TO_PROCESS)} GRBs"
    )

else:
    GRBS_TO_PROCESS = SELECTED_GRBS
    print(
        f"\nPlotting selected GRBs: {GRBS_TO_PROCESS}"
    )

#loops over GRBs to make MAP and median plots
for GRB_name in GRBS_TO_PROCESS:
    print()
    print("=" * 70)
    print(f"PROCESSING PLOT DATA: {GRB_name}")
    print("=" * 70)
    grb_dir = os.path.join(DATA_DIR, GRB_name)

    #load samples
    unconditioned_path = os.path.join(
        grb_dir,
        f"{GRB_name}_unconditioned_Transient.csv"
    )
    conditioned_path = os.path.join(
        grb_dir,
        f"{GRB_name}_conditioned_Transient.csv"
    )
    unconditioned = load_nested_csv(unconditioned_path)
    conditioned = load_nested_csv(conditioned_path)
    runs = [
        ("Unconditioned", unconditioned),
        ("Conditioned", conditioned),
    ]

    # setting up plots
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(20, 14),
        constrained_layout=True
    )
    axes = axes.flatten()

    # calculates MAP and percentiles
    for ax, param in zip(axes, PARAM_NAMES):
        map_values = []
        median_values = []
        lower_errors = []
        upper_errors = []
        for run_name, samples in runs:
            values = samples[param].to_numpy()
            weights = np.asarray(
                samples.get_weights(),
                dtype=float
            ).copy()
            weights /= weights.sum()

            #calculates MAP using function from _plot_utils.py
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

            #calculates weighted percentiles
            sort_idx = np.argsort(values)
            sorted_values = values[sort_idx]
            sorted_weights = weights[sort_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            cumulative_weights /= cumulative_weights[-1]
            p16 = np.interp(
                0.16,
                cumulative_weights,
                sorted_values
            )
            median = np.interp(
                0.50,
                cumulative_weights,
                sorted_values
            )
            p84 = np.interp(
                0.84,
                cumulative_weights,
                sorted_values
            )

            #saves results
            map_values.append(map_value)
            median_values.append(median)
            lower_errors.append(
                median - p16
            )
            upper_errors.append(
                p84 - median
            )

        #plotting
        x = np.array([0.3, 0.7])
        ax.set_xlim(0,1)
        yerr = np.array([
            lower_errors,
            upper_errors
        ])

        #median + 16-84 percentile interval
        ax.errorbar(
            x,
            median_values,
            yerr=yerr,
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=1.5,
            label="Median (16–84%)"
        )

        #MAP separately
        ax.plot(
            x,
            map_values,
            "x",
            markersize=10,
            markeredgewidth=2,
            label="MAP"
        )

        #aesthetics
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Uncond", "Cond"], fontsize=16,
        )
        ax.set_title(
            PARAM_LABELS.get(param, param),
            fontsize=20
        )
    fig.suptitle(
        f"{GRB_name}: MAP and Medians (with 16-84% errorbars)",
        fontsize=24,
    )

    #saving
    output_path = os.path.join(
        grb_dir,
        f"{GRB_name}_MAP_comparison.png"
    )
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)
    print(f"✓ Saved {output_path}")