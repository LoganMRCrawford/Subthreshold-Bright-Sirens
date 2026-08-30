import os
import numpy as np
import sys
from anesthetic import make_2d_axes
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from _plot_utils import (
    load_nested_csv,
    map_from_hist,
    compute_hpd_samples,
)

####################
##INITIAL SETTINGS##
####################

DATA_DIR = "../data"

#SELECTED_GRBS = []  #empty list to process ALL GRB directories
SELECTED_GRBS = ["GRB170816599"]

#optional settings
strain_plotting = True
logZ_and_maps = False
corner_plotting = False


if len(SELECTED_GRBS) == 0:
    GRBS_TO_PROCESS = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and d.startswith("GRB")
    ])
    print(f"Processing ALL {len(GRBS_TO_PROCESS)} GRB directories")

else:
    GRBS_TO_PROCESS = SELECTED_GRBS
    print(f"Processing selected GRBs: {GRBS_TO_PROCESS}")



###################
##STRAIN PLOTTING##
###################

if strain_plotting == True:
    for GRB_name in GRBS_TO_PROCESS:
        print()
        print("=" * 70)
        print(f"PROCESSING {GRB_name}")
        print("=" * 70)

        dir_path = os.path.join(
            DATA_DIR,
            GRB_name
        )

        #reads in metadata
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
        successful_detectors = metadata["detectors"]

        #reads in strain data for each successful detector
        GWOSC_LOCAL_FILES = {}
        strains = []
        labels = []
        detector_labels = {
            "H1": "LIGO Hanford",
            "L1": "LIGO Livingston",
            "V1": "Virgo",
        }
        for ifo_name in successful_detectors:
            strain_filename = f"{GRB_name}_{ifo_name}.hdf5"
            GWOSC_LOCAL_FILES[ifo_name] = os.path.join(
                dir_path,
                strain_filename
            )
            path = GWOSC_LOCAL_FILES[ifo_name]
            strain = TimeSeries.read(
                path,
                format="hdf5"
            ).crop(start, end)
            strains.append(strain)
            labels.append(detector_labels[ifo_name])


        #plots and saves strain data for each successful detector
        fig, axes = plt.subplots(
            len(strains), 1,
            figsize=(12, 3 * len(strains)),
            sharex=True
        )
        if len(strains) == 1:
            axes = [axes]
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
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                dir_path,
                f"{GRB_name}_strain_data.png"
            ),
            dpi=600,
            bbox_inches="tight"
        )
        plt.show()


#################
##LOGZ AND MAPS##
#################

if logZ_and_maps == True:
    results_list = []
    for GRB_name in GRBS_TO_PROCESS:
        print()
        print("=" * 70)
        print(f"PROCESSING LOGZ AND MAPS: {GRB_name}")
        print("=" * 70)

        dir_path = os.path.join(
            "../data",
            GRB_name
        )

        #rereads in metadata for each GRB
        metadata_filename = f"{GRB_name}_metadata.pkl"
        metadata_file = os.path.join(
            dir_path,
            metadata_filename
        )
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        gps = float(metadata["GPS"])
        grb_ra = float(metadata["RA_rad"])
        grb_dec = float(metadata["DEC_rad"])

        #read in samples and calculate logZ
        tran_unconditioned_path = os.path.join(dir_path, f"{GRB_name}_unconditioned_Transient.csv")
        tran_unconditioned_samples = load_nested_csv(tran_unconditioned_path)
        tran_conditioned_path = os.path.join(dir_path, f"{GRB_name}_conditioned_Transient.csv")
        tran_conditioned_samples = load_nested_csv(tran_conditioned_path)
        logZ_tran_unconditioned = tran_unconditioned_samples.logZ(100).mean()
        logZ_tran_unconditioned_err = tran_unconditioned_samples.logZ(100).std()
        logZ_tran_conditioned = tran_conditioned_samples.logZ(100).mean()
        logZ_tran_conditioned_err = tran_conditioned_samples.logZ(100).std()
        delta_logZ_tran = logZ_tran_conditioned - logZ_tran_unconditioned
        delta_logZ_err_tran = np.sqrt(logZ_tran_unconditioned_err**2 + logZ_tran_conditioned_err**2)
        print(f"Delta logZ (Transient):    {delta_logZ_tran:.2f} +/- {delta_logZ_err_tran:.2f}")

        #ARCHAIC: commented out code for heterodyned logZ calculations
        # transient_path = os.path.join(dir_path, f"{GRB_name}_Transient.csv")
        # transient_samples = load_nested_csv(transient_path)
        # hetero_unconditioned_path = os.path.join(dir_path, f"{GRB_name}_unconditioned_Heterodyned.csv")
        # hetero_unconditioned_samples = load_nested_csv(hetero_unconditioned_path)
        # hetero_conditioned_path = os.path.join(dir_path, f"{GRB_name}_conditioned_Heterodyned.csv")
        # hetero_conditioned_samples = load_nested_csv(hetero_conditioned_path)
        # logZ_transient = transient_samples.logZ(100).mean()
        # logZ_transient_err = transient_samples.logZ(100).std()
        # logZ_hetero_unconditioned = hetero_unconditioned_samples.logZ(100).mean()
        # logZ_hetero_unconditioned_err = hetero_unconditioned_samples.logZ(100).std()
        # logZ_hetero_conditioned = hetero_conditioned_samples.logZ(100).mean()
        # logZ_hetero_conditioned_err = hetero_conditioned_samples.logZ(100).std()
        # delta_logZ_hetero = logZ_hetero_conditioned - logZ_hetero_unconditioned
        # delta_logZ_err_hetero = np.sqrt(logZ_hetero_unconditioned_err**2 + logZ_hetero_conditioned_err**2)
        # print(f"Delta logZ (Heterodyned):    {delta_logZ_hetero:.2f} +/- {delta_logZ_err_hetero:.2f}")

        #saves GRB info and logZ results to dictionary for saving to master CSV
        results = {
            "GRB": GRB_name,
            "GPS": gps,
            "RA_rad": grb_ra,
            "DEC_rad": grb_dec,
            "detectors": ",".join(metadata["detectors"]),

            # "logZ_Transient": logZ_transient,
            # "logZ_Transient_err": logZ_transient_err,

            "logZ_Unconditioned_tran": logZ_tran_unconditioned,
            "logZ_Unconditioned_err_tran": logZ_tran_unconditioned_err,

            "logZ_Conditioned_tran": logZ_tran_conditioned,
            "logZ_Conditioned_err_tran": logZ_tran_conditioned_err,

            "delta_logZ_tran": delta_logZ_tran,
            "delta_logZ_err_tran": delta_logZ_err_tran,

            # "logZ_Unconditioned_hetero": logZ_hetero_unconditioned,
            # "logZ_Unconditioned_err_hetero": logZ_hetero_unconditioned_err,

            # "logZ_Conditioned_hetero": logZ_hetero_conditioned,
            # "logZ_Conditioned_err_hetero": logZ_hetero_conditioned_err,

            # "delta_logZ_hetero": delta_logZ_hetero,
            # "delta_logZ_err_hetero": delta_logZ_err_hetero,
        }

        #calculates MAP values for each parameter over all runs and saves to results dictionary
        PARAM_NAMES = [
            "M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c",
            "psi", "ra", "dec", "lambda_1", "lambda_2",
        ]
        runs = [
            # ("Transient", transient_samples),
            ("Transient Unconditioned", tran_unconditioned_samples),
            ("Transient Conditioned", tran_conditioned_samples),
            # ("Heterodyned Unconditioned", hetero_unconditioned_samples),
            # ("Heterodyned Conditioned", hetero_conditioned_samples),
        ]
        for run_name, samples in runs:
            weights = np.asarray(
                samples.get_weights(),
                dtype=float
            ).copy()
            weights /= weights.sum()
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
                results[f"{run_name}_MAP_{param}"] = map_value
        results_list.append(results)
        print(f"✓ Finished {GRB_name}")
    
    #saves results to master CSV file, appending to existing file if it exists
    if len(results_list) > 0:
        results_df = pd.DataFrame(results_list)
        master_csv = os.path.join(
            "../data",
            "grb_results.csv"
        )
        if os.path.exists(master_csv):
            existing_df = pd.read_csv(master_csv)
            #remove GRBs that have just been reprocessed
            existing_df = existing_df[
                ~existing_df["GRB"].isin(
                    results_df["GRB"]
                )
            ]
            results_df = pd.concat(
                [
                    existing_df,
                    results_df
                ],
                ignore_index=True
            )
        results_df.to_csv(
            master_csv,
            index=False
        )



###################
##CORNER PLOTTING##
###################

#function taken from _plot_utils.py file
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


if corner_plotting == True:
    for GRB_name in GRBS_TO_PROCESS:
        print()
        print("=" * 70)
        print(f"PLOTTING CORNER PLOTS: {GRB_name}")
        print("=" * 70)

        dir_path = os.path.join(
            "../data",
            GRB_name
        )

        #plots corner for unconditioned samples
        make_corner(
            [(tran_unconditioned_samples,"IMRPhenomD NRTidalv2","C0")],
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
            out_name=f"{GRB_name}_unconditioned_corner",
        )
        #plots corner for conditioned samples
        make_corner(
            [(tran_conditioned_samples,"IMRPhenomD NRTidalv2","C0")],
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
            out_name=f"{GRB_name}_conditioned_corner",
        )
