#!/usr/bin/env python3
"""Generate first-pass data-quality dossiers for selected BayesLine candidates."""

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import welch


DEFAULT_CANDIDATES = (
    "GRB240824549",
    "GRB170403583",
    "GRB240817057",
    "GRB240615744",
    "GRB170121067",
    "GRB170817529",
    "GRB240715239",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grbs", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "SFM_baseline_GW170817_GRB_extended"
        / "data",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def load_weighted_tc(samples_path):
    with samples_path.open(newline="") as samples_file:
        reader = csv.reader(samples_file)
        header = next(reader)
        tc_column = header.index("t_c")
        values = []
        weights = []
        next(reader)
        next(reader)
        for row in reader:
            if len(row) <= tc_column:
                continue
            try:
                values.append(float(row[tc_column]))
                weights.append(float(row[1]))
            except ValueError:
                continue
    values = np.asarray(values)
    weights = np.asarray(weights)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0 or weights.sum() == 0:
        raise ValueError(f"No weighted t_c samples in {samples_path}")
    return values, weights / weights.sum()


def plot_strain_and_qtransform(grb, strains, gps, output_dir):
    figure, axes = plt.subplots(
        len(strains), 2, figsize=(13, 3.8 * len(strains)), squeeze=False
    )
    for row, (ifo, strain) in enumerate(strains.items()):
        plot_start = max(float(strain.t0.value), gps - 12.0)
        plot_end = min(float(strain.span[1]), gps)
        display_strain = strain.crop(plot_start, plot_end)
        time = display_strain.times.value - gps
        axes[row, 0].plot(time, display_strain.value, color="black", linewidth=0.45)
        axes[row, 0].axvline(0, color="tab:red", linewidth=0.9)
        axes[row, 0].set_ylabel(f"{ifo} strain")
        axes[row, 0].grid(alpha=0.25)

        qtransform = display_strain.q_transform(
            frange=(20, 1024),
            outseg=(plot_start, plot_end),
            qrange=(4, 64),
            mismatch=0.2,
        )
        image = axes[row, 1].pcolormesh(
            qtransform.times.value - gps,
            qtransform.frequencies.value,
            qtransform.value.T,
            shading="auto",
            cmap="magma",
        )
        axes[row, 1].axvline(0, color="cyan", linewidth=0.9)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylim(20, 2048)
        axes[row, 1].set_ylabel(f"{ifo} frequency [Hz]")
        figure.colorbar(image, ax=axes[row, 1], pad=0.02, label="Q-transform normalized energy")

    axes[-1, 0].set_xlabel("Time relative to GRB trigger [s]")
    axes[-1, 1].set_xlabel("Time relative to GRB trigger [s]")
    axes[0, 0].set_title("Strain")
    axes[0, 1].set_title("Whitened Q-transform")
    figure.suptitle(f"{grb}: BayesLine improved-PSD candidate data", y=1.01)
    figure.tight_layout()
    figure.savefig(output_dir / f"{grb}_strain_time_frequency.png", dpi=200)
    plt.close(figure)


def plot_periodograms(grb, strains, output_dir):
    figure, axis = plt.subplots(figsize=(9, 5))
    for ifo, strain in strains.items():
        frequencies, psd = welch(
            strain.value,
            fs=1.0 / strain.dt.value,
            window="hann",
            nperseg=min(16384, len(strain)),
            noverlap=min(8192, len(strain) // 2),
            scaling="density",
        )
        axis.loglog(frequencies[1:], psd[1:], label=ifo, linewidth=1.0)
    axis.set_xlim(20, 2048)
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("One-sided PSD [1/Hz]")
    axis.set_title(f"{grb}: on-source Welch periodograms")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{grb}_periodogram.png", dpi=200)
    plt.close(figure)


def plot_tc_posterior(grb, results_dir, output_dir):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, color in (("unconditioned", "tab:gray"), ("conditioned", "tab:blue")):
        samples_path = results_dir / f"{grb}_{label}_Transient.csv"
        values, weights = load_weighted_tc(samples_path)
        axis.hist(
            values,
            bins=80,
            weights=weights,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=label,
        )
    axis.axvline(0, color="tab:red", linewidth=1.0, label="GRB trigger")
    axis.set_xlabel(r"Coalescence time $t_c$ relative to GRB [s]")
    axis.set_ylabel("Posterior density")
    axis.set_title(f"{grb}: coalescence-time posterior")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{grb}_tc_posterior.png", dpi=200)
    plt.close(figure)


def make_dossier(grb, data_root, results_root):
    data_dir = data_root / grb
    results_dir = results_root / grb
    output_dir = results_dir / "candidate_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"{grb}_metadata.pkl").open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)
    gps = float(metadata["GPS"])
    strains = {
        ifo: TimeSeries.read(data_dir / f"{grb}_{ifo}.hdf5", format="hdf5").crop(
            float(metadata["start"]), float(metadata["end"])
        )
        for ifo in metadata["detectors"]
    }
    plot_strain_and_qtransform(grb, strains, gps, output_dir)
    plot_periodograms(grb, strains, output_dir)
    plot_tc_posterior(grb, results_dir, output_dir)
    print(f"Completed diagnostic dossier: {output_dir}")


def main():
    args = parse_args()
    for grb in args.grbs:
        make_dossier(grb, args.data_root, args.results_root)


if __name__ == "__main__":
    main()