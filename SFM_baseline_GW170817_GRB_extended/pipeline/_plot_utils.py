"""
Shared plotting utilities for GW paper figures.

Provides:
  - LaTeX rendering setup
  - Data loaders for nested sampling CSVs, reweighted CSVs, and GWTC reference
  - H_0 plotting with MAP, HPD credible intervals, and SHoES/Planck bands
  - Consistent color scheme across all plots
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import shutil
from anesthetic import MCMCSamples, read_chains, make_2d_axes
from scipy.stats import gaussian_kde
import pandas as pd

# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
if False: # shutil.which('pdflatex') or shutil.which('latex'):
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern']
else:
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


# #SFM CHANGED
# _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# RESULTS_DIR = os.path.join(_ROOT, "Results")
# OUT_DIR = os.path.join(RESULTS_DIR, "plots")

# os.makedirs(OUT_DIR, exist_ok=True)




# _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# RESULTS_DIR = os.path.join(_ROOT, 'results')
# OUT_DIR = os.path.join(_ROOT, 'results', 'gwtc1_phasemarg', 'plots')
# os.makedirs(OUT_DIR, exist_ok=True)

# GWTC1_HDF5 = os.environ.get('GWTC1_HDF5', os.path.join(RESULTS_DIR, 'GW170817_GWTC-1.hdf5'))
# GWTC2P1_GW150914_HDF5 = os.environ.get(
#     'GWTC2P1_GW150914_HDF5',
#     os.path.join(RESULTS_DIR, 'IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5'),
# )



# --------------------------------------------------------------------------- #
# Consistent color scheme
# --------------------------------------------------------------------------- #
COLORS = {
    # Data lines — each source gets a unique, distinct color
    'gwtc':           'tab:blue',
    'imr_baseline':   'maroon',
    'tf2_baseline':   'tab:purple',
    'flatZ':          'teal',
    'vp250':          'tab:red',
    'reweighted':     'tab:cyan',
    'unhetero_imr':   '#555555',
    'unhetero_tf2':   '#888888',
    'small_h0_imr':   'tab:brown',
    'small_h0_tf2':   'tab:pink',
    # Planck / SHoES reference bands — traditional colors
    'planck_inner':   '#0CDE79',
    'planck_outer':   '#6DE6AC',
    'shoes_inner':    '#E87317',
    'shoes_outer':    '#F19851',
}

# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_nested_csv(csv_path):
    """Load an anesthetic nested sampling CSV."""
    s = read_chains(csv_path)
    print(f"  Loaded {csv_path}  ({len(s)} samples)")
    return s


def load_reweighted_csv(csv_path):
    """Load a reweighted CSV (plain CSV with a 'weight' column)."""
    df = pd.read_csv(csv_path)
    weights = df['weight'].to_numpy()
    cols = [c for c in df.columns if c != 'weight']
    s = MCMCSamples(df[cols].to_numpy(), columns=cols, weights=weights)
    print(f"  Loaded {csv_path}  ({len(s)} samples, reweighted)")
    return s


def load_gwtc1_gw170817(columns=None):
    """Load GWTC-1 GW170817 posteriors, returning MCMCSamples.

    Default columns: M_c, q, s1_z, s2_z, d_L, iota
    """
    dataset = 'IMRPhenomPv2NRT_lowSpin_posterior'
    with h5py.File(GWTC1_HDF5, 'r') as f:
        data = f[dataset][:]

    m1 = data['m1_detector_frame_Msun']
    m2 = data['m2_detector_frame_Msun']
    M_c = (m1 * m2)**0.6 / (m1 + m2)**0.2
    q = m2 / m1
    d_L = data['luminosity_distance_Mpc']
    iota = np.arccos(data['costheta_jn'])
    s1_z = data['spin1'] * data['costilt1']
    s2_z = data['spin2'] * data['costilt2']

    if columns is None:
        columns = ['M_c', 'q', 's1_z', 's2_z', 'd_L', 'iota']
    col_map = {
        'M_c': M_c, 'q': q, 'd_L': d_L, 'iota': iota,
        's1_z': s1_z, 's2_z': s2_z,
    }
    arr = np.column_stack([col_map[c] for c in columns])
    return MCMCSamples(arr, columns=columns)


def load_gwtc2p1_gw150914():
    """Load GWTC-2p1 GW150914 posteriors for corner plot comparison."""
    dataset = 'C01:IMRPhenomXPHM/posterior_samples'
    with h5py.File(GWTC2P1_GW150914_HDF5, 'r') as f:
        data = f[dataset][:]

    M_c = data['chirp_mass']
    q = data['mass_ratio']
    d_L = data['luminosity_distance']
    iota = data['iota']
    chi_eff = data['chi_eff']

    cols = [r'$\mathcal{M}_c$', r'$q$', r'$\chi_{\rm eff}$', r'$d_L$', r'$\iota$']
    return MCMCSamples(
        np.column_stack([M_c, q, chi_eff, d_L, iota]), columns=cols,
    )


# --------------------------------------------------------------------------- #
# HPD interval computation
# --------------------------------------------------------------------------- #
def compute_hpd(x_eval, pdf_vals, cred_level):
    """Compute HPD (highest posterior density) interval boundaries from a pdf grid."""
    dx = x_eval[1] - x_eval[0]
    total_area = np.sum(pdf_vals) * dx
    sorted_pdf = np.sort(pdf_vals)[::-1]
    cumarea = np.cumsum(sorted_pdf) * dx / total_area
    threshold = sorted_pdf[np.searchsorted(cumarea, cred_level)]
    above = pdf_vals >= threshold
    indices = np.where(above)[0]
    return x_eval[indices[0]], x_eval[indices[-1]]


def compute_hpd_samples(x, w, cred_level):
    """HPD interval computed directly from weighted samples — no KDE smoothing.

    Returns the shortest interval [lo, hi] that contains `cred_level` of the
    cumulative weight. Robust to multimodality: returns the convex hull of
    the highest-density region implied by the empirical CDF, which is the
    correct frequentist construction for a single-mode or unimodal-dominated
    posterior. For genuinely multimodal posteriors prefer a histogram-based
    level-set, but for H_0 in this paper the empirical-CDF version is what
    the user asked for.
    """
    x = np.asarray(x); w = np.asarray(w, dtype=float)
    w = w / w.sum()
    order = np.argsort(x)
    xs = x[order]; ws = w[order]
    cdf = np.cumsum(ws)
    target = cred_level
    n = len(xs)
    # Slide a window of cumulative weight `target` across the sorted samples
    # and return the narrowest one. O(n) after sort.
    j = 0
    best = (np.inf, xs[0], xs[-1])
    for i in range(n):
        while j < n and cdf[j] - (cdf[i-1] if i > 0 else 0.0) < target:
            j += 1
        if j >= n:
            break
        width = xs[j] - xs[i]
        if width < best[0]:
            best = (width, xs[i], xs[j])
    return best[1], best[2]


def map_from_hist(x, w, bins):
    """MAP estimate from a weighted histogram (bin centre with max density)."""
    counts, edges = np.histogram(x, bins=bins, weights=w)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return float(centres[int(np.argmax(counts))])


# --------------------------------------------------------------------------- #
# LVK H_0 derivation from luminosity-distance samples
# --------------------------------------------------------------------------- #
# Standard-siren constants matching GW170817_heterodyned_*.py:
#   v_obs ~ N(v_p + H_0 * d_L, sigma_v),  v_obs = 3327 km/s,  sigma_v = 72 km/s
#   v_p   ~ N(310, 150) km/s
# Marginalising v_p analytically: v_obs | H_0, d_L ~ N(310 + H_0 * d_L, sqrt(72^2 + 150^2))
V_OBS_NGC4993       = 3327.0   # km/s — observed recession velocity (NGC 4993, helio-frame)
V_P_PRIOR_MEAN      =  310.0   # km/s — peculiar velocity prior mean
V_P_PRIOR_SIGMA     =  150.0   # km/s — peculiar velocity prior sigma
SIGMA_V_OBS         =   72.0   # km/s — measurement sigma on v_r
SIGMA_V_MARG        = float(np.sqrt(SIGMA_V_OBS**2 + V_P_PRIOR_SIGMA**2))


def derive_lvk_h0_samples(d_L_mpc, rng=None):
    """Map LVK d_L posterior samples to H_0 samples using our standard-siren model.

    For each input d_L (Mpc) draw one H_0 sample from
        H_0 | d_L  ~  N((V_OBS - V_P_MEAN) / d_L,  SIGMA_V_MARG / d_L)
    which is what the user's likelihood reduces to once v_p is marginalised
    out under the prior in GW170817_heterodyned_1.py. This is the right way
    to plot 'LVK posterior samples' against this work in H_0 space — it uses
    LVK's d_L distribution but our recession-velocity model, so any H_0
    difference between LVK and this work is purely the d_L difference.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d_L_mpc = np.asarray(d_L_mpc, dtype=float)
    mu = (V_OBS_NGC4993 - V_P_PRIOR_MEAN) / d_L_mpc
    sd = SIGMA_V_MARG / d_L_mpc
    return mu + sd * rng.standard_normal(d_L_mpc.shape)


# --------------------------------------------------------------------------- #
# H_0 plot with MAP, HPD, SHoES, Planck
# --------------------------------------------------------------------------- #
def plot_h0(runs, out_name, xlim=(20, 250), n_eval=2000):
    """Create an H_0 posterior plot using weighted KDE.

    Area-normalised KDE with MAP, HPD credible intervals, SHoES and Planck bands.

    Parameters
    ----------
    runs : list of (samples_or_dict, label, color)
        Each entry is (anesthetic samples object OR dict with 'H_0' and 'weights'), label, color.
    out_name : str
        Output filename stem (saved to OUT_DIR).
    xlim : tuple
        x-axis limits.
    n_eval : int
        Number of points for KDE evaluation grid.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x_eval = np.linspace(xlim[0], xlim[1], n_eval)

    for samples, label, color in runs:
        # Extract H_0 values and weights
        if isinstance(samples, dict):
            h0_vals = samples['H_0']
            weights = samples['weights']
        else:
            h0_vals = samples['H_0'].to_numpy()
            weights = np.asarray(samples.get_weights())

        weights = weights / weights.sum()

        # Weighted KDE
        kde = gaussian_kde(h0_vals, weights=weights)
        pdf_vals = kde(x_eval)

        # Area-normalise
        _trapz = getattr(np, 'trapezoid', np.trapz)
        pdf_vals = pdf_vals / _trapz(pdf_vals, x_eval)

        # Plot
        ax.plot(x_eval, pdf_vals, color=color, lw=2, label=label)

        # MAP
        map_val = x_eval[np.argmax(pdf_vals)]
        print(f"  {label}: H_0 MAP = {map_val:.1f} km/s/Mpc")

        # HPD intervals
        for cred_level, sigma_label, ls in [(0.68269, r'1$\sigma$', '--'),
                                             (0.95450, r'2$\sigma$', ':')]:
            lo, hi = compute_hpd(x_eval, pdf_vals, cred_level)
            ax.axvline(lo, color=color, ls=ls, lw=1.2, alpha=0.7)
            ax.axvline(hi, color=color, ls=ls, lw=1.2, alpha=0.7)
            print(f"    {sigma_label} HPD: [{lo:.1f}, {hi:.1f}]")

    # Planck and SHoES reference bands
    ax.axvspan(65.7, 68.2, color=COLORS['planck_outer'], alpha=0.3, zorder=0)
    ax.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
               alpha=0.3, zorder=0, label='Planck')
    ax.axvspan(69.76, 76.72, color=COLORS['shoes_outer'], alpha=0.3, zorder=0)
    ax.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
               alpha=0.3, zorder=0, label='SHoES')

    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_ylabel(r'$P(H_0)$ (km$^{-1}$ s Mpc)')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.pdf / .png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# H_0 plot — weighted-histogram version (no KDE smoothing on 1-D marginals)
# --------------------------------------------------------------------------- #
def plot_h0_hist(runs, out_name, xlim=(40, 180), bins=140,
                 add_planck_shoes=True, lvk_band=False, hpd_lines=True,
                 figsize=(10, 6)):
    """H_0 posterior plot using weighted step-histograms with sample-derived HPDs.

    Parameters
    ----------
    runs : list of (samples_or_dict, label, color)
        Each entry is (anesthetic samples / dict with 'H_0','weights' / tuple of (h0,w), label, color).
    out_name : str
        Output filename stem (saved to OUT_DIR; .pdf and .png written).
    xlim : (lo, hi)
    bins : int or array
        Number of histogram bins or bin edges. If int, uniform bins on xlim.
    add_planck_shoes : bool
    lvk_band : bool
        If True, also draw the Abbott+2017 70 [62,82] band (legacy compatibility;
        prefer adding LVK as a real run in `runs` instead).
    hpd_lines : bool
        Draw vertical dashed/dotted lines at 68% and 95% sample-HPD endpoints per run.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(bins, int):
        bins = np.linspace(xlim[0], xlim[1], bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])

    for entry in runs:
        samples, label, color = entry
        if isinstance(samples, dict):
            h0 = np.asarray(samples['H_0'], dtype=float)
            w = np.asarray(samples['weights'], dtype=float)
        elif isinstance(samples, tuple):
            h0, w = (np.asarray(a, dtype=float) for a in samples)
        else:
            h0 = samples['H_0'].to_numpy().astype(float)
            w = np.asarray(samples.get_weights(), dtype=float)
        w = w / w.sum()

        counts, _ = np.histogram(h0, bins=bins, weights=w, density=True)
        # Step plot — pre-/post-end zeros so the line closes at the baseline.
        x_step = np.r_[bins[0], np.repeat(bins[1:-1], 2), bins[-1]]
        y_step = np.r_[np.repeat(counts, 2)]
        ax.plot(x_step, y_step, color=color, lw=2.0, label=label, drawstyle='default')

        map_ = float(centres[int(np.argmax(counts))])
        lo68, hi68 = compute_hpd_samples(h0, w, 0.68269)
        lo95, hi95 = compute_hpd_samples(h0, w, 0.95450)
        print(f"  {label}: MAP={map_:.1f}; 68% HPD=[{lo68:.1f},{hi68:.1f}]; 95% HPD=[{lo95:.1f},{hi95:.1f}]")
        if hpd_lines:
            for v, ls in [(lo68, '--'), (hi68, '--'), (lo95, ':'), (hi95, ':')]:
                ax.axvline(v, color=color, ls=ls, lw=1.0, alpha=0.6)

    if lvk_band:
        ax.axvline(70.0, color='0.45', ls='-', lw=1.5, zorder=1,
                   label=r'Abbott+2017 ($70^{+12}_{-8}$)')

    if add_planck_shoes:
        ax.axvspan(65.7, 68.2, color=COLORS['planck_outer'], alpha=0.3, zorder=0)
        ax.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
                   alpha=0.3, zorder=0, label='Planck')
        ax.axvspan(69.76, 76.72, color=COLORS['shoes_outer'], alpha=0.3, zorder=0)
        ax.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
                   alpha=0.3, zorder=0, label='SH0ES')

    ax.set_xlim(xlim); ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_ylabel(r'$P(H_0)$ (km$^{-1}$ s Mpc)')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.pdf / .png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# H_0 plot — Silverman-bandwidth KDE (smoothed 1-D marginals)
# --------------------------------------------------------------------------- #
def plot_h0_kde(runs, out_name, xlim=(40, 180), n_eval=1000,
                add_planck_shoes=True, lvk_band=False, hpd_lines=False,
                figsize=(10, 6)):
    """H_0 posterior plot using Silverman-bandwidth Gaussian KDE.

    Parameters
    ----------
    runs : list of (samples_or_dict_or_tuple, label, color [, linestyle])
    out_name : str
    xlim : (lo, hi)
    n_eval : int — number of evaluation points for KDE
    add_planck_shoes : bool
    lvk_band : bool
    hpd_lines : bool
    """
    from scipy.stats import gaussian_kde as _gkde
    _trapz = getattr(np, 'trapezoid', None) or np.trapz

    fig, ax = plt.subplots(figsize=figsize)
    x_eval = np.linspace(xlim[0], xlim[1], n_eval)

    for entry in runs:
        samples, label, color = entry[:3]
        ls = entry[3] if len(entry) > 3 else '-'
        if isinstance(samples, dict):
            h0 = np.asarray(samples['H_0'], dtype=float)
            w = np.asarray(samples['weights'], dtype=float)
        elif isinstance(samples, tuple):
            h0, w = (np.asarray(a, dtype=float) for a in samples)
        else:
            h0 = samples['H_0'].to_numpy().astype(float)
            w = np.asarray(samples.get_weights(), dtype=float)
        finite = np.isfinite(h0) & np.isfinite(w) & (w > 0)
        h0, w = h0[finite], w[finite]
        w = w / w.sum()

        kde = _gkde(h0, weights=w, bw_method='silverman')
        pdf = kde(x_eval)
        pdf /= _trapz(pdf, x_eval)
        ax.plot(x_eval, pdf, color=color, lw=2.0, ls=ls, label=label)

        lo68, hi68 = compute_hpd_samples(h0, w, 0.68269)
        lo95, hi95 = compute_hpd_samples(h0, w, 0.95450)
        map_val = float(x_eval[np.argmax(pdf)])
        print(f"  {label}: MAP={map_val:.1f}; 68% HPD=[{lo68:.1f},{hi68:.1f}]; 95% HPD=[{lo95:.1f},{hi95:.1f}]")
        if hpd_lines:
            for v, dls in [(lo68, '--'), (hi68, '--'), (lo95, ':'), (hi95, ':')]:
                ax.axvline(v, color=color, ls=dls, lw=0.8, alpha=0.55)

    if lvk_band:
        ax.axvline(70.0, color='0.45', ls='-', lw=1.5, zorder=1,
                   label=r'Abbott+2017 ($70^{+12}_{-8}$)')

    if add_planck_shoes:
        ax.axvspan(65.7, 68.2, color=COLORS['planck_outer'], alpha=0.3, zorder=0)
        ax.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
                   alpha=0.3, zorder=0, label='Planck')
        ax.axvspan(69.76, 76.72, color=COLORS['shoes_outer'], alpha=0.3, zorder=0)
        ax.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
                   alpha=0.3, zorder=0, label='SH0ES')

    ax.set_xlim(xlim); ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_ylabel(r'$P(H_0)$ (km$^{-1}$ s Mpc)')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    ax.legend(frameon=False, fontsize=11, loc='upper right')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.pdf / .png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Corner plot helper
# --------------------------------------------------------------------------- #
def make_corner(datasets, params, out_name, figsize=(10, 10), lims=None):
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

    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.pdf / .png")
    plt.close(fig)
