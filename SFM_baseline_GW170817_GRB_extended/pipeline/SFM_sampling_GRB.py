"""
Heterodyned Nested Sampling for GW170817 — flat-in-z prior
============================================================

Variant of GW170817_heterodyned_1.py with one change:
  d_L prior: flat in redshift z (= uniform in d_L, LVK convention)
  instead of Beta(3,1) ∝ d_L^2.
  H_0 prior remains log-uniform (unchanged from baseline).

Usage:
  python GW170817_heterodyned_2.py [--waveform {IMRPhenomD_NRTidalv2,TaylorF2}]
                                    [--ref-params {gwtc1,optimize}]
                                    [--phase-marginalization]
"""

# ============================================================================
# 0. IMPORTS & JAX CONFIGURATION
# ============================================================================
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import blackjax
import h5py
import time
import tqdm
from astropy.time import Time
from scipy.interpolate import interp1d
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise
import pandas as pd
import pickle
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD_NRTidalv2,
    RippleTaylorF2,
    RippleIMRPhenomXAS_NRTidalv3,
    RippleIMRPhenomPv2,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from gwpy.timeseries import TimeSeries

# ============================================================================
# 0.1. COMMAND-LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='Unconditioned and conditioned nested sampling')
parser.add_argument('--n-live', type=int, default=500,
                    help='Number of live points (default: 500)')
parser.add_argument('--num-delete', type=int, default=None,
                    help='Points deleted per NS iteration. Default: 0.3 * n_live (flatZ-script convention).')
parser.add_argument('--n-bins', type=int, default=501,
                    help='Number of heterodyne bins (default: 501).')
parser.add_argument('--seed', type=int, default=0,
                    help='JAX PRNGKey seed for reproducibility (default: 0).')
parser.add_argument('--n-mcmc', type=int, default=None, dest='n_mcmc',
                    help='Override the BlackJAX-NS num_inner_steps slice-step count. '
                         'Default: 8 * NUM_DIMS (=112 for phase-marginalised 14-d GW170817). '
                         'Used for the M7 convergence sweep at {5, 10, 20} * NUM_DIMS.')
parser.add_argument("--grb-names", nargs="+", type=str, required=True,
                    help="Names of GRBs to analyse from the saved GRB catalogue.")
args = parser.parse_args()


#data-source = local
#psd-source = self
#took out narrow sky, overriding bounds or reference waveform, narrow sky, mode B
#commented out H0 and vp fitting

# Numerically stable log I_0 for phase marginalization:
# log(I_0(x)) = log(i0e(x)) + x, where i0e(x) = exp(-|x|) * I_0(x)
from jax.scipy.special import i0e

@jax.jit
def log_i0(x):
    return jnp.log(i0e(x)) + x


# ============================================================================
# 1. EVENT CONFIGURATION & DETECTOR DATA
# ============================================================================


GRB_DATA_DIR = os.environ.get(
    'GRB_DATA_DIR',
    '../data'
)

GRB_CATALOGUE = os.environ.get(
    'GRB_CATALOGUE',
    '../data/grb_catalog.csv'
)

grb_df = pd.read_csv(GRB_CATALOGUE)

print(f"Loaded GRB catalogue: {len(grb_df)} GRBs")


grb_name = args.grb_names[0]

matches = grb_df[grb_df["GRB"] == grb_name]

if len(matches) == 0:
    raise ValueError(
        f"GRB {grb_name} not found in catalogue"
    )

selected_grb = matches.iloc[0]

print("\nSelected GRB:")
print(
    selected_grb[
        ["GRB", "GPS", "RA_rad", "DEC_rad", "T90_s", "LIGO_run"]
    ].to_string()
)

grb_dir = os.path.join(
    GRB_DATA_DIR,
    grb_name
)

metadata_filename = f"{grb_name}_metadata.pkl"

metadata_file = os.path.join(
    grb_dir,
    metadata_filename
)


with open(metadata_file, "rb") as f:
    metadata = pickle.load(f)


gps = float(metadata["GPS"])
start = float(metadata["start"])
end = float(metadata["end"])
psd_start = float(metadata["psd_start"])
psd_end = float(metadata["psd_end"])
grb_ra = float(metadata["RA_rad"])
grb_dec = float(metadata["DEC_rad"])


fmin = 23.0
fmax = 2048.0
duration = 128
roll_off = 0.4
tukey_alpha = 2 * roll_off / duration


print("\n===== TIME SEGMENTS =====")
print(f"GPS       : {gps}")
print(f"start     : {start}")
print(f"end       : {end}")
print(f"duration  : {end - start}")
print(f"psd_start : {psd_start}")
print(f"psd_end   : {psd_end}")
print(f"PSD dur.  : {psd_end - psd_start}")
print("=========================\n")

t0 = time.time()

# Local GWOSC HDF5 file mapping: ifo name -> file path

GWOSC_LOCAL_FILES = {}

for ifo_name in ["H1", "L1", "V1"]:
    strain_filename = f"{grb_name}_{ifo_name}.hdf5"
    GWOSC_LOCAL_FILES[ifo_name] = os.path.join(grb_dir, strain_filename)


def load_gwosc_local(ifo_name, gps_start, gps_end):
    """Load GWOSC strain from a local HDF5 file, slicing to [gps_start, gps_end]."""
    path = GWOSC_LOCAL_FILES[ifo_name]
    ts = TimeSeries.read(path, format='hdf5')
    ts = ts.crop(gps_start, gps_end)
    return Data(ts.value, ts.dt.value, ts.epoch.value, ifo_name)

def load_gwosc_local_gwpy(ifo_name, gps_start, gps_end):
    """Load GWOSC strain as a gwpy TimeSeries (for PSD estimation)."""
    path = GWOSC_LOCAL_FILES[ifo_name]
    ts = TimeSeries.read(path, format='hdf5')
    ts = ts.crop(gps_start, gps_end)
    return ts


# ============================================================================
# 2. PARAMETER CONFIGURATION (static arrays, no dicts in hot path)
# ============================================================================
# Aligned-spin tidal: 14 dims (phase-marg) / 15 dims (no marg).
# in-plane spin params in spherical coords. d_L prior type stays 0 (uniform =
# flat-in-z) for this flatZ-script variant.



# Aligned-spin tidal layout (matches the historical 14-D vector).
PARAM_NAMES = [
    "M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c",
    "psi", "ra", "dec", "lambda_1", "lambda_2", 
    #"H_0", "v_p",
]
PARAM_LABELS = [
    r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\iota$", r"$d_L$", r"$t_c$",
    r"$\psi$", r"$\alpha$", r"$\delta$", r"$\Lambda_1$", r"$\Lambda_2$", 
    #r"$H_0$", r"$v_p$",
]

# Static parameter indices (compile-time constants for array access)
I_MC, I_Q, I_S1Z, I_S2Z, I_IOTA, I_DL, I_TC = 0, 1, 2, 3, 4, 5, 6
I_PSI, I_RA, I_DEC, I_L1, I_L2 = 7, 8, 9, 10, 11
#I_H0, I_VP = 12, 13

_PRIOR_LO_BASE = [
    1.0, 0.125, -0.05, -0.05,             # M_c, q, s1_z, s2_z
    0.0, 20.0, -10.0,                       # iota, d_L, t_c
    0.0, 0.0, -jnp.pi / 2,                   # psi, ra, dec
    0.0, 0.0,                # lambda_1, lambda_2,
    #20.0, -1000.0,                 # H_0, v_p
]
_PRIOR_HI_BASE = [
    2.0, 1.00, 0.05, 0.05,                # M_c, q, s1_z, s2_z
    jnp.pi, 800.0, 0.0,                     # iota, d_L, t_c
    jnp.pi, 2 * jnp.pi, jnp.pi / 2,         # psi, ra, dec
    5000.0, 5000.0,          # lambda_1, lambda_2,
    #250.0, 1000.0,           # H_0, v_p
]
# d_L prior type: 0 (uniform) instead of 3 (Beta(3,1)) — flat-in-z.
#note removed final two items were 4 and 0 for both
_PRIOR_TYPE_BASE_U = [0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0]
_PRIOR_TYPE_BASE_C = [0, 0, 0, 0, 1, 0, 0, 0, 5, 5, 0, 0]


NUM_DIMS = len(PARAM_NAMES)

#ra and dec conditioning
RA_GAUSS_STD = 0.2
DEC_GAUSS_STD = 0.1

GAUSS_MEAN = jnp.zeros(NUM_DIMS)
GAUSS_SIGMA = jnp.ones(NUM_DIMS)

GAUSS_MEAN = GAUSS_MEAN.at[I_RA].set(grb_ra)
GAUSS_MEAN = GAUSS_MEAN.at[I_DEC].set(grb_dec)

GAUSS_SIGMA = GAUSS_SIGMA.at[I_RA].set(RA_GAUSS_STD)
GAUSS_SIGMA = GAUSS_SIGMA.at[I_DEC].set(DEC_GAUSS_STD)

PRIOR_LO = jnp.array(_PRIOR_LO_BASE)
PRIOR_HI = jnp.array(_PRIOR_HI_BASE)

# Component mass bounds (applied as hard cut in M_c-q space).
# Override via --m-comp-lo/--m-comp-hi (e.g. LVK low-spin BNS bounds [0.87, 1.74]).
M_COMP_LO =  0.5    # M_sun
M_COMP_HI =  7.7    # M_sun

# Prior type encoding: 0=uniform, 1=sin(iota), 2=cos(dec), 4=log-uniform(H_0)
# (d_L is now type 0/uniform; flat-in-z Jacobian added explicitly in logprior_fn)
PRIOR_TYPE_U = jnp.array(_PRIOR_TYPE_BASE_U)
PRIOR_TYPE_C = jnp.array(_PRIOR_TYPE_BASE_C)

# Pre-computed prior constants (avoid recomputation in JIT)
_PRIOR_RANGE = PRIOR_HI - PRIOR_LO
_PRIOR_LOG_RANGE = jnp.log(_PRIOR_RANGE)
_PRIOR_LOG_LOG_RATIO = jnp.log(jnp.log(PRIOR_HI / PRIOR_LO))
_BETA_LN = jax.scipy.special.betaln(3.0, 1.0)


# ============================================================================
# 3. VECTORIZED LOG-PRIOR (no Python loops, fully JIT-traced)
# ============================================================================

@jax.jit
def logprior_fn(x, prior_type):
    """Evaluate total log-prior for a flat parameter vector.

    Computes all prior types vectorially and selects via jnp.where.
    Includes hard cut on component masses: m1, m2 in [M_COMP_LO, M_COMP_HI].
    No Python loops, list comprehensions, or dict access.
    """
    in_bounds = (x >= PRIOR_LO) & (x <= PRIOR_HI)

    # Uniform: log(1/(hi-lo)) = -log(hi-lo)
    lp_uniform = jnp.where(in_bounds, -_PRIOR_LOG_RANGE, -jnp.inf)

    # Sin prior (iota): log(sin(x)/2) on [0, pi]
    lp_sin = jnp.where(in_bounds, jnp.log(jnp.abs(jnp.sin(x)) + 1e-300) - jnp.log(2.0), -jnp.inf)

    # Cos prior (dec): log(cos(x)/2) on [-pi/2, pi/2]
    lp_cos = jnp.where(in_bounds, jnp.log(jnp.abs(jnp.cos(x)) + 1e-300) - jnp.log(2.0), -jnp.inf)

    # Beta(3,1) prior (d_L): log(3*u^2 / range) where u = (x-lo)/(hi-lo)
    u = (x - PRIOR_LO) / _PRIOR_RANGE
    lp_beta = jnp.where(in_bounds, 2.0 * jnp.log(jnp.abs(u) + 1e-300) - _PRIOR_LOG_RANGE - _BETA_LN, -jnp.inf)

    # Log-uniform (Jeffreys) prior (H_0): -log(log(hi/lo)) - log(x)
    lp_log = jnp.where(in_bounds, -_PRIOR_LOG_LOG_RATIO - jnp.log(jnp.abs(x) + 1e-300), -jnp.inf)

    #SFM gaussian prior
    lp_gaussian = jnp.where(in_bounds, -0.5 * ((x - GAUSS_MEAN) / GAUSS_SIGMA) ** 2 - jnp.log(GAUSS_SIGMA)- 0.5 * jnp.log(2.0 * jnp.pi), -jnp.inf)

    # Select per-parameter prior using type index
    lp = jnp.where(prior_type == 0, lp_uniform,
         jnp.where(prior_type == 1, lp_sin,
         jnp.where(prior_type == 2, lp_cos,
         jnp.where(prior_type == 3, lp_beta,
         jnp.where(prior_type == 4, lp_log,
                    lp_gaussian)))))


    total = jnp.sum(lp)

    # Component mass constraint: m1, m2 must be in [M_COMP_LO, M_COMP_HI]
    # From M_c, q: eta = q/(1+q)^2, M_total = M_c / eta^(3/5), m1 = M_total/(1+q), m2 = q*m1
    q = x[I_Q]
    eta = q / (1 + q) ** 2
    M_total = x[I_MC] / eta ** 0.6
    m1 = M_total / (1 + q)
    m2 = q * m1
    mass_ok = (m1 >= M_COMP_LO) & (m1 <= M_COMP_HI) & (m2 >= M_COMP_LO) & (m2 <= M_COMP_HI)
    total = jnp.where(mass_ok, total, -jnp.inf)

    # Flat-in-z prior (LVK convention): uniform in d_L at fixed H_0.
    # Since z ∝ d_L at fixed H_0, flat-in-d_L ≡ flat-in-z.
    # The base d_L prior is already uniform (type 0), so no Jacobian needed.
    # The H_0 prior remains log-uniform (unchanged).

    # Jacobian |∂(m1,m2)/∂(M_c,q)| = M_c * (1+q)^(2/5) / q^(6/5)
    # Converts uniform-in-(M_c,q) to uniform-in-(m1,m2), as assumed in
    # Abbott et al., PhysRevX 9, 011001, Sec. II.D (z=0.0099 for NGC 4993)
    log_jacobian = jnp.log(x[I_MC]) - 1.2 * jnp.log(x[I_Q]) + 0.4 * jnp.log(1.0 + x[I_Q])
    total = total + log_jacobian

    return total


# PSD estimation config (matching bilby/kazewong):
#   - 32s Tukey-windowed segments, 50% overlap, median averaging
PSD_FFT_LENGTH = 32  # seconds per FFT segment
PSD_OVERLAP_FRAC = 0.5
PSD_METHOD = 'median'


#checks successful detectors from metadata
successful_detectors = metadata["detectors"]

print("\n===== DETECTORS FROM METADATA =====")
print(successful_detectors)
print("===================================\n")


detector_getters = {
    "H1": get_H1,
    "L1": get_L1,
    "V1": get_V1,
}

detectors = []

for ifo_name in successful_detectors:

    # Get the appropriate detector object
    ifo = detector_getters[ifo_name]()

    t_det = time.time()

    print(f"\nLoading {ifo_name}...")


    strain_data = load_gwosc_local(
        ifo.name,
        start,
        end,
    )

    psd_ts = load_gwosc_local_gwpy(
        ifo.name,
        psd_start,
        psd_end,
    )

    t_fetch = time.time() - t_det


    strain_data.set_tukey_window(
        alpha=tukey_alpha
    )

    strain_data.fft()

    ifo.set_data(strain_data)


    t_psd0 = time.time()

    psd_alpha = 2 * roll_off / PSD_FFT_LENGTH

    psd_gwpy = psd_ts.psd(
        fftlength=PSD_FFT_LENGTH,
        overlap=PSD_FFT_LENGTH * PSD_OVERLAP_FRAC,
        window=('tukey', psd_alpha),
        method=PSD_METHOD,
    )

    psd_interp_fn = interp1d(
        psd_gwpy.frequencies.value,
        psd_gwpy.value,
        kind='linear',
        fill_value=(
            psd_gwpy.value[0],
            psd_gwpy.value[-1],
        ),
        bounds_error=False,
    )

    strain_freqs = np.array(
        strain_data.frequencies
    )

    psd_obj = PowerSpectrum(
        values=jnp.array(
            psd_interp_fn(strain_freqs)
        ),
        frequencies=jnp.array(
            strain_freqs
        ),
        name=ifo.name,
    )

    ifo.set_psd(psd_obj)

    t_psd = time.time() - t_psd0

    ifo.set_frequency_bounds(
        fmin,
        fmax,
    )

    detectors.append(ifo)

    print(
        f"{ifo.name}: "
        f"data={t_fetch:.1f}s, "
        f"PSD={t_psd:.1f}s, "
        f"total={time.time()-t_det:.1f}s"
    )


print("\n===== DETECTORS USED =====")
print([ifo.name for ifo in detectors])
print("===========================\n")

N_DET = len(detectors)



t_data = time.time() - t0
print(f"[TIMING] Data loading: {t_data:.1f}s")

waveform_tag = 'IMRPhenomD_NRTidalv2'
waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False, no_taper=False)
print(f"Waveform: {waveform_tag}")

frequencies = detectors[0].sliced_frequencies
epoch = duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad


# ============================================================================
# 4. LIKELIHOOD SETUP
# ============================================================================
from jimgw.core.single_event.likelihood import TransientLikelihoodFD

t_lik0 = time.time()

likelihood = TransientLikelihoodFD(
    detectors=detectors,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
    phase_marginalization=True,
)

t_lik = time.time() - t_lik0
print(f"[TIMING] Likelihood setup: {t_lik:.1f}s")


def loglikelihood_fn(x):
    params = {
        'M_c': x[I_MC],
        'q': x[I_Q],
        's1_z': x[I_S1Z],
        's2_z': x[I_S2Z],
        'iota': x[I_IOTA],
        'd_L': x[I_DL],
        't_c': x[I_TC],
        'psi': x[I_PSI],
        'ra': x[I_RA],
        'dec': x[I_DEC],
        'lambda_1': x[I_L1],
        'lambda_2': x[I_L2],
        'eta': x[I_Q] / (1 + x[I_Q]) ** 2,
        'phase_c': 0.0,
    }

    return likelihood.evaluate(params)


# ============================================================================
# 5. NESTED SAMPLING SETUP
# ============================================================================

num_live = args.n_live
num_delete = args.num_delete if args.num_delete is not None else int(num_live * 0.3)
num_mcmc_steps = int(args.n_mcmc) if args.n_mcmc is not None else int(NUM_DIMS * 8)
print(f"num_mcmc_steps (slice steps per update): {num_mcmc_steps}  "
      f"({'CLI override' if args.n_mcmc is not None else f'default 8*NUM_DIMS={8*NUM_DIMS}'})")



# ============================================================================
# 6. PRIOR SAMPLING & INITIALIZATION
# ============================================================================
def sample_from_prior(key, n, prior_type):
    """Draw n samples for all parameters. Returns (n, NUM_DIMS) array.

    Uses rejection sampling to enforce component mass constraint [M_COMP_LO, M_COMP_HI].
    Oversamples by 4x then filters, repeating until n valid samples are obtained.
    """
    collected = []
    remaining = n
    while remaining > 0:
        key, subkey = jax.random.split(key)
        n_try = remaining * 4  # oversample
        keys = jax.random.split(subkey, NUM_DIMS)
        batch = jnp.zeros((n_try, NUM_DIMS))

        for i in range(NUM_DIMS):
            lo, hi = float(PRIOR_LO[i]), float(PRIOR_HI[i])
            ptype = int(prior_type[i])
            if ptype == 0:    # uniform
                col = jax.random.uniform(keys[i], (n_try,), minval=lo, maxval=hi)
            elif ptype == 1:  # sin (iota): inverse CDF = arccos(1 - 2u)
                col = jnp.arccos(1 - 2 * jax.random.uniform(keys[i], (n_try,)))
            elif ptype == 2:  # cos (dec): inverse CDF = arcsin(2u - 1)
                col = jnp.arcsin(2 * jax.random.uniform(keys[i], (n_try,)) - 1)
            elif ptype == 4:  # log-uniform: x = lo * (hi/lo)^u
                col = lo * (hi / lo) ** jax.random.uniform(keys[i], (n_try,))
            elif ptype == 5:  # Gaussian RA/Dec
                col = GAUSS_MEAN[i] + GAUSS_SIGMA[i] * jax.random.normal(keys[i], (n_try,))
            batch = batch.at[:, i].set(col)

        # Filter by component mass constraint
        q = batch[:, I_Q]
        eta = q / (1 + q) ** 2
        M_total = batch[:, I_MC] / eta ** 0.6
        m1 = M_total / (1 + q)
        m2 = q * m1
        valid = jnp.all((batch >= PRIOR_LO) & (batch <= PRIOR_HI),axis=1) & (m1 >= M_COMP_LO) & (m1 <= M_COMP_HI) & (m2 >= M_COMP_LO) & (m2 <= M_COMP_HI)
        good = batch[valid]
        collected.append(np.array(good[:remaining]))
        remaining -= len(collected[-1])

    return jnp.array(np.concatenate(collected)[:n])


def make_logprior(prior_type):
    return lambda x: logprior_fn(x, prior_type)


total_init_time = 0.0
total_jit_time = 0.0
total_ns_time = 0.0

runs = [
    ("unconditioned", PRIOR_TYPE_U),
    ("conditioned", PRIOR_TYPE_C),
]
rng_key = jax.random.PRNGKey(args.seed)

logZ_unconditioned = None
logZ_conditioned = None

for run_name, prior_type in runs:

    print("\n" + "=" * 70)
    print(f"STARTING {run_name.upper()} RUN")
    print("=" * 70)

    logprior = make_logprior(prior_type)

    @jax.jit
    def stepper_fn(x, d, t):
        """Linear step with periodic wrapping for psi (pi), ra (2pi), phase_c (2pi),
        and (precessing branch) phi_1, phi_2 (2pi)."""
        y = x + t * d
        y = y.at[I_PSI].set(jnp.mod(y[I_PSI], jnp.pi))
        y = y.at[I_RA].set(jnp.mod(y[I_RA], 2 * jnp.pi))
        return y, True


    nested_sampler = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_mcmc_steps,
        stepper_fn=stepper_fn,
    )

        
    t_init0 = time.time()
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_prior(init_key, num_live, prior_type)


    state = nested_sampler.init(initial_particles)
    t_init = time.time() - t_init0
    print(f"[TIMING] Prior sampling + init: {t_init:.1f}s")


    # ============================================================================
    # 7. RUN NESTED SAMPLING
    # ============================================================================

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    print(f"Running nested sampling: {num_live} live, {NUM_DIMS}D")
    print("JIT-compiling first step...")
    t_jit0 = time.time()
    (state, rng_key), dead_first = one_step((state, rng_key), None)
    jax.block_until_ready(state)
    t_jit = time.time() - t_jit0
    print(f"[TIMING] JIT compilation (first step): {t_jit:.1f}s")

    ns_start = time.time()
    dead = [dead_first]
    dead_points = num_delete
    next_report = 500

    with tqdm.tqdm(desc="Dead points", initial=num_delete, unit=" dead points") as pbar:
        while not state.integrator.logZ_live - state.integrator.logZ < -3:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            dead_points += num_delete
            pbar.update(num_delete)

            if dead_points >= next_report:

                logZ = float(state.integrator.logZ)
                logZ_live = float(state.integrator.logZ_live)
                logZ_diff = logZ_live - logZ

                print(
                    f"\nDead points: {dead_points} | "
                    f"logZ = {logZ:.3f} | "
                    f"logZ_live = {logZ_live:.3f} | "
                    f"logZ_live - logZ = {logZ_diff:.3f}",
                    flush=True
                )

                next_report += 500

    ns_time = time.time() - ns_start


    # ============================================================================
    # 8. POST-PROCESSING & OUTPUT
    # ============================================================================

    # Combine dead + live points via blackjax utility.
    # finalise() concatenates all dead NSInfo objects with the final live particles,
    # returning a single NSInfo with .particles (StateWithLogLikelihood).

    result = finalise(state, dead, update_info=False)

    samples = NestedSamples(
        np.array(result.particles.position),
        logL=np.array(result.particles.loglikelihood),
        logL_birth=np.array(result.particles.loglikelihood_birth),
        columns=PARAM_NAMES,
        labels=PARAM_LABELS,
    )

    run_label = os.path.join(grb_dir,
            f"{grb_name}_{run_name}_Transient"
        )

    logzs = samples.logZ(100)

    logZ_mean = logzs.mean()
    logZ_std = logzs.std()

    print(
        f"{run_name}: "
        f"Log Evidence = {logZ_mean:.2f} +/- {logZ_std:.2f}"
    )

    if run_name == "unconditioned":
        logZ_unconditioned = logZ_mean
        logZ_unconditioned_err = logZ_std

    elif run_name == "conditioned":
        logZ_conditioned = logZ_mean
        logZ_conditioned_err = logZ_std

    samples.to_csv(f"{run_label}.csv")

    print(f"Saved to {run_label}.csv")

# Timing summary
t_total = time.time() - t0
print(f"\n{'='*50}")
print(f"TIMING SUMMARY")
print(f"{'='*50}")
print(f"  Data loading:     {t_data:7.1f}s")
print(f"  Likelihood setup: {t_lik:7.1f}s")
print(f"  Init + prior:     {t_init:7.1f}s")
print(f"  JIT compilation:  {t_jit:7.1f}s")
print(f"  Sampling:         {ns_time:7.1f}s")
print(f"  Total:            {t_total:7.1f}s")
print(f"{'='*50}")



print("\n===== EVIDENCE COMPARISON =====")
print(f"Unconditioned: {logZ_unconditioned:.2f} +/- {logZ_unconditioned_err:.2f}")
print(f"Conditioned:   {logZ_conditioned:.2f} +/- {logZ_conditioned_err:.2f}")

delta_logZ = logZ_conditioned - logZ_unconditioned
delta_logZ_err = np.sqrt(logZ_unconditioned_err**2 + logZ_conditioned_err**2)
print(f"Delta logZ:    {delta_logZ:.2f} +/- {delta_logZ_err:.2f}")