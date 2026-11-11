"""Independent GW nested sampling with a released GRB posterior as sky prior."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import signal
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import blackjax
import jax.numpy as jnp
import numpy as np
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise
from gwpy.timeseries import TimeSeries
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from scipy.integrate import quad

from sequential_sky_prior import SequentialSkyPrior


ROOT = Path(__file__).resolve().parent
PARAM_NAMES = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "lambda_1", "lambda_2"]
LOWER = np.array([1., .125, -.05, -.05, 0., 20., -10., 0., 0., -np.pi / 2, 0., 0.])
UPPER = np.array([2., 1., .05, .05, np.pi, 800., 0., np.pi, 2 * np.pi, np.pi / 2, 5000., 5000.])
MASS_LOWER = .5
MASS_UPPER = 7.7
DETECTORS = {"H1": get_H1, "L1": get_L1, "V1": get_V1}


def mass_integrand(ratio):
    factor = (1 + ratio) ** .2 / ratio ** .6
    lower = max(LOWER[0], MASS_LOWER / (ratio * factor))
    upper = min(UPPER[0], MASS_UPPER / factor)
    return max(upper ** 2 - lower ** 2, 0) * factor ** 2 / 2


class SequentialPrior:
    def __init__(self, sky):
        self.sky = sky
        self.mass_normalization = quad(mass_integrand, LOWER[1], UPPER[1], epsabs=1e-10)[0]
        self.uniform_log_density = -np.log((UPPER - LOWER)[[2, 3, 5, 6, 7, 10, 11]]).sum()

    def log_prob(self, position, mode):
        chirp, ratio = position[0], position[1]
        primary = chirp * (1 + ratio) ** .2 / ratio ** .6
        secondary = ratio * primary
        supported = jnp.all((position >= jnp.asarray(LOWER)) & (position <= jnp.asarray(UPPER)))
        supported &= (primary >= MASS_LOWER) & (primary <= MASS_UPPER)
        supported &= (secondary >= MASS_LOWER) & (secondary <= MASS_UPPER)
        supported &= (position[4] > 0) & (position[4] < jnp.pi)
        supported &= (position[8] < 2 * jnp.pi) & (jnp.abs(position[9]) < jnp.pi / 2)
        mass = jnp.log(chirp) + .4 * jnp.log1p(ratio) - 1.2 * jnp.log(ratio) - np.log(self.mass_normalization)
        inclination = jnp.log(jnp.sin(position[4])) - jnp.log(2.)
        if mode == "grb_map":
            sky = self.sky.log_prob(position[8], position[9])
        elif mode == "whole_sky":
            sky = jnp.log(jnp.cos(position[9])) - jnp.log(4 * jnp.pi)
        else:
            raise ValueError(f"Unknown prior mode: {mode}")
        return jnp.where(supported, self.uniform_log_density + mass + inclination + sky, -jnp.inf)

    def sample(self, key, size, mode):
        rng = np.random.default_rng(int(jax.random.bits(key, shape=(), dtype=jnp.uint32)))
        collected = []
        remaining = size
        maximum_jacobian = UPPER[0] * (1 + LOWER[1]) ** .4 / LOWER[1] ** 1.2
        while remaining:
            batch = rng.uniform(LOWER, UPPER, size=(max(1024, remaining * 8), len(PARAM_NAMES)))
            batch[:, 4] = np.arccos(1 - 2 * rng.random(len(batch)))
            if mode == "grb_map":
                batch[:, [8, 9]] = self.sky.localization.sample_sky(len(batch), rng=rng)
            elif mode == "whole_sky":
                batch[:, 9] = np.arcsin(2 * rng.random(len(batch)) - 1)
            else:
                raise ValueError(f"Unknown prior mode: {mode}")
            chirp, ratio = batch[:, 0], batch[:, 1]
            primary = chirp * (1 + ratio) ** .2 / ratio ** .6
            secondary = ratio * primary
            jacobian = chirp * (1 + ratio) ** .4 / ratio ** 1.2
            valid = (primary >= MASS_LOWER) & (primary <= MASS_UPPER)
            valid &= (secondary >= MASS_LOWER) & (secondary <= MASS_UPPER)
            valid &= rng.random(len(batch)) < jacobian / maximum_jacobian
            accepted = batch[valid][:remaining]
            collected.append(accepted)
            remaining -= len(accepted)
        return jnp.asarray(np.concatenate(collected))


def load_raw_psd(directory, detector, frequencies):
    path = directory / f"glitch_median_PSD_{detector}.dat"
    raw = np.loadtxt(path, usecols=(0, 1))
    valid = np.all(np.isfinite(raw), axis=1) & (raw[:, 1] > 0)
    raw = raw[valid]
    native, inverse, counts = np.unique(raw[:, 0], return_inverse=True, return_counts=True)
    if len(native) < 2:
        raise ValueError(f"No valid PSD grid: {path}")
    values = np.bincount(inverse, weights=raw[:, 1] * 2 / 128) / counts
    active = frequencies[(frequencies >= 23) & (frequencies <= 2048)]
    if active[0] < native[0] or active[-1] > native[-1] + 2 * np.median(np.diff(native)):
        raise ValueError(f"PSD does not cover 23-2048 Hz: {path}")
    return np.interp(frequencies, native, values), path


def build_likelihood(args, metadata):
    detectors = []
    provenance = {}
    for name in metadata["detectors"]:
        if name not in DETECTORS:
            raise ValueError(f"Unsupported detector {name}")
        path = args.data_dir / args.event / f"{args.event}_{name}.hdf5"
        series = TimeSeries.read(path, format="hdf5")
        analysis = series.crop(float(metadata["start"]), float(metadata["end"]))
        if len(analysis) != round(128 / analysis.dt.value) or not np.all(np.isfinite(analysis.value)):
            raise ValueError(f"Invalid 128-second signal data: {path}")
        data = Data(analysis.value, analysis.dt.value, analysis.epoch.value, name)
        data.set_tukey_window(alpha=2 * .4 / 128)
        data.fft()
        frequencies = np.asarray(data.frequencies)
        context = series.crop(float(metadata["psd_start"]), float(metadata["psd_end"]))
        if len(context) != round(1024 / context.dt.value) or not np.all(np.isfinite(context.value)):
            raise ValueError(f"Invalid off-source PSD context: {path}")
        if args.psd_source == "welch":
            estimate = context.psd(fftlength=32, overlap=16, window=("tukey", .025), method="median")
            psd = np.interp(frequencies, estimate.frequencies.value, estimate.value)
            psd_path = None
        else:
            psd, psd_path = load_raw_psd(args.psd_dir, name, frequencies)
        active = (frequencies >= 23) & (frequencies <= 2048)
        if not np.all(np.isfinite(psd[active])) or np.any(psd[active] <= 0):
            raise ValueError(f"Invalid likelihood PSD for {name}")
        detector = DETECTORS[name]()
        detector.set_data(data)
        detector.set_psd(PowerSpectrum(values=jnp.asarray(psd), frequencies=jnp.asarray(frequencies), name=name))
        detector.set_frequency_bounds(23., 2048.)
        detectors.append(detector)
        with path.open("rb") as stream:
            data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance[name] = {"strain_path": str(path), "strain_sha256": data_hash,
                            "psd_path": str(psd_path) if psd_path else None,
                            "psd_sha256": hashlib.sha256(psd_path.read_bytes()).hexdigest() if psd_path else None}
    likelihood = TransientLikelihoodFD(
        detectors=detectors, waveform=RippleIMRPhenomD_NRTidalv2(f_ref=20., use_lambda_tildes=False, no_taper=False),
        trigger_time=float(metadata["GPS"]), f_min=23., f_max=2048.,
        phase_marginalization=True, include_noise_normalization=False,
    )

    @jax.jit
    def evaluate(position):
        parameters = dict(zip(PARAM_NAMES, position))
        parameters["eta"] = position[1] / (1 + position[1]) ** 2
        parameters["phase_c"] = 0.
        return likelihood.evaluate(parameters)

    return evaluate, float(likelihood.noise_normalization), provenance


def checkpoint_identity(args, output_dir):
    metadata = output_dir / "run_metadata.json"
    report = json.loads(metadata.read_text()) if metadata.exists() else {}
    fields = ("event", "prior", "psd_source", "sky_package_sha256", "data", "metadata_sha256", "versions")
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run": {field: report[field] for field in fields if field in report},
        "prior": args.prior, "n_live": args.n_live, "n_mcmc": args.n_mcmc,
        "num_delete": args.num_delete, "seed": args.seed,
    }


def save_checkpoint(output_dir, identity, state, key, dead, iteration, seconds):
    payload = {"identity": identity, "state": jax.device_get(state), "key": jax.device_get(key),
               "dead": jax.device_get(dead), "iteration": iteration, "sampling_seconds": seconds}
    temporary = output_dir / ".checkpoint.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output_dir / "checkpoint.pkl")


def run_nested(args, prior, likelihood, output_dir):
    logprior = jax.jit(lambda position: prior.log_prob(position, args.prior))

    @jax.jit
    def stepper(position, direction, distance):
        proposed = position + distance * direction
        proposed = proposed.at[7].set(jnp.mod(proposed[7], jnp.pi))
        proposed = proposed.at[8].set(jnp.mod(proposed[8], 2 * jnp.pi))
        return proposed, True

    sampler = blackjax.nss(
        logprior_fn=logprior, loglikelihood_fn=likelihood,
        num_delete=args.num_delete, num_inner_steps=args.n_mcmc, stepper_fn=stepper,
    )
    identity = checkpoint_identity(args, output_dir)
    elapsed = 0.
    if args.resume:
        with (output_dir / "checkpoint.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint source, inputs or sampler configuration differ from this run")
        state = jax.tree.map(jnp.asarray, saved["state"])
        key = jnp.asarray(saved["key"])
        dead = jax.tree.map(jnp.asarray, saved["dead"])
        iteration = saved["iteration"]
        elapsed = saved["sampling_seconds"]
        print(f"Resuming trusted local checkpoint at iteration={iteration}", flush=True)
    else:
        key, initial_key = jax.random.split(jax.random.PRNGKey(args.seed))
        particles = prior.sample(initial_key, args.n_live, args.prior)
        state = sampler.init(particles)
        jax.block_until_ready(state)
        dead = []
        iteration = 0

    @jax.jit
    def step(current, random_key):
        return sampler.step(random_key, current)

    started = time.monotonic()
    remaining = float(state.integrator.logZ_live - state.integrator.logZ) if iteration else np.inf
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        while remaining >= -3:
            if stop_requested:
                save_checkpoint(output_dir, identity, state, key, dead, iteration, elapsed + time.monotonic() - started)
                print(f"Checkpointed before termination at iteration={iteration}; resume is required", flush=True)
                raise SystemExit(75)
            key, step_key = jax.random.split(key)
            state, discarded = step(state, step_key)
            jax.block_until_ready(state)
            dead.append(discarded)
            iteration += 1
            remaining = float(state.integrator.logZ_live - state.integrator.logZ)
            if iteration == 1 or iteration % 10 == 0:
                print(f"iteration={iteration} logB_CBC_noise={float(state.integrator.logZ):.6f} remaining_logZ_gap={remaining:.6f} sampling_seconds={elapsed + time.monotonic() - started:.1f}", flush=True)
            if not np.isfinite(remaining):
                raise RuntimeError("Nonfinite nested-sampling stopping statistic")
            if iteration % args.checkpoint_every == 0 or remaining < -3 or stop_requested:
                save_checkpoint(output_dir, identity, state, key, dead, iteration, elapsed + time.monotonic() - started)
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    result = finalise(state, dead, update_info=False)
    samples = NestedSamples(
        np.asarray(result.particles.position), logL=np.asarray(result.particles.loglikelihood),
        logL_birth=np.asarray(result.particles.loglikelihood_birth), columns=PARAM_NAMES,
    )
    samples.to_csv(output_dir / "posterior.csv")
    log_evidences = np.asarray(samples.logZ(100))
    return {"log_bayes_factor_cbc_noise": float(log_evidences.mean()),
            "log_bayes_factor_std": float(log_evidences.std()), "iterations": iteration,
            "remaining_logZ_gap": remaining, "sampling_seconds": elapsed + time.monotonic() - started,
            "posterior_rows": len(samples)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--sky-package", required=True, type=Path)
    parser.add_argument("--prior", required=True, choices=("whole_sky", "grb_map"))
    parser.add_argument("--psd-source", required=True, choices=("welch", "bayesline"))
    parser.add_argument("--psd-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "SFM_baseline_GW170817_GRB_extended/data")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_sequential")
    parser.add_argument("--n-live", type=int, default=1000)
    parser.add_argument("--n-mcmc", type=int, default=240)
    parser.add_argument("--num-delete", type=int)
    parser.add_argument("--seed", type=int, default=170817)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume only a trusted local checkpoint with matching source and inputs")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.num_delete = args.num_delete if args.num_delete is not None else int(args.n_live * .3)
    if not 0 < args.num_delete < args.n_live or args.n_live < 4 or args.n_mcmc < 1:
        parser.error("Invalid live/deleted/MCMC counts")
    if args.checkpoint_every < 1 or args.resume and args.preflight:
        parser.error("Checkpoint interval must be positive; resume cannot be combined with preflight")
    if args.psd_source == "bayesline" and args.psd_dir is None:
        parser.error("BayesLine requires --psd-dir")
    args.output_dir = args.output_dir.resolve()
    args.output_root = args.output_root.resolve()
    if not args.output_dir.is_relative_to(args.output_root) or args.output_dir == args.output_root:
        parser.error("Output must be in a new subdirectory of the dedicated sequential output root")
    if args.resume:
        if not (args.output_dir / "checkpoint.pkl").is_file() or not (args.output_dir / "run_metadata.json").is_file():
            parser.error("Resume requires an existing checkpoint and run metadata")
        if (args.output_dir / "evidence_summary.json").exists():
            parser.error("Run already has a final evidence summary; refusing to overwrite it")
    elif args.output_dir.exists():
        parser.error("Output directory already exists; refusing to overwrite an existing run")
    args.data_dir = args.data_dir.resolve()
    return args


def main():
    args = parse_args()
    sky = SequentialSkyPrior(args.sky_package, args.event)
    metadata_path = args.data_dir / args.event / f"{args.event}_metadata.pkl"
    metadata = pickle.loads(metadata_path.read_bytes())
    if abs(float(metadata["GPS"]) - sky.trigger_gps) > 1e-3:
        raise ValueError("GRB sky and GW metadata trigger times differ")
    if not np.isclose(float(metadata["end"]) - float(metadata["start"]), 128):
        raise ValueError("Signal interval must be 128 seconds")
    if not np.isclose(float(metadata["psd_end"]) - float(metadata["psd_start"]), 1024):
        raise ValueError("PSD context must be 1024 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    prior = SequentialPrior(sky)
    likelihood, noise, data_provenance = build_likelihood(args, metadata)
    reference = jnp.array([1.1977, .85, 0., 0., .4, 40., -1.7, 1., 3.446, -.408, 400., 400.])
    log_ratio = float(likelihood(reference))
    if not np.isfinite(log_ratio):
        raise ValueError("Nonfinite JIT likelihood at the reference parameters")
    print(f"JAX devices: {jax.devices()}; x64={jax.config.x64_enabled}; reference log likelihood ratio={log_ratio}; noise log density={noise}", flush=True)
    report = {
        "event": args.event, "prior": args.prior, "psd_source": args.psd_source,
        "factorization": "p(theta|D_GRB,D_GW) proportional to L_GW(theta)*p_GRB(Omega|D_GRB)*pi(other); map used once as prior, not an additional GRB likelihood",
        "conditional_evidence": "Z_GW_given_GRB for grb_map; Z_GW_given_whole_sky for whole_sky; not joint GRB+GW evidence",
        "time_prior": {"tc_minus_trigger_support_s": [-10, 0], "density_per_s": .1, "status": "Assumed independent uniform delay; not a measured timing posterior"},
        "sky_package": str(sky.path), "sky_package_sha256": sky.sha256,
        "sky_provenance": sky.record, "data": data_provenance,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "detectors": metadata["detectors"], "noise_log_density": noise,
        "signal_window_gps": [float(metadata["start"]), float(metadata["end"])],
        "psd_context_gps": [float(metadata["psd_start"]), float(metadata["psd_end"])],
        "phase_marginalization": True, "mass_prior_normalization": prior.mass_normalization,
        "mass_initialization": "Jacobian-weighted rejection matching the normalized component-mass prior; legacy initializer unchanged",
        "likelihood": "CBC/noise likelihood ratio for numerical stability; full log evidence adds noise_log_density",
        "noise_configuration": "Legacy fixed median PSDs; Welch32s/Tukey0.025/50%; raw BayesLine factor2/128. No leakage/glitch-model correction is claimed.",
        "jax_devices": [str(device) for device in jax.devices()], "jax_x64": jax.config.x64_enabled,
        "reference_log_likelihood_ratio": log_ratio, "n_live": args.n_live,
        "n_mcmc": args.n_mcmc, "num_delete": args.num_delete, "seed": args.seed,
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jax-healpy", "blackjax", "jimgw", "anesthetic")},
        "preflight_only": args.preflight,
    }
    if args.resume:
        previous = json.loads((args.output_dir / "run_metadata.json").read_text())
        fields = ("event", "prior", "psd_source", "sky_package_sha256", "data", "metadata_sha256", "versions", "n_live", "n_mcmc", "num_delete", "seed")
        if any(previous.get(field) != report.get(field) for field in fields):
            raise ValueError("Resume inputs or sampler settings differ from the original run")
    else:
        (args.output_dir / "run_metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.preflight:
        print("PASS: full data/PSD load and JIT waveform/likelihood preflight", flush=True)
        return
    summary = run_nested(args, prior, likelihood, args.output_dir)
    summary.update(prior=args.prior, psd_source=args.psd_source,
                   noise_log_density=noise, full_log_evidence=noise + summary["log_bayes_factor_cbc_noise"])
    (args.output_dir / "evidence_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()