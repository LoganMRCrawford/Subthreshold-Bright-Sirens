"""Released GRB sky posteriors represented as priors for sequential GW inference."""

import hashlib
import json
from pathlib import Path

import astropy.units as units
import jax
import jax.numpy as jnp
import jax_healpy
import numpy as np
from astropy.coordinates import FK5, SkyCoord
from astropy.time import Time

from grb_localization import HealpixLocalization


class SequentialSkyPrior:
    def __init__(self, package_path, event):
        if not jax.config.x64_enabled:
            raise ValueError("Enable JAX x64 before constructing the sky prior")
        self.path = Path(package_path).resolve()
        self.record = json.loads(self.path.with_name("grb_inputs.json").read_text())
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.record["event"] != event or not self.record["probability_usable"]:
            raise ValueError("The probability package is not approved for this event")
        if self.sha256 != self.record["artifact_sha256"]:
            raise ValueError("Probability-package checksum does not match its provenance")
        if self.record["timing"]["tc_minus_trigger_support_s"] != [-10.0, 0.0]:
            raise ValueError("The package timing support differs from this sampler")
        with np.load(self.path, allow_pickle=False) as package:
            probability = np.asarray(package["probability_per_native_pixel"], dtype=float)
            nside = int(package["nside"])
            order = str(package["ordering"])
            frame = str(package["native_frame"])
            equinox = float(package["equinox"])
            self.trigger_gps = float(package["gamma_trigger_gps"])
        if nside <= 0 or nside & (nside - 1) or probability.shape != (12 * nside ** 2,):
            raise ValueError("Invalid full-sky HEALPix dimensions")
        if order not in {"nested", "ring"} or frame not in {"fk5", "icrs"}:
            raise ValueError("Unsupported HEALPix ordering or native frame")
        if not np.isfinite(equinox) or not np.isfinite(self.trigger_gps):
            raise ValueError("Invalid frame equinox or trigger GPS")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0):
            raise ValueError("Probability pixels must be finite and nonnegative")
        total = probability.sum()
        if not np.isclose(total, 1, rtol=1e-6, atol=1e-10):
            raise ValueError("Sky-pixel probabilities do not sum to one")
        probability = probability / total
        self.localization = HealpixLocalization(
            path=self.path, nside=nside, order=order, probability=probability,
            native_frame=frame, equinox=equinox,
        )
        rotation = np.eye(3)
        if frame == "fk5":
            basis = SkyCoord(
                x=rotation[0] * units.one, y=rotation[1] * units.one,
                z=rotation[2] * units.one, representation_type="cartesian", frame="icrs",
            )
            rotation = basis.transform_to(FK5(equinox=Time(equinox, format="jyear"))).cartesian.xyz.value
        self.rotation = jnp.asarray(rotation)
        self.probability = jnp.asarray(probability)
        self.nside = nside
        self.nested = order == "nested"
        self.log_pixel_area = jnp.log(self.localization.pixel_area_sr)

    def pixel_indices(self, ra, dec):
        vector = jnp.stack((jnp.cos(dec) * jnp.cos(ra), jnp.cos(dec) * jnp.sin(ra), jnp.sin(dec)))
        native = jnp.einsum("ij,j...->i...", self.rotation, vector)
        theta = jnp.arccos(jnp.clip(native[2], -1, 1))
        phi = jnp.mod(jnp.arctan2(native[1], native[0]), 2 * jnp.pi)
        return jax_healpy.ang2pix(self.nside, theta, phi, nest=self.nested)

    def log_density_sr(self, ra, dec):
        return jnp.log(self.probability[self.pixel_indices(ra, dec)]) - self.log_pixel_area

    def log_prob(self, ra, dec):
        supported = (ra >= 0) & (ra < 2 * jnp.pi) & (dec > -jnp.pi / 2) & (dec < jnp.pi / 2)
        value = self.log_density_sr(ra, dec) + jnp.log(jnp.cos(dec))
        return jnp.where(supported, value, -jnp.inf)

    def sample(self, key, size):
        seed = int(jax.random.bits(key, shape=(), dtype=jnp.uint32))
        return jnp.asarray(self.localization.sample_sky(size, rng=seed))