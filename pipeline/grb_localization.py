"""Validated access to normalized GRB HEALPix localization maps."""

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import FK5, ICRS, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy_healpix import HEALPix


@dataclass(frozen=True)
class HealpixLocalization:
    """A normalized probability map on a celestial HEALPix grid."""

    path: Path
    nside: int
    order: str
    probability: np.ndarray
    native_frame: str = "icrs"
    equinox: float = 2000.0

    @property
    def healpix(self):
        frame = FK5(equinox=Time(self.equinox, format="jyear")) if self.native_frame == "fk5" else ICRS()
        return HEALPix(nside=self.nside, order=self.order, frame=frame)

    @property
    def pixel_area_sr(self):
        return 4.0 * np.pi / self.probability.size

    @property
    def probability_sum(self):
        return float(self.probability.sum())

    @classmethod
    def from_fits(cls, path):
        """Load a GBM implicit HEALPix map with a PROBABILITY column."""
        path = Path(path)
        with fits.open(path, memmap=False) as hdul:
            try:
                extension = hdul["HEALPIX"]
            except KeyError as error:
                raise ValueError(f"{path} has no HEALPIX extension") from error

            header = extension.header
            if header.get("INDXSCHM") != "IMPLICIT":
                raise ValueError(f"{path} must use implicit HEALPix indexing")
            if header.get("COORDSYS") != "C":
                raise ValueError(f"{path} must use celestial (COORDSYS='C') coordinates")

            ordering = header.get("ORDERING", "").strip().lower()
            if ordering not in {"nested", "ring"}:
                raise ValueError(f"{path} has unsupported ORDERING={ordering!r}")
            nside = int(header["NSIDE"])
            probability = np.asarray(extension.data["PROBABILITY"], dtype=float).reshape(-1)
            primary = hdul[0].header
            native_frame = header.get("RADESYS", header.get("RADECSYS",
                           primary.get("RADESYS", primary.get("RADECSYS", "")))).strip().lower()
            equinox = float(header.get("EQUINOX", primary.get("EQUINOX", 2000.0)))
            if native_frame not in {"fk5", "icrs"}:
                raise ValueError(f"{path} has unsupported or unspecified celestial frame {native_frame!r}")
            if not np.isfinite(equinox):
                raise ValueError(f"{path} has an invalid equinox")

        expected_pixels = 12 * nside**2
        if probability.size != expected_pixels:
            raise ValueError(
                f"{path} has {probability.size} probability values; expected {expected_pixels}"
            )
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError(f"{path} has invalid probability values")

        probability_sum = probability.sum()
        if not np.isclose(probability_sum, 1.0, rtol=1e-6, atol=1e-10):
            raise ValueError(f"{path} probability sum is {probability_sum}, not one")
        probability /= probability_sum
        return cls(path=path, nside=nside, order=ordering, probability=probability,
                   native_frame=native_frame, equinox=equinox)

    def _pixel_indices(self, ra_rad, dec_rad):
        sky = SkyCoord(np.asarray(ra_rad) * u.rad, np.asarray(dec_rad) * u.rad, frame="icrs")
        return self.healpix.skycoord_to_healpix(sky)

    def pixel_centres(self):
        """Return ICRS pixel centres as an (N, 2) array of RA/Dec radians."""
        sky = self.healpix.healpix_to_skycoord(np.arange(self.probability.size)).icrs
        return np.column_stack([sky.ra.rad, sky.dec.rad])

    def credible_region(self, level):
        """Select a highest-density pixel set enclosing at least level probability."""
        if not np.isfinite(level) or not 0 < level <= 1:
            raise ValueError("Credible level must lie in (0, 1]")
        ordering = np.argsort(-self.probability, kind="stable")
        count = min(int(np.searchsorted(np.cumsum(self.probability[ordering]), level)) + 1,
                    len(ordering))
        mask = np.zeros(self.probability.size, dtype=bool)
        mask[ordering[:count]] = True
        return mask

    def sample_sky(self, size, rng=None):
        """Draw continuous ICRS RA/Dec radians, uniformly in area within each pixel."""
        if not isinstance(size, (int, np.integer)) or size < 0:
            raise ValueError("Sample size must be a nonnegative integer")
        rng = np.random.default_rng(rng)
        indices = rng.choice(self.probability.size, size=size, p=self.probability)
        sky = self.healpix.healpix_to_skycoord(indices, dx=rng.random(size), dy=rng.random(size)).icrs
        return np.column_stack([sky.ra.rad, sky.dec.rad])

    def density(self, ra_rad, dec_rad):
        """Return the piecewise-constant sky density in inverse steradians."""
        return self.probability[self._pixel_indices(ra_rad, dec_rad)] / self.pixel_area_sr

    def log_density(self, ra_rad, dec_rad):
        """Return the log sky density, with zero-probability pixels at -inf."""
        density = self.density(ra_rad, dec_rad)
        with np.errstate(divide="ignore"):
            return np.log(density)