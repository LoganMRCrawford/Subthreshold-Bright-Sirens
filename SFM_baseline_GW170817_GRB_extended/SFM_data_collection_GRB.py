import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update('jax_enable_x64', True)

import pickle
import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.heasarc import Heasarc
from gwpy.timeseries import TimeSeries
from gwosc import datasets

from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

ifos = [get_H1(), get_L1(), get_V1()]

# ============================================================
# SETTINGS
# ============================================================

# Set to [] if you only want to make the catalogue and calculate the times, without downloading strain.
# ------------------------------------------------------------
SELECTED_GRBS = ["GRB190728271", "GRB190724031"]
catalogue_read = False

OUTPUT_DIR = "data/GRB_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
catalogue_file = os.path.join(OUTPUT_DIR,"grb_catalog.csv")

#settings for data collection
duration = 128
post_trigger_duration = 0
psd_pad = 16
psd_duration = 1024

#recreates GRB catalogue
if catalogue_read == True:

    observing_runs = {
        "O1": datasets.run_segment("O1"),
        "O2": datasets.run_segment("O2"),
        "O3a": datasets.run_segment("O3a"),
        "O3b": datasets.run_segment("O3b"),
        "O4a": datasets.run_segment("O4a"),
        "O4b": datasets.run_segment("O4b"),
    }


    def find_run(gps):
        """
        Return the LIGO observing run containing the supplied GPS time.
        """

        for run, (start, end) in observing_runs.items():

            if start <= gps <= end:
                return run

        return None

    def process_grb_table(table):
        """
        Convert the HEASARC Fermi GBM catalogue into a pandas DataFrame.

        Returns:
            GRB      : GRB name
            GPS      : trigger time in GPS seconds
            RA_rad   : right ascension in radians
            DEC_rad  : declination in radians
            T90_s    : T90 duration in seconds
        """

        results = []

        for row in table:

            # HEASARC trigger_time is MJD
            gps_time = Time(
                row["trigger_time"],
                format="mjd",
                scale="utc",
            ).gps

            # HEASARC RA/DEC are degrees
            skycoord = SkyCoord(
                ra=row["ra"],
                dec=row["dec"],
                unit="deg",
            )

            results.append({
                "GRB": row["name"],
                "GPS": gps_time,
                "RA_rad": skycoord.ra.radian,
                "DEC_rad": skycoord.dec.radian,
                "T90_s": row["t90"],
            })

        return pd.DataFrame(results)




    # ============================================================
    # QUERY HEASARC
    # ============================================================

    print("=" * 70)
    print("QUERYING HEASARC FERMI GBM CATALOGUE")
    print("=" * 70)

    h = Heasarc()

    #can change catalogue requirements
    table = h.query_tap("""
    SELECT
        name,
        trigger_time,
        ra,
        dec,
        t90,
        error_radius
    FROM fermigbrst
    WHERE t90 < 3
      AND error_radius < 10
    """)


    grb_df = process_grb_table(table)

    print(f"Found {len(grb_df)} short GRBs.")

    grb_df["LIGO_run"] = grb_df["GPS"].apply(find_run)


    # Only retain GRBs occurring during an observing run
    candidate_grbs = grb_df.dropna(
        subset=["LIGO_run"]
    ).copy()

    candidate_grbs = candidate_grbs.reset_index(drop=True)

    candidate_grbs.index.name = "index"


    print()
    print("=" * 70)
    print("GRBs WITHIN LIGO OBSERVING RUNS")
    print("=" * 70)

    print(
        candidate_grbs[
            [
                "GRB",
                "GPS",
                "RA_rad",
                "DEC_rad",
                "T90_s",
                "LIGO_run",
            ]
        ].to_string()
    )

    print()
    print(f"Number of candidates: {len(candidate_grbs)}")

    candidate_grbs.to_csv(
        catalogue_file, index=False
    )

    print()
    print(f"Saved candidate catalogue to:")
    print(f"  {catalogue_file}")
else:
    candidate_grbs = pd.read_csv(catalogue_file)










# ============================================================
# DOWNLOAD / SAVE SELECTED GRBs
# ============================================================


print()
print("=" * 70)
print("DOWNLOADING SELECTED GRBs")
print("=" * 70)

print(f"Selected GRBs: {SELECTED_GRBS}")


for grb_name in SELECTED_GRBS:

    # --------------------------------------------------------
    # Find GRB in catalogue by name
    # --------------------------------------------------------

    matches = candidate_grbs[
        candidate_grbs["GRB"] == grb_name
    ]

    if len(matches) == 0:
        print(
            f"WARNING: GRB {grb_name} does not exist "
            f"in the candidate catalogue. Skipping."
        )
        continue

    if len(matches) > 1:
        print(
            f"WARNING: Multiple entries found for {grb_name}. "
            f"Using the first."
        )

    row = matches.iloc[0]

    # --------------------------------------------------------
    # GRB information
    # --------------------------------------------------------

    gps = float(row["GPS"])

    start = gps - (duration - post_trigger_duration)
    end = gps + post_trigger_duration

    psd_start = start - psd_pad - psd_duration
    psd_end = start - psd_pad

    ligo_run = row["LIGO_run"]

    ra = float(row["RA_rad"])
    dec = float(row["DEC_rad"])

    print()
    print("-" * 70)
    print(f"GRB: {grb_name}")
    print("-" * 70)

    print(f"GPS:       {gps}")
    print(f"Run:       {ligo_run}")
    print(f"RA:        {ra}")
    print(f"Dec:       {dec}")

    print()
    print("Analysis:")
    print(f"  start = {start}")
    print(f"  end   = {end}")

    print()
    print("PSD:")
    print(f"  start = {psd_start}")
    print(f"  end   = {psd_end}")

    # --------------------------------------------------------
    # GRB output directory
    # --------------------------------------------------------

    grb_dir = os.path.join(
        OUTPUT_DIR,
        grb_name,
    )

    os.makedirs(
        grb_dir,
        exist_ok=True,
    )

    print()
    print(f"GRB directory:")
    print(f"  {grb_dir}")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "GRB": grb_name,
        "GPS": gps,
        "RA_rad": ra,
        "DEC_rad": dec,
        "T90_s": float(row["T90_s"]),
        "LIGO_run": ligo_run,

        "duration": duration,
        "post_trigger_duration": post_trigger_duration,

        "start": start,
        "end": end,

        "psd_pad": psd_pad,
        "psd_duration": psd_duration,

        "psd_start": psd_start,
        "psd_end": psd_end,
    }

    # --------------------------------------------------------
    # Detector data
    # --------------------------------------------------------

    detector_data = {}

    for ifo in ifos:

        print()
        print(f"Downloading {ifo.name}...")

        ts = TimeSeries.fetch_open_data(
            ifo.name,
            psd_start,
            end,
            version=2,
        )

        strain_file = os.path.join(
            grb_dir,
            f"{grb_name}_{ifo.name}.hdf5",
        )

        ts.write(
            strain_file,
            format="hdf5",
            overwrite=True,
        )

        print(
            f"Saved {ifo.name}: {strain_file}"
        )

        detector_data[ifo.name] = strain_file

    # --------------------------------------------------------
    # Save detector file paths in metadata
    # --------------------------------------------------------

    metadata["detector_files"] = detector_data

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_file = os.path.join(
        grb_dir,
        f"{grb_name}_metadata.pkl",
    )

    with open(
        metadata_file,
        "wb",
    ) as f:
        pickle.dump(
            metadata,
            f,
        )

    print()
    print(
        f"Saved metadata: {metadata_file}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("Candidate catalogue:")
print(f"  {catalogue_file}")

print()
print("Selected GRB data were saved under:")
print(f"  {OUTPUT_DIR}")