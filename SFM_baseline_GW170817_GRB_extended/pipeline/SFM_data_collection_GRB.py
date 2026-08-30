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




####################
##INITIAL SETTINGS##
####################

#SELECTED_GRBS = []  #leave empty to process all candidate GRBs
SELECTED_GRBS = ["GRB170816599"]
catalogue_read = True   #make false to use current catalogue


#SAVING SETTINGS
OUTPUT_DIR = "../data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
catalogue_file = os.path.join(OUTPUT_DIR,"grb_catalog.csv")

#DATA COLLECTION SETTINGS
duration = 128
post_trigger_duration = 0
psd_pad = 16
psd_duration = 1024





##########################
##CREATING GRB CATALOGUE##
##########################

if catalogue_read == True:
    #Ligo observing runs
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

    #converts into correct units
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

    #queries the GRB catalogue
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

    #prints catalogue
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

    #saves catalogue to csv
    candidate_grbs.to_csv(
        catalogue_file, index=False,

    )
    print()
    print(f"Saved candidate catalogue to:")
    print(f"  {catalogue_file}")
else:
    candidate_grbs = pd.read_csv(catalogue_file)







##################################
##DATA READ IN FOR SELECTED GRBS##
##################################


print()
print("=" * 70)
print("DOWNLOADING SELECTED GRBs")
print("=" * 70)

if len(SELECTED_GRBS) == 0:
    grbs_to_process = candidate_grbs["GRB"].tolist()
    print(f"Processing ALL {len(grbs_to_process)} candidate GRBs")
else:
    grbs_to_process = SELECTED_GRBS
    print(f"Processing selected GRBs: {grbs_to_process}")


for grb_name in grbs_to_process:
    #finds GRB name in catalogue
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

    #collecting metadata
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

    #GRB specific output directory
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

    #adding GRB and strain segment info to metadata
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

    #reading in detector data
    detector_data = {}
    for ifo in ifos:
        print()
        print(f"Downloading {ifo.name}...")
        try:
            ts = TimeSeries.fetch_open_data(
                ifo.name,
                psd_start,
                end,
                version=2,
            )
            strain = np.asarray(ts.value)

            #check if data is finite
            data_finite = np.all(np.isfinite(strain))
            print(f"\n===== {ifo.name} DATA CHECK =====")
            print(f"Shape:       {strain.shape}")
            print(f"Data finite: {data_finite}")
            print(f"NaNs:        {np.sum(np.isnan(strain))}")
            if not data_finite:
                print(
                    f"✗ {ifo.name} EXCLUDED: "
                    f"data contains NaNs or non-finite values"
                )
                continue

            #saves useable data to hdf5 file
            strain_file = os.path.join(
                grb_dir,
                f"{grb_name}_{ifo.name}.hdf5",
            )
            ts.write(
                strain_file,
                format="hdf5",
                overwrite=True,
            )
            detector_data[ifo.name] = strain_file
            print(
                f"✓ {ifo.name} INCLUDED"
            )
            print(
                f"Saved {ifo.name}: {strain_file}"
            )

        #excludes detectors with missing data
        except Exception as e:
            print(
                f"✗ {ifo.name} EXCLUDED: "
                f"could not download/read data"
            )
            print(f"  Reason: {type(e).__name__}: {e}")
            continue
    
    #saves list of successful detectors to metadata
    successful_detectors = list(detector_data.keys())
    print()
    print("=" * 70)
    print("SUCCESSFUL DETECTORS")
    print("=" * 70)
    print(successful_detectors)
    metadata["detectors"] = successful_detectors

    #saves metadata to pickle file
    metadata_file = os.path.join(
        grb_dir,
        f"{grb_name}_metadata.pkl",
    )
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    print()
    print(f"Saved metadata: {metadata_file}")

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