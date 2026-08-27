![Project Screenshot](fig1-5.jpg)

# Subthreshold-Bright-Sirens
The venture to seek bright siren events that have slipped under the radar of gravitational wave inferometers. So called "shadow sirens" are attempted to be uncovered via the use of GRB catalogues to pinpoint them on the sky, and improved noise modelling to increase our confidence that an event occurred.

# Getting Started...
1. Create a python virtual environment (via spack/conda/pyenv etc.)
2. !important! pip install a cuda activated jax version i.e. 'pip install "jax[cuda12]"==0.10.2 '
3. pip install blackjax from the handley-lab repo: most importantly you need a version with the "stepper_fn". Here is the exact branch we used, in our environments: 

    git clone https://github.com/handley-lab/blackjax/tree/2180e29ffb645b2c46b76c768229c7c24212446c
    cd into the cloned repository
    python -m pip install .

    You may find a more recent branch from this repo will still have this functionality, and new features, too.
4. pip install -r "requirements.txt" 
    Note: you may have to cd back into the repo that requirements.txt is installed


# SFM_baseline_GW170817_GRB_extended
- data
    - grb_catalogue.csv saves here: contains list of GRBs with GPS, RA, DEC, T90 and LIGO run info
    - grb_results.csv saves here: contains list of GRB results including valid detectors, unconditioned, conditioned and delta logZ with errors and MAPs across all parameters for both unconditioned and conditioned runs
    - individual GRB output directories save here: may contain H1, L1 and V1 strain files (.hdf5), metadata file (.pkl), unconditioned and conditioned transient samples (.csv), unconditioned and conditioned corner plots (.png), strain data plot (.png) and MAP comparison plot (.png)
- pipeline
    - deprecated folder: contains older files which may be relevant if moving onto heterodyned likelihood in the future
    - _plot_utils.py: contains useful functions for plotting corner plots or finding the MAP of a posterior
    - SFM_data_collection_GRB.py:
        - If catalogue_read = True at start of file, it queries the Fermi GBM Burst Catalogue and saves the list of suitable GRBs to grb_catalogue.csv. The conditions for 'suitable' can be changed, and currently requires trigger times to be within LIGO observing windows, T90<3s and error radius <10 degrees.
        - GRBs in the list SELECTED_GRBs = ["GRB250119945"] will have their strain data downloaded from LIGO Hanford, LIGO Livingston and Virgo detectors. Leaving the list empty will loop over all GRBs in the grb_catalogue.csv, and specified GRBs must exist in the downloaded catalogue. Data is saved to the GRB specific output folder, and is only saved if the full segment selected exists and there are no Nans, hence not all GRBs will have three strain files.
    - SFM_sampling_GRB.py: 
        - Must be run with parser argument --grb-names GRB012345678.
        - Performs unconditioned (full sky prior) and conditioned (GRB location gaussian prior) nested sampling on a single GRB location and saves both sets of samples to the GRB specific output folder.
        - Sampling uses TransientLikelihoodFD (full parameter space), phase marginalization, IMRPhenomD_NRTidalv2 waveform and a Welch estimate for the psd.
        - Priors are set reasonably wide to encompass most physical parameters whilst keeping run time suitable. Most importantly, coalescent time has a prior between [-10,0] as the time delay between GW and GRB events is not well constrained.
        - Default settings use 500 live points, 0.3 x live point ndelete and 8 x dimensions for nmcmc.
        - Script was built from a previous file using heterodyning, so there may be left over artefacts that are no longer used. References to H0 and vp fitting are purposely left commented out but not deleted as they may be relevant in the future.
    - SFM_analysis_GRB.py:
        - Uses same format as SFM_data_collection_GRB.py with SELECTED_GRBs = ["GRB250119945"] and looping over all directories if list is empty.
        - If strain_plotting = True, plots and saves graph of strain data from all valid detectors.
        - If logZ_and_maps = True, calculates unconditioned, conditioned and delta logZs as well as MAPs across all parameters. Saves results along with other important metadata to grb_results.csv.
        - If corner_plotting = True, plots and saves unconditioned and conditioned corner plots.
        - All images are saved to grb specific output folder.
    - SFM_results_GRB.py:
        - Uses same format as SFM_data_collection_GRB.py with SELECTED_GRBs = ["GRB250119945"] and looping over all GRBs in results file if list is empty.
        - Prints table of unconditioned, conditioned and delta logZ with errors for all GRBs, ordering from highest to lowest delta logZ.
        - For specified GRBs, plots MAPs and medians with 16-84 percentile error bars for each parameter for both unconditioned and conditioned samples. Saves to grb specific output folder.
