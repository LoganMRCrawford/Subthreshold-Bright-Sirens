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


