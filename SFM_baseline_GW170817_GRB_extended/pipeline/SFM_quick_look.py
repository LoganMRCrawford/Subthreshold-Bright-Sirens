import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update('jax_enable_x64', True)

import pickle
import numpy as np
import pandas as pd

dir_path = "../data"
filename = "grb_results.csv"

grb_file = os.path.join(dir_path, filename)

df = pd.read_csv(grb_file)

table = df[
    [
        "GRB",
        "logZ_Unconditioned_tran",
        "logZ_Unconditioned_err_tran",
        "logZ_Conditioned_tran",
        "logZ_Conditioned_err_tran",
        "delta_logZ_tran",
        "delta_logZ_err_tran",
    ]
].copy()

table["Unconditioned"] = (
    table["logZ_Unconditioned_tran"].map("{:.3f}".format)
    + " +/- "
    + table["logZ_Unconditioned_err_tran"].map("{:.3f}".format)
)

table["Conditioned"] = (
    table["logZ_Conditioned_tran"].map("{:.3f}".format)
    + " +/- "
    + table["logZ_Conditioned_err_tran"].map("{:.3f}".format)
)

table["Delta logZ"] = (
    table["delta_logZ_tran"].map("{:.3f}".format)
    + " +/- "
    + table["delta_logZ_err_tran"].map("{:.3f}".format)
)

table = table[
    ["GRB", "Unconditioned", "Conditioned", "Delta logZ"]
]

print(table.to_string(index=False))