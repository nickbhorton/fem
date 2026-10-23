import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0, mu_0

from lib import (
    bistatic_echo_width,
    bistatic_echo_width_die_tez_analytical,
    bistatic_echo_width_die_tmz_analytical,
    bistatic_echo_width_pec_tez_analytical,
    bistatic_echo_width_pec_tmz_analytical,
    build_dielectric_mesh,
    build_pec_mesh,
    fem_die_tez,
    fem_die_tmz,
    fem_pec_tez,
    fem_pec_tmz,
)


def rmse(sig_sim, sig):
    return np.sqrt((np.sum(np.square(sig_sim - sig))) / len(sig)) / np.sqrt(
        np.sum(np.square(sig)) / len(sig)
    )


dB = lambda x, wavelength: 10.0 * np.log10(x / wavelength)

f = 10e9
E0 = 1.0
H0 = E0 / np.sqrt(mu_0 / epsilon_0)
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
a = wavelength / 2
k = 2.0 * np.pi * f / phase_velocity

phi_samples = 1000
N = 20
lcs = np.logspace(-1.7, -3.7, 50)

error_pec_tmz = np.zeros_like(lcs, dtype=np.float64)
error_pec_tez = np.zeros_like(lcs, dtype=np.float64)
error_die_tmz = np.zeros_like(lcs, dtype=np.float64)
error_die_tez = np.zeros_like(lcs, dtype=np.float64)
nodes_pec = np.zeros_like(lcs, dtype=np.int32)
nodes_die = np.zeros_like(lcs, dtype=np.int32)

for i, lc in enumerate(lcs):
    phi = np.linspace(0, 2.0 * np.pi, phi_samples)
    mesh_pec = build_pec_mesh(f, lc)
    mesh_die = build_dielectric_mesh(f, lc)

    sol_pec_tmz = fem_pec_tmz(mesh_pec, f)
    sol_pec_tez = fem_pec_tez(mesh_pec, f)
    sol_die_tmz = fem_die_tmz(mesh_die, f)
    sol_die_tez = fem_die_tez(mesh_die, f)

    bist_pec_tmz = bistatic_echo_width(
        mesh_pec, sol_pec_tmz, phi, k, 1.5 * wavelength, 15, E0
    )
    bist_pec_tez = bistatic_echo_width(
        mesh_pec, sol_pec_tez, phi, k, 1.5 * wavelength, 15, H0
    )
    bist_die_tmz = bistatic_echo_width(
        mesh_die, sol_die_tmz, phi, k, 1.5 * wavelength, 15, E0
    )
    bist_die_tez = bistatic_echo_width(
        mesh_die, sol_die_tez, phi, k, 1.5 * wavelength, 15, H0
    )

    bist_a_pec_tmz = bistatic_echo_width_pec_tmz_analytical(phi, int(N))
    bist_a_pec_tez = bistatic_echo_width_pec_tez_analytical(phi, int(N))
    bist_a_die_tmz = bistatic_echo_width_die_tmz_analytical(phi, int(N))
    bist_a_die_tez = bistatic_echo_width_die_tez_analytical(phi, int(N))

    error_pec_tmz[i] = rmse(bist_pec_tmz, bist_a_pec_tmz)
    error_pec_tez[i] = rmse(bist_pec_tez, bist_a_pec_tez)
    error_die_tmz[i] = rmse(bist_die_tmz, bist_a_die_tmz)
    error_die_tez[i] = rmse(bist_die_tez, bist_a_die_tez)

    nodes_pec[i] = mesh_pec.points.shape[0]
    nodes_die[i] = mesh_die.points.shape[0]

np.savez(
    "node_convergence.npz",
    **{
        "nodes_pec": nodes_pec,
        "nodes_die": nodes_die,
        "error_pec_tmz": error_pec_tmz,
        "error_pec_tez": error_pec_tez,
        "error_die_tmz": error_die_tmz,
        "error_die_tez": error_die_tez,
    },
)
