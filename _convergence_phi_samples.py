import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0, mu_0

from lib import (
    bistatic_echo_width,
    bistatic_echo_width_pec_tmz_analytical,
    build_pec_mesh,
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

lc = 0.001
N = 15
phi_samples_arr = np.linspace(16, 2000, 15)
error = np.zeros_like(phi_samples_arr, dtype=np.float64)

for i, phi_samples in enumerate(phi_samples_arr):
    phi = np.linspace(0, 2.0 * np.pi, int(phi_samples))
    mesh_pec = build_pec_mesh(f, lc)
    sol_pec_tmz = fem_pec_tmz(mesh_pec, f)
    bist_pec_tmz = bistatic_echo_width(
        mesh_pec, sol_pec_tmz, phi, k, 1.5 * wavelength, N, E0
    )
    bist_a_pec_tmz = bistatic_echo_width_pec_tmz_analytical(phi, N)
    error[i] = rmse(bist_pec_tmz, bist_a_pec_tmz)

fig, ax = plt.subplots()
ax.scatter(phi_samples_arr, error)
plt.show()

# 1000 seems entirely reasonable