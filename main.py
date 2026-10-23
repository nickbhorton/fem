import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
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
    fem_plot2,
    polar_plot,
)

dB = lambda x, wavelength: 10.0 * np.log10(x / wavelength)


f = 10e9
E0 = 1.0
H0 = E0 / np.sqrt(mu_0 / epsilon_0)
phi_samples = 2000
phi = np.linspace(0, 2.0 * np.pi, phi_samples)
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
a = wavelength / 2
k = 2.0 * np.pi * f / phase_velocity


lc = 0.001
mesh_pec = build_pec_mesh(f, lc)
mesh_die = build_dielectric_mesh(f, lc)

sol_pec_tmz = fem_pec_tmz(mesh_pec, f)
sol_pec_tez = fem_pec_tez(mesh_pec, f)
sol_die_tmz = fem_die_tmz(mesh_die, f)
sol_die_tez = fem_die_tez(mesh_die, f)

# field plot
cmap = "bwr"
apply = lambda x: np.real(x)
fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10, 10))
triang_pec = tri.Triangulation(
    mesh_pec.points[:, 0],
    mesh_pec.points[:, 1],
    triangles=mesh_pec.get_cells_type("triangle"),
)
triang_die = tri.Triangulation(
    mesh_die.points[:, 0],
    mesh_die.points[:, 1],
    triangles=mesh_die.get_cells_type("triangle"),
)
pc00 = ax[0][0].tripcolor(
    triang_pec,
    apply(sol_pec_tmz),
    shading="gouraud",
    cmap=cmap,
    vmin=-np.max(np.abs(sol_pec_tmz)),
    vmax=np.max(np.abs(sol_pec_tmz)),
)
pc01 = ax[0][1].tripcolor(
    triang_pec,
    apply(sol_pec_tez),
    shading="gouraud",
    cmap=cmap,
    vmin=-np.max(np.abs(sol_pec_tez)),
    vmax=np.max(np.abs(sol_pec_tez)),
)
pc10 = ax[1][0].tripcolor(
    triang_die,
    apply(sol_die_tmz),
    shading="gouraud",
    cmap=cmap,
    vmin=-np.max(np.abs(sol_die_tmz)),
    vmax=np.max(np.abs(sol_die_tmz)),
)
pc11 = ax[1][1].tripcolor(
    triang_die,
    apply(sol_die_tez),
    shading="gouraud",
    cmap=cmap,
    vmin=-np.max(np.abs(sol_die_tez)),
    vmax=np.max(np.abs(sol_die_tez)),
)
ax[0][0].set_title(r"PEC $\text{TM}_z$")
ax[0][1].set_title(r"PEC $\text{TE}_z$")
ax[1][0].set_title(r"Dielectric $\text{TM}_z$")
ax[1][1].set_title(r"Dielectric $\text{TE}_z$")
for a in ax.flat:
    a.set_aspect("equal")
fig.colorbar(pc00, ax=ax[0][0], fraction=0.046, pad=0.04)
fig.colorbar(pc01, ax=ax[0][1], fraction=0.046, pad=0.04)
fig.colorbar(pc10, ax=ax[1][0], fraction=0.046, pad=0.04)
fig.colorbar(pc11, ax=ax[1][1], fraction=0.046, pad=0.04)

# bistatic echo width plot
N = 15
bist_pec_tmz = bistatic_echo_width(mesh_pec, sol_pec_tmz, phi, k, 1.5 * wavelength, N, E0)
bist_pec_tez = bistatic_echo_width(mesh_pec, sol_pec_tez, phi, k, 1.5 * wavelength, N, H0)
bist_die_tmz = bistatic_echo_width(mesh_die, sol_die_tmz, phi, k, 1.5 * wavelength, N, E0)
bist_die_tez = bistatic_echo_width(mesh_die, sol_die_tez, phi, k, 1.5 * wavelength, N, H0)

bist_a_pec_tmz = bistatic_echo_width_pec_tmz_analytical(phi, N)
bist_a_pec_tez = bistatic_echo_width_pec_tez_analytical(phi, N)
bist_a_die_tmz = bistatic_echo_width_die_tmz_analytical(phi, N)
bist_a_die_tez = bistatic_echo_width_die_tez_analytical(phi, N)

apply = lambda x: 10.0 * np.log10(x / wavelength)
fig, ax = plt.subplots(layout="constrained")
ax.remove()

ax00 = fig.add_subplot(2, 2, 1, projection="polar")
ax00.plot(phi, apply(bist_pec_tmz))
ax00.plot(phi, apply(bist_a_pec_tmz))

ax01 = fig.add_subplot(2, 2, 2, projection="polar")
ax01.plot(phi, apply(bist_pec_tez))
ax01.plot(phi, apply(bist_a_pec_tez))

ax10 = fig.add_subplot(2, 2, 3, projection="polar")
ax10.plot(phi, apply(bist_die_tmz))
ax10.plot(phi, apply(bist_a_die_tmz))

ax11 = fig.add_subplot(2, 2, 4, projection="polar")
ax11.plot(phi, apply(bist_die_tez))
ax11.plot(phi, apply(bist_a_die_tez))

plt.show()
