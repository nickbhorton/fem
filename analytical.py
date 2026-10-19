import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.special import hankel2, jn

f = 10e9
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
pec_radius = wavelength / 2
k = 2.0 * np.pi * f / phase_velocity

grid_points = 300
wavelength_grid = 2.5
abc_radius = wavelength_grid * wavelength
x = np.linspace(
    -wavelength * wavelength_grid, wavelength * wavelength_grid, grid_points
)
y = np.linspace(
    -wavelength * wavelength_grid, wavelength * wavelength_grid, grid_points
)
x, y = np.meshgrid(x, y)
rho = np.sqrt(x**2 + y**2)
phi = np.atan2(y, x)

E0 = 1.0
Ez_i = E0 * np.exp(-1j * k * x)

N = 20
Ez_s_n = np.zeros((2 * N + 1,) + x.shape, dtype=np.complex128)
for n in range(-N, N + 1):
    i = n + N
    Ez_s_n[i][rho > pec_radius] = (
        ((-1.0j) ** n * jn(n, k * pec_radius) / hankel2(n, k * pec_radius))
        * hankel2(n, k * rho)
        * np.exp(1.0j * n * phi)
    )[rho > pec_radius]

Ez_s = -E0 * np.sum(Ez_s_n, axis=0)
Ez_s[rho <= pec_radius] = -Ez_i[rho <= pec_radius]

Ez_t = Ez_i + Ez_s

rho = np.sqrt(x**2 + y**2)
annulus_mask = (rho < pec_radius) | (rho > abc_radius)
abs_total_E_masked = np.ma.masked_where(annulus_mask, np.abs(Ez_t))

fig, ax = plt.subplots(layout="constrained")
mesh = ax.pcolormesh(x / wavelength, y / wavelength, abs_total_E_masked, cmap="hot")
cbar = fig.colorbar(mesh)

ax.set_xlabel(r"$x/\lambda$")
ax.set_ylabel(r"$y/\lambda$")
ax.set_aspect("equal")

plt.show()
