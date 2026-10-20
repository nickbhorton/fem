import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.constants import epsilon_0, mu_0
from scipy.special import hankel2

from lib import bistatic_echo_width_pec_tmz_analytical, fem_pec_tmz, polar_plot

dB = lambda x, wavelength: 10.0 * np.log10(x / wavelength)

def bistatic_echo_width(mesh, solution, phi, k, investigation_radius, E0=1.0):
    node_coords = mesh.points[:, :2]
    element_indices = mesh.get_cells_type("triangle")
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    triangulation = Triangulation(x, y, element_indices)
    solution_interpolation_generator_re = LinearTriInterpolator(
        triangulation, np.real(solution)
    )
    solution_interpolation_generator_im = LinearTriInterpolator(
        triangulation, np.imag(solution)
    )

    investigation_x = investigation_radius * np.cos(phi)
    investigation_y = investigation_radius * np.sin(phi)
    investigation_solution_re = solution_interpolation_generator_re(
        investigation_x, investigation_y
    )
    investigation_solution_im = solution_interpolation_generator_im(
        investigation_x, investigation_y
    )
    investigation_solution = investigation_solution_re + 1j * investigation_solution_im
    investigation_scattered = investigation_solution - E0 * np.exp(
        -1j * k * investigation_x
    )
    investigation_scattered_fft = np.fft.fft(investigation_scattered) / phi.shape[0]
    N = 15
    ns = np.arange(-N, N + 1)
    investigation_scattered_fft_n_wise = investigation_scattered_fft[ns]
    hankel_coefficients = investigation_scattered_fft_n_wise / hankel2(
        ns, k * investigation_radius
    )
    return (4.0 / k) * np.abs(
        (
            -((1j) ** ns[:, None])
            * hankel_coefficients[:, None]
            * np.exp(-1j * ns[:, None] * phi[None, :])
        ).sum(axis=0)
    ) ** 2


f = 10e9
E0 = 1.0
phi_samples = 10000
phi = np.linspace(0, 2.0 * np.pi, phi_samples)
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
a = wavelength / 2
k = 2.0 * np.pi * f / phase_velocity

bist_analytical = bistatic_echo_width_pec_tmz_analytical(phi)


mesh = meshio.read("mesh/pec.msh")
solution = fem_pec_tmz(mesh)

bist_1p0 = bistatic_echo_width(mesh, solution, phi, k, 1.0 * wavelength, E0)
bist_1p5 = bistatic_echo_width(mesh, solution, phi, k, 1.5 * wavelength, E0)
bist_2p0 = bistatic_echo_width(mesh, solution, phi, k, 2.0 * wavelength, E0)

polar_plot(
    phi,
    [
        dB(bist_analytical, wavelength),
        dB(bist_1p0, wavelength),
        # dB(bist_1p5, wavelength),
        # dB(bist_2p0, wavelength),
    ],
)

fig, ax = plt.subplots()
ax.plot(np.abs((bist_1p0 - bist_analytical) / bist_analytical) * 100)
ax.plot(np.abs((bist_1p5 - bist_analytical) / bist_analytical) * 100)
ax.plot(np.abs((bist_2p0 - bist_analytical) / bist_analytical) * 100)
plt.show()
