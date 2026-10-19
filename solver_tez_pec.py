import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0, mu_0

from lib import assemble_k_alpha, assemble_k_beta, assemble_k_gamma, fem_plot2

mesh = meshio.read("mesh/pec.msh")

element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
line_tag = mesh.get_cell_data("gmsh:physical", "line")
name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

node_coords = mesh.points[:, :2]
x = node_coords[:, 0]
y = node_coords[:, 1]
rho = np.sqrt(x**2 + y**2)

element_indices = mesh.get_cells_type("triangle")
line_indices = mesh.get_cells_type("line")

# boundary condition lines
inner_line_indices = line_indices[line_tag == name_to_tag["inner"]]
outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]

f = 10e9
k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
mu_r = np.ones(node_coords.shape[0])
epsilon_r = np.ones(node_coords.shape[0])
E0 = 1.0
H0 = 1.0 / np.sqrt(mu_0 / epsilon_0)

beta = -(k_0**2) * mu_r
alpha = 1 / epsilon_r
gamma = 1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1)

element_unique_node_indices = np.unique(element_indices.ravel())
inner_unique_indices = np.unique(inner_line_indices.ravel())
free_node_indices = element_unique_node_indices

# k_alpha
k_alpha = assemble_k_alpha(node_coords, element_indices, alpha)
k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

# k_beta
k_beta = assemble_k_beta(node_coords, element_indices, beta)
k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

# k_gamma
k_gamma = assemble_k_gamma(node_coords, outer_line_indices, gamma)
k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

# b_neumann
b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

kappa = (
    H0
    * np.exp(-1j * k_0 * x[outer_line_indices])
    * (
        -(1j * k_0 * x[outer_line_indices]) / rho[outer_line_indices]
        + 1j * k_0
        + 1 / (2 * rho[outer_line_indices])
    )
).mean(axis=1)
# (number_neumann_edges, (p1, p2), (x, y))
edges_neumann = node_coords[outer_line_indices]
edge_lengths_neumann = np.linalg.norm(
    edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1
)
np.add.at(
    b_neumann,
    outer_line_indices.ravel(),
    np.repeat(0.5 * kappa * edge_lengths_neumann, 2),
)
b_neumann = b_neumann[free_node_indices]

# final assembly
k_total_free = k_alpha_free + k_beta_free + k_gamma_free
b_total = b_neumann

# solve for phi
phi = np.zeros(len(node_coords), dtype=np.complex128)
phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))

# plots
fem_plot2(node_coords, element_indices, np.abs(phi))

plt.show()
