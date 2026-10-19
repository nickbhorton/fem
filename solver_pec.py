import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0, mu_0

from lib import assemble_k_alpha, assemble_k_beta, fem_plot2

mesh = meshio.read("pec.msh")

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

beta = -(k_0**2) * epsilon_r
alpha = 1 / mu_r

element_unique_node_indices = np.unique(element_indices.ravel())
inner_unique_indices = np.unique(inner_line_indices.ravel())
dirichlet_node_indices = np.concatenate([inner_unique_indices])
free_node_indices = np.setdiff1d(element_unique_node_indices, dirichlet_node_indices)

# k_alpha
k_alpha = assemble_k_alpha(node_coords, element_indices, alpha)
k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

# k_beta
k_beta = assemble_k_beta(node_coords, element_indices, beta)
k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

# k_gamma
gamma = 1j * k_0 + 1 / (2 * rho[outer_line_indices]).mean(axis=1)
edges_gamma = node_coords[outer_line_indices]
edge_lengths_gamma = np.linalg.norm(edges_gamma[:, 0, :] - edges_gamma[:, 1, :], axis=1)
k_gamma_vals = np.squeeze(
    gamma[:, np.newaxis]
    * edge_lengths_gamma[:, np.newaxis]
    * np.array([1 / 3, 1 / 6, 1 / 6, 1 / 3])[np.newaxis, :]
)
rows_gamma = np.repeat(outer_line_indices, 2, axis=1)  # [1 1 2 2]
cols_gamma = np.tile(outer_line_indices, (1, 2))  # [1 2 1 2]
k_gamma = sp.coo_matrix(
    (k_gamma_vals.ravel(), (rows_gamma.ravel(), cols_gamma.ravel())),
    shape=(node_coords.shape[0], node_coords.shape[0]),
    dtype=np.complex128,
).tocsr()

k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

# b_dirichlet
phi_dirichlet = np.concatenate(
    [
        np.zeros(len(dirichlet_node_indices)),
    ]
)
b_dirichlet = (
    -k_alpha[np.ix_(free_node_indices, dirichlet_node_indices)] @ phi_dirichlet
)

# b_neumann
b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

kappa = (
    E0
    * np.exp(-1j * k_0 * x[outer_line_indices])
    * (
        1j * k_0
        - (1j * k_0 * x[outer_line_indices]) / rho[outer_line_indices]
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

k_total_free = k_alpha_free + k_beta_free + sp.csr_matrix(k_gamma_free)
b_total = b_neumann + b_dirichlet

phi = np.zeros(len(node_coords), dtype=np.complex128)
phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))

fem_plot2(node_coords, element_indices, np.abs(phi))

plt.show()
