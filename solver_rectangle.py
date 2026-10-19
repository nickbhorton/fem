import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0, mu_0

from lib import assemble_k_alpha, assemble_k_beta, fem_plot

mesh = meshio.read("rectangle.msh")

element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
line_tag = mesh.get_cell_data("gmsh:physical", "line")
name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

node_coords = mesh.points[:, :2]
element_indices = mesh.get_cells_type("triangle")
line_indices = mesh.get_cells_type("line")

# boundary condition lines
port1_line_indices = line_indices[line_tag == name_to_tag["left"]]
port2_line_indices = line_indices[line_tag == name_to_tag["right"]]
bot_line_indices = line_indices[line_tag == name_to_tag["bot"]]

f = 374.74e6
k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
mu_r = np.ones(node_coords.shape[0])
epsilon_r = np.ones(node_coords.shape[0])
epsilon_r[node_coords[:, 0] > 0.0] = 4.0

beta = -mu_r * k_0**2
alpha = 1 / epsilon_r

element_unique_node_indices = np.unique(element_indices.ravel())
port1_unique_node_indices = np.unique(port1_line_indices.ravel())
port2_unique_node_indices = np.unique(port2_line_indices.ravel())
bot_unique_node_indices = np.unique(bot_line_indices.ravel())
neumann_node_indices = np.concatenate(
    [port1_unique_node_indices, port2_unique_node_indices]
)

free_node_indices = element_unique_node_indices

# k_alpha
k_alpha = assemble_k_alpha(node_coords, element_indices, alpha)
k_alpha_free = k_alpha[np.ix_(free_node_indices, free_node_indices)]

# k_beta
k_beta = assemble_k_beta(node_coords, element_indices, beta)
k_beta_free = k_beta[np.ix_(free_node_indices, free_node_indices)]

# k_gamma
port_line_indices = np.concatenate([port1_line_indices, port2_line_indices])
port_gamma_lines = (
    -1j * k_0 * np.sqrt(mu_r[port_line_indices] / epsilon_r[port_line_indices])
).mean(axis=1)
port_edges_gamma = node_coords[port_line_indices]
port_edge_lengths_gamma = np.linalg.norm(
    port_edges_gamma[:, 0, :] - port_edges_gamma[:, 1, :], axis=1
)
k_gamma_vals = np.squeeze(
    port_gamma_lines[:, np.newaxis]
    * port_edge_lengths_gamma[:, np.newaxis]
    * np.array([[1 / 3, 1 / 6, 1 / 6, 1 / 3]])[np.newaxis, :]
)
rows_gamma = np.repeat(port_line_indices, 2, axis=1)  # [1 1 2 2]
cols_gamma = np.tile(port_line_indices, (1, 2))  # [1 2 1 2]

k_gamma = sp.coo_matrix(
    (k_gamma_vals.ravel(), (rows_gamma.ravel(), cols_gamma.ravel())),
    shape=(node_coords.shape[0], node_coords.shape[0]),
    dtype=np.complex128,
).tocsr()

k_gamma_free = k_gamma[np.ix_(free_node_indices, free_node_indices)]

# b_neumann
b_neumann = np.zeros(len(node_coords), dtype=np.complex128)

port1_kappa_neumann = (
    -2j
    * k_0
    * np.sqrt(mu_r[port1_line_indices] / epsilon_r[port1_line_indices])
    * np.exp(-1j * k_0 * node_coords[port1_line_indices, 0])
).mean(axis=1)

# (number_neumann_edges, (p1, p2), (x, y))
edges_neumann = node_coords[port1_line_indices]
edge_lengths = np.linalg.norm(edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1)
np.add.at(
    b_neumann,
    port1_line_indices.ravel(),
    np.repeat(0.5 * port1_kappa_neumann * edge_lengths, 2),
)
b_neumann = b_neumann[free_node_indices]

k_total_free = k_alpha_free + k_beta_free + sp.csr_matrix(k_gamma_free)
b_total = b_neumann

phi = np.zeros(len(node_coords), dtype=np.complex128)
phi[free_node_indices] = np.asarray(spla.spsolve(k_total_free.tocsc(), b_total))

# fem_plot(node_coords, element_indices, np.real(phi))

fig, ax = plt.subplots()
ax.scatter(node_coords[bot_line_indices, 0], np.abs(phi[bot_line_indices]))

plt.show()
