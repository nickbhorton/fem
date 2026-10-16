import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0

from lib import assemble_k_alpha, fem_plot

mesh = meshio.read("partial_coax.msh")
if mesh is None:
    exit(1)


element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
line_tag = mesh.get_cell_data("gmsh:physical", "line")
name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

node_coords = mesh.points[:, :2]
element_indicies = mesh.get_cells_type("triangle")
line_indicies = mesh.get_cells_type("line")

# boundary condition lines
inner_n_line_indicies = line_indicies[line_tag == name_to_tag["inner_d"]]
outer_d_line_indicies = line_indicies[line_tag == name_to_tag["outer_d"]]
zero_n_line_indicies = line_indicies[line_tag == name_to_tag["zero_n"]]

# this could be required if you have multiple domains
# domain = elements_indicies[element_tag == name_to_tag["domain"]]

# alpha
a = np.abs(node_coords[inner_n_line_indicies].max())
b = np.abs(node_coords[outer_d_line_indicies].max())
epsilon_m = 5
rho = np.sqrt(node_coords[:, 0] ** 2 + node_coords[:, 1] ** 2)
epsilon_r = epsilon_m * (a / rho)

initial_inner_potential = 1.0
outer_potential = 0.0
c_prime = 2 * np.pi * epsilon_0 * epsilon_m * a / (b - a)
kappa_neumann = (c_prime * initial_inner_potential) / (2.0 * np.pi * a * epsilon_0)

k_alpha = assemble_k_alpha(node_coords, element_indicies, epsilon_r)
element_node_indicies = np.unique(element_indicies.ravel())
outer_d_node_indicies = np.unique(outer_d_line_indicies.ravel())
inner_n_nodes_indicies = np.unique(inner_n_line_indicies.ravel())
dirichlet_node_indicies = np.concatenate([outer_d_node_indicies])
neumann_node_indicies = np.concatenate([inner_n_nodes_indicies])
free_node_indicies = np.setdiff1d(element_node_indicies, dirichlet_node_indicies)

phi_dirichlet = np.concatenate(
    [
        outer_potential * np.ones(len(outer_d_node_indicies)),
    ]
)

k_alpha_free = k_alpha[np.ix_(free_node_indicies, free_node_indicies)]
b_dirichlet = (
    -k_alpha[np.ix_(free_node_indicies, dirichlet_node_indicies)] @ phi_dirichlet
)

b_neumann = np.zeros(len(node_coords))

# (number_neumann_edges, (p1, p2), (x, y))
edges_neumann = node_coords[inner_n_line_indicies]
edge_lengths = np.linalg.norm(edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1)

np.add.at(
    b_neumann,
    inner_n_line_indicies.ravel(),  # every node index on a line
    np.repeat(0.5 * kappa_neumann * edge_lengths, 2),
)
b_neumann = b_neumann[free_node_indicies]

b_total = b_dirichlet + b_neumann

phi = np.zeros(len(node_coords))
phi[dirichlet_node_indicies] = phi_dirichlet
phi[free_node_indicies] = np.asarray(spla.spsolve(k_alpha_free.tocsc(), b_total))

inner_potential = phi[neumann_node_indicies].mean()
voltage = inner_potential - outer_potential
C_sim = epsilon_0 * phi @ (k_alpha @ phi) / voltage**2
C_exact = 2 * np.pi * epsilon_0 * epsilon_m * a / (b - a) * (1 / 12)
print(np.abs(C_sim - C_exact) / C_exact * 100, "% error")

# shows rho vs phi on y=0
# fig, ax = plt.subplots()
# ax.scatter(
#     rho[np.unique(zero_n_line_indicies.ravel())],
#     phi[np.unique(zero_n_line_indicies.ravel())],
# )

fem_plot(node_coords, element_indicies, phi)
plt.show()
