import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0

from lib import assemble_k_alpha, fem_plot

mesh = meshio.read("partial_coax.msh")
if mesh is None:
    exit(1)

nodes = mesh.points[:, :2]
elements = mesh.get_cells_type("triangle")
element_tag = mesh.get_cell_data("gmsh:physical", "triangle")

lines = mesh.get_cells_type("line")
line_tag = mesh.get_cell_data("gmsh:physical", "line")

name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

# boundary condition lines
inner_n = lines[line_tag == name_to_tag["inner_d"]]
outer_d = lines[line_tag == name_to_tag["outer_d"]]

# elements
domain = elements[element_tag == name_to_tag["domain"]]

# alpha
a = np.abs(nodes[inner_n].max())
b = np.abs(nodes[outer_d].max())
epsilon_m = 5
rho = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
epsilon_r = epsilon_m * (a / rho)

outer_potential = 0.0
inner_potential = 1.0
c_prime = 2 * np.pi * epsilon_0 * epsilon_m * a / (b - a)
rho_s = c_prime / (2.0 * np.pi * a * epsilon_0)

k_alpha = assemble_k_alpha(nodes, domain, epsilon_r)
nodes_outer_d = np.unique(outer_d.ravel())
nodes_inner_n = np.unique(inner_n.ravel())
nodes_dirichlet = np.concatenate([nodes_outer_d])
nodes_neumann = np.concatenate([nodes_inner_n])
nodes_boundary = np.concatenate([nodes_dirichlet, nodes_neumann])
nodes_free = np.setdiff1d(np.unique(domain.ravel()), nodes_dirichlet)

phi_dirichlet = np.concatenate(
    [
        outer_potential * np.ones(len(nodes_outer_d)),
    ]
)
kappa_neumann = rho_s

k_alpha_free = k_alpha[np.ix_(nodes_free, nodes_free)]
b_dirichlet = -k_alpha[np.ix_(nodes_free, nodes_dirichlet)] @ phi_dirichlet

b_neumann = np.zeros(len(nodes))
edges_neumann = nodes[inner_n]  # (number_neumann_edges, (p1, p2), (x, y))
edge_lengths = np.linalg.norm(edges_neumann[:, 0, :] - edges_neumann[:, 1, :], axis=1)
np.add.at(b_neumann, inner_n.ravel(), np.repeat(0.5 * kappa_neumann * edge_lengths, 2))
b_neumann = b_neumann[nodes_free]

b_total = b_dirichlet + b_neumann

phi = np.zeros(len(nodes))
phi[nodes_dirichlet] = phi_dirichlet
phi[nodes_free] = np.asarray(spla.spsolve(k_alpha_free.tocsc(), b_total))

C_sim = epsilon_0 * phi @ (k_alpha @ phi) / (inner_potential - outer_potential) ** 2
C_exact = 2 * np.pi * epsilon_0 * epsilon_m * a / (b - a) * (1 / 12)
print(C_sim * 1e12 * 12)
print(C_exact * 1e12 * 12)
print(np.abs(C_sim - C_exact) / C_exact * 100, "% error")

fem_plot(nodes, domain, phi)
plt.show()
