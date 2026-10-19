import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio

from lib import assemble_k_alpha

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
inner_d = lines[line_tag == name_to_tag["inner_d"]]
outer_d = lines[line_tag == name_to_tag["outer_d"]]

# elements
domain = elements[element_tag == name_to_tag["domain"]]

# alpha
a = np.abs(nodes[inner_d].max())
b = np.abs(nodes[outer_d].max())
epsilon_m = 5
rho = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
epsilon_r = epsilon_m * (a / rho)

inner_potential = 1.0
outer_potential = 0.0
k_alpha = assemble_k_alpha(nodes, domain, epsilon_r)
nodes_inner_d = np.unique(inner_d.ravel())
nodes_outer_d = np.unique(outer_d.ravel())
nodes_dirichlet = np.concatenate([nodes_inner_d, nodes_outer_d])
nodes_free = np.setdiff1d(np.unique(domain.ravel()), nodes_dirichlet)

phi_dirichlet = np.concatenate(
    [
        inner_potential * np.ones(len(nodes_inner_d)),
        outer_potential * np.ones(len(nodes_outer_d)),
    ]
)

k_alpha_free = k_alpha[np.ix_(nodes_free, nodes_free)]
b_dirichlet = -k_alpha[np.ix_(nodes_free, nodes_dirichlet)] @ phi_dirichlet

phi = np.zeros(len(nodes))
phi[nodes_dirichlet] = phi_dirichlet
phi[nodes_free] = np.asarray(spla.spsolve(k_alpha_free.tocsc(), b_dirichlet))

C_sim = epsilon_0 * phi @ (k_alpha @ phi) / (inner_potential - outer_potential) ** 2
C_exact = 2 * np.pi * epsilon_0 * epsilon_m * a / (b - a) * (1 / 12)
# print(C_sim * 1e12 * 12)
# print(C_exact * 1e12 * 12)
print(np.abs(C_sim - C_exact) / C_exact * 100, "% error")

# plot
fig, ax = plt.subplots()
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], domain)
pc = ax.tripcolor(triang, phi, shading="gouraud", cmap="inferno")
fig.colorbar(pc, ax=ax)
ax.set_aspect("equal")
plt.show()
