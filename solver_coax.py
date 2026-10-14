import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio

from lib import assemble_k_alpha

mesh = meshio.read("coax.msh")
if mesh is None:
    exit(1)

nodes = mesh.points[:, :2]
elements = mesh.get_cells_type("triangle")
element_tag = mesh.get_cell_data("gmsh:physical", "triangle")

lines = mesh.get_cells_type("line")
line_tag = mesh.get_cell_data("gmsh:physical", "line")

name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

inner = lines[line_tag == name_to_tag["inner"]]
outer = lines[line_tag == name_to_tag["outer"]]
domain = elements[element_tag == name_to_tag["domain"]]


a = np.abs(nodes[inner].max())
b = np.abs(nodes[outer].max())
inner_potential = 1.0
outer_potential = 0.0
k_alpha = assemble_k_alpha(nodes, domain)
nodes_inner = np.unique(inner.ravel())
nodes_outer = np.unique(outer.ravel())
nodes_dirichlet = np.concatenate([nodes_inner, nodes_outer])
nodes_free = np.setdiff1d(np.unique(domain.ravel()), nodes_dirichlet)

phi_dirichlet = np.concatenate(
    [
        inner_potential * np.ones(len(nodes_inner)),
        outer_potential * np.ones(len(nodes_outer)),
    ]
)

k_alpha_free = k_alpha[np.ix_(nodes_free, nodes_free)]
b_dirichlet = -k_alpha[np.ix_(nodes_free, nodes_dirichlet)] @ phi_dirichlet

phi = np.zeros(len(nodes))
phi[nodes_dirichlet] = phi_dirichlet
phi[nodes_free] = np.asarray(spla.spsolve(k_alpha_free.tocsc(), b_dirichlet))

C_sim = epsilon_0 * phi @ (k_alpha @ phi) / (inner_potential - outer_potential) ** 2
C_exact = 2 * np.pi * epsilon_0 / np.log(b / a)
print(np.abs(C_sim - C_exact) / C_exact * 100, "% error")

# plot
fig, ax = plt.subplots()
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], domain)
pc = ax.tripcolor(triang, phi, shading="gouraud", cmap="inferno")
fig.colorbar(pc, ax=ax)
ax.set_aspect("equal")
plt.show()
