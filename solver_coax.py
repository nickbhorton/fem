import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0

from lib import assemble_k_alpha, fem_plot

mesh = meshio.read("coax.msh")

element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
line_tag = mesh.get_cell_data("gmsh:physical", "line")
name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

node_coords = mesh.points[:, :2]

element_indices = mesh.get_cells_type("triangle")
line_indices = mesh.get_cells_type("line")

inner_line_indices = line_indices[line_tag == name_to_tag["inner"]]
outer_line_indices = line_indices[line_tag == name_to_tag["outer"]]
domain_element_indicies = element_indices[element_tag == name_to_tag["domain"]]


inner_radius = np.abs(node_coords[inner_line_indices].max())
outer_radius = np.abs(node_coords[outer_line_indices].max())

k_alpha = assemble_k_alpha(node_coords, domain_element_indicies)
inner_unique_indicies = np.unique(inner_line_indices.ravel())
outer_unique_indicies = np.unique(outer_line_indices.ravel())
dirichlet_unique_indicies = np.concatenate(
    [inner_unique_indicies, outer_unique_indicies]
)
free_unique_indicies = np.setdiff1d(
    np.unique(domain_element_indicies.ravel()), dirichlet_unique_indicies
)

inner_potential = 1.0
outer_potential = 0.0
phi_dirichlet = np.concatenate(
    [
        inner_potential * np.ones(len(inner_unique_indicies)),
        outer_potential * np.ones(len(outer_unique_indicies)),
    ]
)

k_alpha_free = k_alpha[np.ix_(free_unique_indicies, free_unique_indicies)]
b_dirichlet = (
    -k_alpha[np.ix_(free_unique_indicies, dirichlet_unique_indicies)] @ phi_dirichlet
)

phi = np.zeros(len(node_coords))
phi[dirichlet_unique_indicies] = phi_dirichlet
phi[free_unique_indicies] = np.asarray(spla.spsolve(k_alpha_free.tocsc(), b_dirichlet))

C_sim = epsilon_0 * phi @ (k_alpha @ phi) / (inner_potential - outer_potential) ** 2
C_exact = 2 * np.pi * epsilon_0 / np.log(outer_radius / inner_radius)
print(C_sim * 1e12)
print(C_exact * 1e12)
print(np.abs(C_sim - C_exact) / C_exact * 100, "% error")

fig1 = fem_plot(node_coords, domain_element_indicies, phi, True, "plots/coax.png")
plt.show()
