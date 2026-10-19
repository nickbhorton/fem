import meshio
import numpy as np
from scipy.constants import epsilon_0, mu_0

from lib import assemble_k_alpha, assemble_k_beta

mesh = meshio.read("rectangle.msh")

element_tag = mesh.get_cell_data("gmsh:physical", "triangle")
line_tag = mesh.get_cell_data("gmsh:physical", "line")
name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

node_coords = mesh.points[:, :2]
element_indicies = mesh.get_cells_type("triangle")
line_indicies = mesh.get_cells_type("line")

# boundary condition lines
left_line_indicies = line_indicies[line_tag == name_to_tag["left"]]
right_line_indicies = line_indicies[line_tag == name_to_tag["right"]]

f = 10e9
k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
mu_r = np.ones(node_coords.shape[0])
epsilon_r = np.ones(node_coords.shape[0])
epsilon_r[node_coords[:, 0] < 0.0] = 4.0

beta = mu_r * k_0**2
alpha = 1 / epsilon_r

element_node_indicies = np.unique(element_indicies.ravel())
port1_node_indicies = np.unique(left_line_indicies.ravel())
port2_nodes_indicies = np.unique(right_line_indicies.ravel())
neumann_node_indicies = np.concatenate([port1_node_indicies, port2_nodes_indicies])

free_node_indicies = element_node_indicies

k_alpha = assemble_k_alpha(node_coords, element_indicies, alpha)
k_alpha_free = k_alpha[np.ix_(free_node_indicies, free_node_indicies)]

k_beta = assemble_k_beta(node_coords, element_indicies, beta)