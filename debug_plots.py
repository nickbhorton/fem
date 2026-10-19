import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0, mu_0

from lib import fem_die_tez, fem_die_tmz, fem_pec_tez, fem_pec_tmz, fem_plot2

mesh_path_pec = "mesh/pec.msh"
mesh_pec = meshio.read(mesh_path_pec)
node_coords_pec = mesh_pec.points[:, :2]
element_indices_pec = mesh_pec.get_cells_type("triangle")

mesh_path_die = "mesh/die.msh"
mesh_die = meshio.read(mesh_path_die)
node_coords_die = mesh_die.points[:, :2]
element_indices_die = mesh_die.get_cells_type("triangle")

# plots
fem_plot2(node_coords_pec, element_indices_pec, np.abs(fem_pec_tmz(mesh_path_pec)))
fem_plot2(node_coords_pec, element_indices_pec, np.abs(fem_pec_tez(mesh_path_pec)))
fem_plot2(node_coords_die, element_indices_die, np.abs(fem_die_tmz(mesh_path_die)))
fem_plot2(node_coords_die, element_indices_die, np.abs(fem_die_tez(mesh_path_die)))

plt.show()
