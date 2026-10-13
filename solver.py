import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio

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


def assemble_k_alpha(nodes: np.ndarray, elements: np.ndarray):
    """
    This function assembles the k_alpha matrix. It does so without using python 
    loops which was somewhat difficult to achieve (thank you claude / gemini).
    I also learned that a coo_matrix is the fastest sparse matrix constructor.
    Then a compressed sparse row matrix (csr) is fast for row operations.
    And a compressed sparse column matrix (csc) is fast for column operations. 
    The k_alpha matrix is returned as a csr for matrix multiplication speed
    for boundary condition calculations but should be converted to a csc for
    spla.spsolve() (or at least that is what gemini says docs don't say that)

    nodes: np.ndarray of (number_of_nodes, 2), x and y coords of nodes
    elements: np.ndarray of (number_of_elements, 3) 3 node indexes into nodes.
    """
    x = nodes[elements, 0]
    y = nodes[elements, 1]
    bs = np.zeros((len(elements), 3))
    bs[:, 0] = y[:, 1] - y[:, 2]
    bs[:, 1] = y[:, 2] - y[:, 0]
    bs[:, 2] = y[:, 0] - y[:, 1]
    cs = np.zeros((len(elements), 3))
    cs[:, 0] = x[:, 2] - x[:, 1]
    cs[:, 1] = x[:, 0] - x[:, 2]
    cs[:, 2] = x[:, 1] - x[:, 0]
    two_delta = np.abs(bs[:, 0] * cs[:, 1] - bs[:, 1] * cs[:, 0])
    # (number_of_elements, 3, 3)
    # [b1 * b1, b1 * b2, b1 * b3]
    # [b2 * b1, b2 * b2, b2 * b3]
    # [b3 * b1, b3 * b2, b3 * b3]
    k = (bs[:, :, None] * bs[:, None, :] + cs[:, :, None] * cs[:, None, :]) / (
        2.0 * two_delta[:, None, None]
    )
    rows = np.repeat(elements, 3, axis=1)  # [1 1 1 2 2 2 3 3 3]
    cols = np.tile(elements, (1, 3))  # [1 2 3 1 2 3 1 2 3]
    return sp.coo_matrix(
        (k.ravel(), (rows.ravel(), cols.ravel())),
        shape=(nodes.shape[0], nodes.shape[0]),
    ).tocsr()


a = 0.5
b = 2.0
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

C_sim = epsilon_0 * phi @ (k_alpha @ phi) / (inner_potential - outer_potential)**2
C_exact = 2 * np.pi * epsilon_0 / np.log(b / a)
print(np.abs(C_sim - C_exact) / C_exact * 100, "% error")

# plot
fig, ax = plt.subplots()
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], domain)
pc = ax.tripcolor(triang, phi, shading="gouraud", cmap="inferno")
fig.colorbar(pc, ax=ax)
ax.set_aspect("equal")
plt.show()
