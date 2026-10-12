import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio

mesh = meshio.read("coax.msh")

nodes = mesh.points[:, :2]
elements = mesh.get_cells_type("triangle")
element_tag = mesh.get_cell_data("gmsh:physical", "triangle")

lines = mesh.get_cells_type("line")
line_tag = mesh.get_cell_data("gmsh:physical", "line")

name_to_tag = {k: int(v[0]) for k, v in mesh.field_data.items()}

inner = lines[line_tag == name_to_tag["inner"]]
outer = lines[line_tag == name_to_tag["outer"]]
domain = elements[element_tag == name_to_tag["domain"]]


def assemble_k_alpha(k_alpha, nodes, elements):
    # k_alpha is a (number_of_nodes, number_of_nodes) matrix
    # elements is all the elements in the domain (number_of_elements, 3)
    x1 = nodes[elements[:, 0]][:, 0]
    x2 = nodes[elements[:, 1]][:, 0]
    x3 = nodes[elements[:, 2]][:, 0]
    y1 = nodes[elements[:, 0]][:, 1]
    y2 = nodes[elements[:, 1]][:, 1]
    y3 = nodes[elements[:, 2]][:, 1]
    # a1 = x2 * y3 - x3 * y2
    # a2 = x3 * y1 - x1 * y3
    # a3 = x1 * y2 - x2 * y1
    bs = np.zeros((3, len(elements)))
    bs[0] = y2 - y3
    bs[1] = y3 - y1
    bs[2] = y1 - y2
    cs = np.zeros((3, len(elements)))
    cs[0] = x3 - x2
    cs[1] = x1 - x3
    cs[2] = x2 - x1
    two_delta = bs[0] * cs[1] - bs[1] * cs[0]

    def ke(e, l, k):
        return (
            1.0
            / (2.0 * np.abs(two_delta[e]))
            * (bs[l, e] * bs[k, e] + cs[l, e] * cs[k, e])
        )

    for ei, ni in enumerate(elements):
        for i in range(3):
            for j in range(3):
                k_alpha[ni[i], ni[j]] += ke(ei, i, j)
    return k_alpha


matrix = np.zeros((len(nodes), len(nodes)))
k_alpha = assemble_k_alpha(matrix, nodes, domain)
nodes_inner = np.unique(inner.ravel())
nodes_outer = np.unique(outer.ravel())
nodes_dirichlet = np.concatenate([nodes_inner, nodes_outer])
nodes_free = np.setdiff1d(np.unique(domain.ravel()), nodes_dirichlet)

phi_dirichlet = np.concatenate(
    [1.0 * np.ones(len(nodes_inner)), 0.0 * np.ones(len(nodes_outer))]
)

k_alpha_free = k_alpha[np.ix_(nodes_free, nodes_free)]
b_dirichlet = -k_alpha[np.ix_(nodes_free, nodes_dirichlet)] @ phi_dirichlet

phi = np.zeros(len(nodes))
phi[nodes_dirichlet] = phi_dirichlet
phi[nodes_free] = np.linalg.solve(k_alpha_free, b_dirichlet)

# plot
fig, ax = plt.subplots()
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], domain)
pc = ax.tripcolor(triang, phi, shading="gouraud", cmap="inferno")
fig.colorbar(pc, ax=ax)
ax.set_aspect("equal")
plt.show()
