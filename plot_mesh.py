import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

mesh = meshio.read("coax.msh")
nodes = mesh.points[:, :2]
elements = mesh.get_cells_type("triangle")
print(elements.dtype)
np.savetxt("nodes.csv", nodes, delimiter=",")
np.savetxt("elements.csv", elements, fmt="%d", delimiter=",")

tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
plt.triplot(tri, lw=0.5)
plt.gca().set_aspect("equal")
plt.show()