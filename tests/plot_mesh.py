import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio

mpl.use("QtAgg")

if len(sys.argv) < 2:
    print("Must pass in a mesh path")
    exit(0)

for path in sys.argv[1:]:
    mesh = meshio.read(path)
    nodes = mesh.points[:, :2]
    elements = mesh.get_cells_type("triangle")

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax.triplot(tri, lw=0.5, color="black")
    ax.scatter(nodes[:, 0], nodes[:, 1], 3, color="red")
    ax.set_aspect("equal")

plt.show()
