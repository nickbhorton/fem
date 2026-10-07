import gmsh
import meshio
import numpy as np

gmsh.initialize()
gmsh.model.add("coax")

a, b = 0.5, 2.0
lc = 0.1  # target element size

# Two concentric circles
inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
outer = gmsh.model.occ.addDisk(0, 0, 0, b, b)
annulus, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
gmsh.model.occ.synchronize()

# Tag boundaries for Dirichlet conditions
curves = gmsh.model.getBoundary(annulus, oriented=False)
for dim, tag in curves:
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    r = max(abs(xmax), abs(xmin))
    name = "inner" if np.isclose(r, a) else "outer"
    gmsh.model.addPhysicalGroup(1, [tag], name=name)
gmsh.model.addPhysicalGroup(2, [annulus[0][1]], name="domain")

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write("coax.msh")
gmsh.finalize()

# plot mesh
mesh = meshio.read("coax.msh")
nodes = mesh.points[:, :2]
elements = mesh.get_cells_type("triangle")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
plt.triplot(tri, lw=0.5)
plt.gca().set_aspect("equal")
plt.show()