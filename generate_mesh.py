import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("coax")

a, b = 0.5, 2.0
lc = 0.25  # target element size

# Two concentric circles
inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
outer = gmsh.model.occ.addDisk(0, 0, 0, b, b)
annulus, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
gmsh.model.occ.synchronize()

# Tag boundaries for Dirichlet conditions
curves = gmsh.model.getBoundary(annulus, oriented=False)
for dim, tag in curves:
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    name = "inner" if np.isclose(xmax, a) else "outer"
    gmsh.model.addPhysicalGroup(1, [tag], name=name)
gmsh.model.addPhysicalGroup(2, [annulus[0][1]], name="domain")

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write("coax.msh")
gmsh.finalize()