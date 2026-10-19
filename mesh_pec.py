import gmsh
import numpy as np
from scipy.constants import epsilon_0, mu_0

gmsh.initialize()

# gmsh.option.setNumber("General.Verbosity", 1)

gmsh.model.add("pec")

f = 10e9
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
wavelength_grid = 2.5
pec_radius = wavelength / 2
abc_radius = wavelength_grid * wavelength

a, b = pec_radius, abc_radius
lc = 0.001  # target element size

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
gmsh.write("mesh/pec.msh")
gmsh.finalize()
