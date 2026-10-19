import gmsh
import numpy as np
from scipy.constants import epsilon_0, mu_0

gmsh.initialize()

# gmsh.option.setNumber("General.Verbosity", 1)

gmsh.model.add("die")

f = 10e9
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / f
wavelength_grid = 2.5
pec_radius = wavelength / 2
abc_radius = wavelength_grid * wavelength

a, b = pec_radius, abc_radius
lc = 0.001

inner = gmsh.model.occ.addDisk(0, 0, 0, a, a)
outer = gmsh.model.occ.addDisk(0, 0, 0, b, b)
annulus, _ = gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
gmsh.model.occ.synchronize()

for dim, tag in gmsh.model.getEntities(2):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    if np.isclose(xmax, a, rtol=1e-3):
        gmsh.model.addPhysicalGroup(2, [tag], tag=2, name="dielectric")
    else:
        gmsh.model.addPhysicalGroup(2, [tag], tag=1, name="air")

for dim, tag in gmsh.model.getEntities(1):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    if np.isclose(xmax, b, rtol=1e-3):
        gmsh.model.addPhysicalGroup(1, [tag], tag=3, name="outer")

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write("mesh/die.msh")
gmsh.finalize()
