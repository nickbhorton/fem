import gmsh
import numpy as np

gmsh.initialize()

gmsh.option.setNumber("General.Verbosity", 1)

gmsh.model.add("partial_coax2")

a, b = 0.5, 8.0
lc = 0.25  # target element size

angle = np.pi / 6

origin = gmsh.model.geo.add_point(0, 0.0, 0.0, lc)
p1 = gmsh.model.geo.add_point(a, 0.0, 0.0, lc)
p2 = gmsh.model.geo.add_point(b, 0.0, 0.0, lc)

p3 = gmsh.model.geo.add_point(a * np.cos(angle / 2), a * np.sin(angle / 2), 0, lc)
p4 = gmsh.model.geo.add_point(a * np.cos(angle), a * np.sin(angle), 0, lc)

p5 = gmsh.model.geo.add_point(b * np.cos(angle / 2), b * np.sin(angle / 2), 0, lc)
p6 = gmsh.model.geo.add_point(b * np.cos(angle), b * np.sin(angle), 0, lc)

inner_curve = gmsh.model.geo.add_circle_arc(p4, origin, p1)
outer_curve = gmsh.model.geo.add_circle_arc(p2, origin, p6)
zero_line = gmsh.model.geo.add_line(p1, p2)
angle_line = gmsh.model.geo.add_line(p6, p4)

loop = gmsh.model.geo.add_curve_loop([zero_line, outer_curve, angle_line, inner_curve])
surface = gmsh.model.geo.add_plane_surface([loop])
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [inner_curve], name="inner_d")
gmsh.model.addPhysicalGroup(1, [outer_curve], name="outer_d")
gmsh.model.addPhysicalGroup(1, [zero_line], name="zero_n")
gmsh.model.addPhysicalGroup(1, [angle_line], name="angle_n")
gmsh.model.addPhysicalGroup(2, [surface], name="domain")

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write("partial_coax2.msh")
gmsh.finalize()
