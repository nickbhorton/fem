import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)
gmsh.model.add("rectangle")

x0 = 0.8
y0 = 0.05

lc = 0.005

p1 = gmsh.model.geo.add_point(-x0, -y0, 0.0, lc)
p2 = gmsh.model.geo.add_point(-x0, y0, 0.0, lc)
p3 = gmsh.model.geo.add_point(x0, y0, 0.0, lc)
p4 = gmsh.model.geo.add_point(x0, -y0, 0.0, lc)

left_side = gmsh.model.geo.add_line(p1, p2)
top_side = gmsh.model.geo.add_line(p2, p3)
right_side = gmsh.model.geo.add_line(p3, p4)
bot_side = gmsh.model.geo.add_line(p4, p1)

loop = gmsh.model.geo.add_curve_loop([left_side, top_side, right_side, bot_side])
surface = gmsh.model.geo.add_plane_surface([loop])
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [left_side], name="left")
gmsh.model.addPhysicalGroup(1, [top_side], name="top")
gmsh.model.addPhysicalGroup(1, [right_side], name="right")
gmsh.model.addPhysicalGroup(1, [bot_side], name="bot")
gmsh.model.addPhysicalGroup(2, [surface], name="domain")

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write("rectangle.msh")
gmsh.finalize()
