from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Start with a simple field exactly represented in the function space over
# the unit square domain.
m = UnitSquareMesh(2, 2)
V = FunctionSpace(m, "CG", 2)
x, y = SpatialCoordinate(m)
f = Function(V).interpolate(x * y)

# We create a 1D mesh immersed 2D from (0, 0) to (1, 1) which we call "line".
# Note that it only has 1 cell
cells = np.asarray([[0, 1]])
vertex_coords = np.asarray([[0.0, 0.0], [1.0, 1.0]])
plex = mesh.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
line = mesh.Mesh(plex, dim=2)
x, y = SpatialCoordinate(line)
V_line = FunctionSpace(line, "CG", 2)
f_line = Function(V_line).interpolate(x * y)
assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)  # for sanity
f_line.zero()
assert np.isclose(assemble(f_line * dx), 0)  # sanity again

# We want to calculate the line integral of f along it. To do this we
# create a function space on the line mesh...
V_line = FunctionSpace(line, "CG", 2)

# ... and interpolate our function f onto it.
f_line = interpolate(f, V_line)

# The integral of f along the line is then a simple form expression which
# we assemble:
assemble(f_line * dx)  # this outputs sqrt(2) / 3
assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)

# plot
fig_f = plt.figure()
ax_f = fig_f.add_subplot(projection="3d")
ax_f.set_title("Function on Unit Square Mesh with Line to Integrate Along")
ax_f.view_init(elev=30, azim=0, roll=0)
f_plot = trisurf(f, axes=ax_f)
line_plot = ax_f.plot(
    line.coordinates.dat.data_ro[:, 0],
    line.coordinates.dat.data_ro[:, 1],
    np.zeros(len(line.coordinates.dat.data_ro)),
    "r-",
)
fig_f.savefig("line_integral.svg")


fig_m = plt.figure()
ax_m = fig_m.add_subplot()
ax_m.set_title("Unit Square Mesh with Line to Integrate Along")
m_plot = triplot(m, axes=ax_m)
line_plot = ax_m.plot(
    line.coordinates.dat.data_ro[:, 0], line.coordinates.dat.data_ro[:, 1], "r-"
)
fig_m.savefig("line_integral_meshes.svg")
