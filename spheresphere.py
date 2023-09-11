from firedrake import *
import matplotlib.pyplot as plt

fig_src_mesh = plt.figure()
ax_src_mesh = fig_src_mesh.add_subplot(projection='3d')
ax_src_mesh.set_title("Source Mesh")

src_mesh = UnitCubedSphereMesh(1, name="src_sphere")

fig_dest_mesh = plt.figure()
ax_dest_mesh = fig_dest_mesh.add_subplot(projection='3d')
ax_dest_mesh.set_title("Destination Mesh")

dest_mesh = UnitIcosahedralSphereMesh(1, name="dest_sphere")

triplot(src_mesh, axes=ax_src_mesh)
fig_src_mesh.savefig("src_mesh.svg")
triplot(dest_mesh, axes=ax_dest_mesh)
fig_dest_mesh.savefig("dest_mesh.svg")

V_src = FunctionSpace(src_mesh, "S", 2)
V_dest = FunctionSpace(dest_mesh, "CG", 2)

x_src, y_src, z_src = SpatialCoordinate(src_mesh)
expr_src = x_src + y_src + z_src

fig_src_func = plt.figure()
ax_src_func = fig_src_func.add_subplot(projection='3d')
ax_src_func.set_title("Source Function")

f_src = Function(V_src).interpolate(expr_src)
trisurf(f_src, axes=ax_src_func)
fig_src_func.savefig("f_src.svg")

fig_dest_func = plt.figure()
ax_dest_func = fig_dest_func.add_subplot(projection='3d')
ax_dest_func.set_title("Destination Function")

f_dest = Function(V_dest).interpolate(f_src)
trisurf(f_dest, axes=ax_dest_func)
fig_dest_func.savefig("f_dest.svg")

V_dest_vec = VectorFunctionSpace(dest_mesh, V_dest.ufl_element())
f_dest_node_coords = interpolate(dest_mesh.coordinates, V_dest_vec)
dest_node_coords = f_dest_node_coords.dat.data_ro
ax_dest_mesh.scatter3D(dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c='b')
fig_dest_mesh.savefig("dest_mesh_w_src_nodes.svg")

ax_src_func.scatter3D(dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c='b')
fig_src_func.savefig("f_src_w_src_nodes.svg")

vom_dest_node_coords_in_src_mesh = VertexOnlyMesh(
                src_mesh,
                dest_node_coords,
                redundant=False,
                missing_points_behaviour=None,
            )

P0DG_vom = FunctionSpace(vom_dest_node_coords_in_src_mesh, "DG", 0)

f_vom = Function(P0DG_vom).interpolate(f_src)

fig_vom = plt.figure()
ax_vom = fig_vom.add_subplot(projection='3d')
ax_vom.set_title("Point Evaluations")

# plt.show()

