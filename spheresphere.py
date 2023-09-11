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

src_mesh_plt = triplot(src_mesh, axes=ax_src_mesh)
fig_src_mesh.savefig("1_src_mesh.svg")
dest_mesh_plt = triplot(dest_mesh, axes=ax_dest_mesh)
fig_dest_mesh.savefig("2_dest_mesh.svg")

V_src = FunctionSpace(src_mesh, "S", 2)
V_dest = FunctionSpace(dest_mesh, "CG", 2)

x_src, y_src, z_src = SpatialCoordinate(src_mesh)
expr_src = x_src + y_src + z_src

fig_src_func = plt.figure()
ax_src_func = fig_src_func.add_subplot(projection='3d')
ax_src_func.set_title("Source Function")

f_src = Function(V_src).interpolate(expr_src)
f_src_plot = trisurf(f_src, axes=ax_src_func)
fig_src_func.savefig("3_f_src.svg")

fig_dest_func = plt.figure()
ax_dest_func = fig_dest_func.add_subplot(projection='3d')
ax_dest_func.set_title("Destination Function")

f_dest = Function(V_dest).interpolate(f_src)
f_dest_plot = trisurf(f_dest, axes=ax_dest_func)
fig_dest_func.savefig("4_f_dest.svg")

ax_dest_mesh.set_title("Destination Mesh with Source Node Locations")
V_dest_vec = VectorFunctionSpace(dest_mesh, V_dest.ufl_element())
f_dest_node_coords = interpolate(dest_mesh.coordinates, V_dest_vec)
dest_node_coords = f_dest_node_coords.dat.data_ro
dest_node_coords_in_dest_mesh_plot = ax_dest_mesh.scatter3D(dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c='b')
fig_dest_mesh.savefig("5_dest_mesh_w_src_nodes.svg")

ax_src_func.set_title("Source Function with Source Node Locations")
dest_node_coords_in_f_src_plot = ax_src_func.scatter3D(dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c='b')
fig_src_func.savefig("6_f_src_w_src_nodes.svg")

vom_dest_node_coords_in_src_mesh = VertexOnlyMesh(
                src_mesh,
                dest_node_coords,
                redundant=False,
                missing_points_behaviour=None,
            )

P0DG_vom = FunctionSpace(vom_dest_node_coords_in_src_mesh, "DG", 0)

f_vom = Function(P0DG_vom).interpolate(f_src)

ax_src_func.set_title("Source Function with Point Evaluations at Source Node Locations")
dest_node_coords_in_f_src_plot.visible = False
vom_coords = vom_dest_node_coords_in_src_mesh.coordinates.dat.data_ro
f_vom_vals = f_vom.dat.data_ro
ax_src_func.scatter3D(vom_coords[:, 0], vom_coords[:, 1], vom_coords[:, 2], c=f_vom_vals)
fig_src_func.savefig("7_f_src_and_f_vom.svg")

# plt.show()

