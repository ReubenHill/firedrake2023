from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# 1
src_mesh = UnitCubedSphereMesh(1)

# 2
V_src = FunctionSpace(src_mesh, "S", 2)  # "S" only valid for quadrilateral cells
x_src, y_src, z_src = SpatialCoordinate(src_mesh)
expr_src = x_src + y_src + z_src
f_src = Function(V_src).interpolate(expr_src)

# 3
dest_mesh = UnitIcosahedralSphereMesh(1)

# 4 and 5
V_dest = FunctionSpace(dest_mesh, "CG", 2)
V_dest_vec = VectorFunctionSpace(dest_mesh, V_dest.ufl_element())
f_dest_node_coords = interpolate(dest_mesh.coordinates, V_dest_vec)
dest_node_coords = f_dest_node_coords.dat.data_ro

# 6
vom_dest_node_coords_in_src_mesh = VertexOnlyMesh(
    src_mesh,
    dest_node_coords,
    redundant=False,
    missing_points_behaviour=None,
)
P0DG_vom = FunctionSpace(vom_dest_node_coords_in_src_mesh, "DG", 0)
f_vom = Function(P0DG_vom).interpolate(f_src)

# 7
vom_i_o = vom_dest_node_coords_in_src_mesh.input_ordering  # New and important!
P0DG_vom_i_o = FunctionSpace(vom_i_o, "DG", 0)
f_vom_i_o = Function(P0DG_vom_i_o).interpolate(f_vom)  # PETSc SF reduce operation

# 8
f_dest_2 = Function(V_dest)
f_dest_2.dat.data_wo[:] = f_vom_i_o.dat.data_ro[:]  # Now safe

f_dest = Function(V_dest).interpolate(f_src)
assert np.array_equal(f_dest_2.dat.data_ro, f_vom_i_o.dat.data_ro)

cofunc_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
cofunc_src = Interpolator(TestFunction(V_src), V_dest).interpolate(
    cofunc_dest, transpose=True
)

if src_mesh.comm.size == 1:
    # Plot 1
    fig_src_mesh = plt.figure()
    ax_src_mesh = fig_src_mesh.add_subplot(projection="3d")
    ax_src_mesh.set_title("Source Mesh")
    src_mesh_plt = triplot(src_mesh, axes=ax_src_mesh)
    fig_src_mesh.savefig("1_src_mesh.svg")

    # Plot 2
    fig_src_func = plt.figure()
    ax_src_func = fig_src_func.add_subplot(projection="3d")
    ax_src_func.set_title("Source Function")
    f_src_plot = trisurf(f_src, axes=ax_src_func)
    fig_src_func.savefig("2_f_src.svg")

    # Plot 3
    fig_dest_mesh = plt.figure()
    ax_dest_mesh = fig_dest_mesh.add_subplot(projection="3d")
    ax_dest_mesh.set_title("Destination Mesh")
    dest_mesh_plt = triplot(dest_mesh, axes=ax_dest_mesh)
    fig_dest_mesh.savefig("3_dest_mesh.svg")

    # Plot 4
    ax_dest_mesh.set_title("Destination Mesh with Destination Node Locations")
    dest_node_coords_in_dest_mesh_plot = ax_dest_mesh.scatter3D(
        dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c="b"
    )
    fig_dest_mesh.savefig("4_dest_mesh_w_dest_nodes.svg")

    # Plot 5
    ax_src_func.set_title("Source Function with Destination Node Locations")
    dest_node_coords_in_f_src_plot = ax_src_func.scatter3D(
        dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c="b"
    )
    fig_src_func.savefig("5_f_src_w_dest_nodes.svg")

    # Plot 6
    ax_src_func.set_title(
        "Source Function with Point Evaluations \nat Destination Node Locations"
    )
    dest_node_coords_in_f_src_plot.visible = False
    vom_coords = vom_dest_node_coords_in_src_mesh.coordinates.dat.data_ro
    f_vom_vals = f_vom.dat.data_ro
    dest_node_coords_point_evals_in_f_src_plot = ax_src_func.scatter3D(
        vom_coords[:, 0], vom_coords[:, 1], vom_coords[:, 2], c=f_vom_vals
    )
    fig_src_func.savefig("6_f_src_and_f_vom.svg")

    # Plot 7
    ax_dest_mesh.set_title(
        "Destination Mesh with Point Evaluations \nat Destination Node Locations"
    )
    f_vom_i_o_vals = f_vom_i_o.dat.data_ro
    vom_i_o_coords = (
        vom_dest_node_coords_in_src_mesh.input_ordering.coordinates.dat.data_ro
    )
    dest_node_coords_in_dest_mesh_plot.visible = False
    dest_node_coords_point_evals_in_dest_mesh_plot = ax_dest_mesh.scatter3D(
        vom_i_o_coords[:, 0],
        vom_i_o_coords[:, 1],
        vom_i_o_coords[:, 2],
        c=f_vom_i_o_vals,
    )
    fig_dest_mesh.savefig("7_dest_mesh_w_f_vom_i_o.svg")

    # Plot 8
    fig_dest_func = plt.figure()
    ax_dest_func = fig_dest_func.add_subplot(projection="3d")
    ax_dest_func.set_title("Destination Function")
    f_dest_plot = trisurf(f_dest_2, axes=ax_dest_func)
    fig_dest_func.savefig("8_f_dest.svg")

elif src_mesh.comm.rank == 0:
    fig_dest_node_coords = plt.figure()
    ax_dest_node_coords = fig_dest_node_coords.add_subplot(projection="3d")
    ax_dest_node_coords.set_title("Destination Mesh Rank 0\nDestination Node Locations")
    dest_node_coords_plot = ax_dest_node_coords.scatter3D(
        dest_node_coords[:, 0], dest_node_coords[:, 1], dest_node_coords[:, 2], c="b"
    )
    fig_dest_node_coords.savefig("rank_0_dest_node_coords_dest_mesh.svg")

    fig_src_func = plt.figure()
    ax_src_func = fig_src_func.add_subplot(projection="3d")
    ax_src_func.set_title("Source Mesh Rank 0\nDestination Node Locations")
    vom_coords = vom_dest_node_coords_in_src_mesh.coordinates.dat.data_ro_with_halos
    dest_node_coords_in_f_src_plot = ax_src_func.scatter3D(
        vom_coords[:, 0], vom_coords[:, 1], vom_coords[:, 2], c="b"
    )
    fig_src_func.savefig("rank_0_dest_node_coords_src_mesh.svg")

    fig_vom_mesh = plt.figure()
    ax_vom_mesh = fig_vom_mesh.add_subplot(projection="3d")
    ax_vom_mesh.set_title("Point Evaluations Function\non Vertex-Only Mesh: Rank 0")
    f_vom_vals = f_vom.dat.data_ro_with_halos
    vom_mesh_plot = ax_vom_mesh.scatter3D(
        vom_coords[:, 0], vom_coords[:, 1], vom_coords[:, 2], c=f_vom_vals
    )
    fig_vom_mesh.savefig("rank_0_vom_func.svg")

    fig_vom_io_mesh = plt.figure()
    ax_vom_io_mesh = fig_vom_io_mesh.add_subplot(projection="3d")
    ax_vom_io_mesh.set_title(
        "Point Evaluations Function\non Input-Ordering Vertex-Only Mesh: Rank 0"
    )
    vom_io_coords = (
        vom_dest_node_coords_in_src_mesh.input_ordering.coordinates.dat.data_ro_with_halos
    )
    f_vom_i_o_vals = f_vom_i_o.dat.data_ro_with_halos
    vom_io_mesh_plot = ax_vom_io_mesh.scatter3D(
        vom_io_coords[:, 0], vom_io_coords[:, 1], vom_io_coords[:, 2], c=f_vom_i_o_vals
    )
    fig_vom_io_mesh.savefig("rank_0_vom_io_func.svg")
else:
    vom_io_coords = (
        vom_dest_node_coords_in_src_mesh.input_ordering.coordinates.dat.data_ro_with_halos
    )
    f_vom_i_o_vals = f_vom_i_o.dat.data_ro_with_halos  # for halo exchange
