import platform
import os
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from typing import Literal
import numpy as np
from numpy import ndarray

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def voxelization(
    vertices: ndarray,
    faces: ndarray,
    grid: int=256,
    scale: float=1.0,
    backend: Literal['pyrender', 'trimesh']='trimesh',
):
    assert backend in ['pyrender', 'trimesh']
    if backend == 'pyrender':
        import pyrender
        znear = 0.05
        zfar = 4.0
        eye_dis = 2.0 # distance from eye to origin
        r_faces = np.stack([faces[:, 0], faces[:, 2], faces[:, 1]], axis=-1)
        # get zbuffers
        mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=vertices,
                    indices=np.concatenate([faces, r_faces]), # double sided
                    mode=pyrender.GLTF.TRIANGLES,
                )
            ]
        )
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        scene.add(mesh)

        camera = pyrender.OrthographicCamera(xmag=scale, ymag=scale, znear=znear, zfar=zfar)
        camera_poses = {}
        # coordinate:
        # see https://pyrender.readthedocs.io/en/latest/examples/cameras.html
        camera_poses['+z'] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, eye_dis],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at +z (bottom to top)
        camera_poses['-z'] = np.array([
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0,-1, -eye_dis],
            [ 0, 0, 0, 1],
        ], dtype=np.float32) # look at -z (top to bottom)
        camera_poses['+y'] = np.array([
            [1, 0, 0, 0],
            [0, 0,-1, -eye_dis],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at +y (because model is looking at -y)(front to back)
        camera_poses['-y'] = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, eye_dis],
            [0,-1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at -y (back to front)
        camera_poses['+x'] = np.array([
            [0, 0,-1, -eye_dis],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at +x (left to right)
        camera_poses['-x'] = np.array([
            [ 0, 0, 1, eye_dis],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.float32) # look at -x (righy to left)
        for name, pose in camera_poses.items():
            scene.add(camera, name=name, pose=pose)
        camera_nodes = [node for node in scene.get_nodes() if isinstance(node, pyrender.Node) and node.camera is not None]
        # if you are having issues with pyrender, change `backend` to 'trimesh' in configs/transform/<name>.yaml
        renderer = pyrender.OffscreenRenderer(viewport_width=grid, viewport_height=grid)

        i, j, k = np.indices((grid, grid, grid))
        grid_indices = np.stack((i.ravel(), j.ravel(), k.ravel()), axis=1, dtype=np.int64)
        grid_coords = np.stack((i.ravel(), j.ravel(), grid-1-k.ravel()), axis=1, dtype=np.float32) * 2 / grid - 1.0 + 1.0 / grid # every position is in the middle of the grid
        depths = {}
        for cam_node in camera_nodes:
            # a = time.time()
            scene.main_camera_node = cam_node
            name = cam_node.name
            proj_depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY | pyrender.constants.RenderFlags.OFFSCREEN)
            proj_depth[proj_depth<znear] = zfar
            proj_depth = znear + zfar - (znear * zfar) / proj_depth # back to origin
            depths[name] = proj_depth

        mask_z = -grid_coords[:, 2] + depths['+z'][grid-1-grid_indices[:, 1], grid_indices[:, 0]] <= eye_dis

        mask_z &= grid_coords[:, 2] + depths['-z'][grid-1-grid_indices[:, 1], grid-1-grid_indices[:, 0]] <= eye_dis

        mask_x = -grid_coords[:, 0] + depths['+x'][grid-1-grid_indices[:, 1], grid-1-grid_indices[:, 2]] <= eye_dis

        mask_x &= grid_coords[:, 0] + depths['-x'][grid-1-grid_indices[:, 1], grid_indices[:, 2]] <= eye_dis

        mask_y = -grid_coords[:, 1] + depths['+y'][grid_indices[:, 2], grid_indices[:, 0]] <= eye_dis

        mask_y &= grid_coords[:, 1] + depths['-y'][grid-1-grid_indices[:, 2], grid_indices[:, 0]] <= eye_dis

        mask = (mask_x & mask_y) | (mask_x & mask_z) | (mask_y & mask_z)
        grid_coords = grid_coords[mask]
        return grid_coords
    elif backend == 'trimesh':
        import trimesh
        # Create trimesh object
        mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces)
        voxel_size = 2 / grid

        # Voxelize the mesh using trimesh
        voxel_grid = mesh_tri.voxelized(pitch=voxel_size)

        # Get the voxel matrix
        voxel_matrix = voxel_grid.matrix

        # Fill interior voxels using projection method
        grids = np.indices(voxel_matrix.shape)
        x_coord = grids[0, ...]
        y_coord = grids[1, ...]
        z_coord = grids[2, ...]

        INF = 2147483647
        x_tmp = x_coord.copy()
        x_tmp[~voxel_matrix] = INF
        x_min = x_tmp.min(axis=0)
        x_tmp[~voxel_matrix] = -1
        x_max = x_tmp.max(axis=0)

        y_tmp = y_coord.copy()
        y_tmp[~voxel_matrix] = INF
        y_min = y_tmp.min(axis=1)
        y_tmp[~voxel_matrix] = -1
        y_max = y_tmp.max(axis=1)

        z_tmp = z_coord.copy()
        z_tmp[~voxel_matrix] = INF
        z_min = z_tmp.min(axis=2)
        z_tmp[~voxel_matrix] = -1
        z_max = z_tmp.max(axis=2)

        in_x = (x_coord >= x_min[None, :, :]) & (x_coord <= x_max[None, :, :])
        in_y = (y_coord >= y_min[:, None, :]) & (y_coord <= y_max[:, None, :])
        in_z = (z_coord >= z_min[:, :, None]) & (z_coord <= z_max[:, :, None])

        count = in_x.astype(int) + in_y.astype(int) + in_z.astype(int)
        fill_mask = count >= 2
        voxel_matrix = voxel_matrix | fill_mask

        # Get voxel coordinates
        x, y, z = np.where(voxel_matrix)
        grid_indices = np.stack([x, y, z], axis=1)

        # Convert to world coordinates
        # Trimesh uses a transform matrix to get the origin
        origin = voxel_grid.transform[:3, 3]  # Extract translation from 4x4 transform
        grid_coords = origin + (grid_indices + 0.5) * voxel_size
        return grid_coords

def voxel_skin(
    grid: int,
    grid_coords: ndarray, # (M, 3)
    joints: ndarray, # (J, 3)
    vertices: ndarray, # (N, 3)
    faces: ndarray, # (F, 3)
    alpha: float=0.5,
    link_dis: float=0.00001,
    grid_query: int=27,
    vertex_query: int=27,
    grid_weight: float=3.0,
    mode: str='square',
):

    # https://dl.acm.org/doi/pdf/10.1145/2485895.2485919
    assert mode in ['square', 'exp']
    J = joints.shape[0]
    M = grid_coords.shape[0]
    N = vertices.shape[0]

    grid_tree = cKDTree(grid_coords)
    vertex_tree = cKDTree(vertices)
    joint_tree = cKDTree(joints)

    # make combined vertices
    # 0   ~ N-1: mesh vertices
    # N   ~ N+M-1: grid vertices
    combined_vertices = np.concatenate([vertices, grid_coords], axis=0)

    # link adjacent grids
    dist, idx = grid_tree.query(grid_coords, grid_query) # 3*3*3
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < 2/grid*1.001)
    source_grid2grid = np.repeat(np.arange(M), grid_query-1)[mask.ravel()] + N
    to_grid2grid = idx[mask] + N
    weight_grid2grid = dist[mask] * grid_weight

    # link very close vertices
    dist, idx = vertex_tree.query(vertices, 4)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < link_dis*1.001)
    source_close = np.repeat(np.arange(N), 3)[mask.ravel()]
    to_close = idx[mask]
    weight_close = dist[mask]

    # link grids to mesh vertices
    dist, idx = vertex_tree.query(grid_coords, vertex_query)
    mask = (0 < dist) & (dist < 2/grid*1.001) # sqrt(3)
    source_grid2vertex = np.repeat(np.arange(M), vertex_query)[mask.ravel()] + N
    to_grid2vertex = idx[mask]
    weight_grid2vertex = dist[mask]

    # build combined vertices tree
    combined_tree = cKDTree(combined_vertices)
    # link joints to the neartest vertices
    _, joint_indices = combined_tree.query(joints)

    # build graph
    source_vertex2vertex = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]], axis=0)
    to_vertex2vertex = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]], axis=0)
    weight_vertex2vertex = np.sqrt(((vertices[source_vertex2vertex] - vertices[to_vertex2vertex])**2).sum(axis=-1))
    graph = csr_matrix(
        (np.concatenate([weight_close, weight_vertex2vertex, weight_grid2grid, weight_grid2vertex]),
        (
            np.concatenate([source_close, source_vertex2vertex, source_grid2grid, source_grid2vertex], axis=0),
            np.concatenate([to_close, to_vertex2vertex, to_grid2grid, to_grid2vertex], axis=0)),
        ),
        shape=(N+M, N+M),
    )

    # get shortest path (J, N+M)
    dist_matrix = shortest_path(graph, method='D', directed=False, indices=joint_indices)

    # (J, N)
    dis_vertex2joint = dist_matrix[:, :N]
    unreachable = np.isinf(dis_vertex2joint).all(axis=0)
    k = min(J, 3)
    dist, idx = joint_tree.query(vertices[unreachable], k)

    # make sure at least one value in dis is not inf
    unreachable_indices = np.where(unreachable)[0]
    row_indices = idx
    col_indices = np.repeat(unreachable_indices, k).reshape(-1, k)
    dis_vertex2joint[row_indices, col_indices] = dist

    finite_vals = dis_vertex2joint[np.isfinite(dis_vertex2joint)]
    max_dis = np.max(finite_vals)
    dis_vertex2joint = np.nan_to_num(dis_vertex2joint, nan=max_dis, posinf=max_dis, neginf=max_dis)
    dis_vertex2joint = np.maximum(dis_vertex2joint, 1e-6)
    # (J, N)
    if mode == 'exp':
        skin = np.exp(-dis_vertex2joint / max_dis * 20.0)
    elif mode == 'square':
        skin = (1./((1-alpha)*dis_vertex2joint + alpha*dis_vertex2joint**2))**2
    else:
        assert False, f'invalid mode: {mode}'
    skin = skin / skin.sum(axis=0)
    # (N, J)
    skin = skin.transpose()
    return skin
