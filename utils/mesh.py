from typing import List

import numpy as np
import open3d as o3d
import torch
import trimesh
from trimesh import Trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh.triangles import barycentric_to_points, points_to_barycentric


def extract_joint_mesh(mesh_data, joint_labels: List[str], mask_threshold: float = 0.0, return_weight: bool = False):
    joint_idxs = [mesh_data['vgrp_label'].tolist().index(l) for l in joint_labels]
    weight = mesh_data['weight'].copy()
    for j_label in joint_labels:
        j_idx = mesh_data['vgrp_label'].tolist().index(j_label)
        dummy_j_indices = []
        while True:
            j_p_idx = mesh_data['vgrp_parents'][j_idx]
            if j_p_idx == -1 or j_p_idx in joint_idxs:
                break
            dummy_j_indices.append(j_p_idx)
            j_idx = j_p_idx
        if len(dummy_j_indices) > 0 and j_p_idx != -1:
            weight[:, j_p_idx] += np.sum(mesh_data['weight'][:, dummy_j_indices], axis=1)
    weight = weight[:, joint_idxs].copy()
    verts_mask = np.any(weight > mask_threshold, axis=1)

    ori_verts = mesh_data['verts'].copy()
    ori_faces = mesh_data['faces'].copy()
    faces_mask = np.all(verts_mask[ori_faces], axis=1)
    verts = ori_verts[verts_mask]
    ori2new_idx = np.where(verts_mask, np.cumsum(verts_mask, axis=0, dtype=int) - 1, -1)
    faces = ori2new_idx[ori_faces[faces_mask]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.double))
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if not return_weight:
        return mesh
    else:
        weight = weight[verts_mask]
        weight = weight / np.sum(weight, axis=1, keepdims=True)
        return mesh, weight


def load_path_from_axis(axis: np.ndarray, origin: np.ndarray, whole_axis: bool = False, color: np.ndarray = np.array([0, 0, 0, 255]), n_points: int = 100):
    if whole_axis:
        path = np.stack([origin + axis * (i - n_points//2) * 0.1 for i in range(n_points)], axis=0)
    else:
        path = np.stack([origin + axis * i * 0.1 for i in range(n_points)], axis=0)
    path = np.stack([path[:-1], path[1:]], axis=1)
    path =  trimesh.load_path(path)
    path.colors = np.tile(color, (len(path.entities), 1))
    return path


def axis_to_bary(mesh: Trimesh, axis: np.ndarray, origin: np.ndarray):
    intersector = RayMeshIntersector(mesh)
    directions = np.stack([-axis, axis], axis=0)
    origins = np.stack([origin, origin], axis=0)
    locations, index_ray, index_tri = intersector.intersects_location(origins, directions, multiple_hits=False)
    if len(index_ray) == 0:
        raise ValueError('Ray does not hit the mesh')
    l = locations[1] - locations[0]
    l = l / np.linalg.norm(l)
    bary = points_to_barycentric(mesh.vertices[mesh.faces[index_tri]], locations, method='cross')
    return bary, index_tri


def bary_to_axis(mesh: Trimesh, bary: np.ndarray, index_tri: np.ndarray):
    assert len(index_tri) == 2
    triangles = mesh.faces[index_tri] # (2, 3)
    triangle_vertices = mesh.vertices[triangles] # (2, 3, 3)
    points = barycentric_to_points(triangle_vertices, bary) # (2, 3)
    axis = points[1] - points[0]
    axis = axis / np.linalg.norm(axis)
    return axis


def bary_to_axis_batch(verts: np.ndarray, faces: np.ndarray, bary: np.ndarray, index_tri: np.ndarray):
    '''
    verts: (B, n_verts, 3)
    faces: (n_faces, 3)
    bary: (2, 2)
    index_tri: (2,)
    '''
    assert len(index_tri) == 2
    B = verts.shape[0]
    triangles = faces[index_tri] # (2, 3)
    triangle_vertices = verts[:, triangles] # (B, 2, 3, 3)
    triangle_vertices = triangle_vertices.reshape(-1, 3, 3) # (2*B, 3, 3)
    bary = np.tile(bary[np.newaxis], (B, 1, 1)) # (B, 2, 3)
    bary = bary.reshape(-1, 3) # (2*B, 3)
    points = barycentric_to_points(triangle_vertices, bary) # (2*B, 3)
    points = points.reshape(B, 2, 3) # (B, 2, 3)
    axis = points[:, 1] - points[:, 0]
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    return axis


def bary_to_points(triangles: torch.Tensor, bary: torch.Tensor):
    '''
    triangles: (*, n_tri, 3, 3)
    bary: (n_tri, 3)
    return: (*, n_tri, 3)
    '''
    assert bary.shape[-1] == 3 and triangles.shape[-2:] == (3, 3)
    points = torch.einsum('...ijk,...ij->...ik', triangles, bary)
    return points
