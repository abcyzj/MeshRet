from functools import cached_property
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pysdf import SDF
from pytorch3d.structures import Meshes
from trimesh import Trimesh
from trimesh.exchange.obj import export_obj
from trimesh.ray.ray_pyembree import RayMeshIntersector

from utils.lbs import SkinnableMesh
from utils.rotation_conversions import rotation_6d_to_matrix


class BaseBodyArmature:
    def __init__(self, joint_names: List[str], joint_loc: np.ndarray, joint_parents: np.ndarray, verts: np.ndarray):
        self._joint_names = tuple(joint_names)
        self._init_joint_map()
        self._joint_parents = joint_parents
        self._rest_verts = verts
        self._joint_loc = joint_loc
        self._init_joint_groups()
        self._joint_rotations = None
        self._root_locations = None
        self._sensor_cache = {}


    def _init_joint_map(self):
        if self._joint_names == self._standard_joint_names:
            self._joint_map = torch.arange(len(self._joint_names), dtype=torch.long)
        else:
            self._joint_map = torch.zeros([len(self._standard_joint_names)], dtype=torch.long)
            for j_idx, j_name in enumerate(self._standard_joint_names):
                if j_name in self._joint_names:
                    self._joint_map[j_idx] = self._joint_names.index(j_name)
                else:
                    self._joint_map[j_idx] = -1


    def _init_joint_groups(self):
        self._joint_group_indices = np.zeros([len(self._joint_groups), 10], dtype=np.int64)
        self._group_name2joint_indices = {}
        for group_idx, (group_name, joint_group) in enumerate(self._joint_groups.items()):
            for j_name in joint_group:
                self._joint_group_indices[group_idx, joint_group.index(j_name)] = self._joint_names.index(j_name)
            self._group_name2joint_indices[group_name] = self._joint_group_indices[group_idx, :len(joint_group)].tolist()

        self._joint_group_rest_t_axis = self._joint_loc[self._joint_group_indices][:, 1:] - self._joint_loc[self._joint_group_indices][:, :-1]
        self._joint_group_rest_t_axis = self._joint_group_rest_t_axis / (np.linalg.norm(self._joint_group_rest_t_axis, axis=-1, keepdims=True) + 1e-8)
        self._joint_group_bone_lengths = np.linalg.norm(self._joint_loc[self._joint_group_indices][:, 1:] - self._joint_loc[self._joint_group_indices][:, :-1], axis=-1)
        self._joint_group_t_matrix = {} # matrix to compute t-axis of each joint in group from joint rotations
        self._joint_rest_t_axes = np.zeros([len(self._joint_names), 3], dtype=np.float32) # t-axis of each joint in rest pose, not in group
        self._joint_bone_lengths = np.zeros([len(self._joint_names)], dtype=np.float32) # bone length of each joint in rest pose, not in group
        self._joint2group_idx = np.zeros([len(self._joint_names)], dtype=np.int64) # group index of each joint
        self._joint2in_group_idx = -np.ones([len(self._joint_names)], dtype=np.int64) # index of each joint in its group
        for group_idx, (group_name, joint_group) in enumerate(self._joint_groups.items()):
            self._joint_group_rest_t_axis[group_idx, len(joint_group) - 1] = self._joint_group_rest_t_axis[group_idx, len(joint_group) - 2] # duplicate the last t_axis for the tip joint
            self._joint_group_bone_lengths[group_idx, len(joint_group) - 1] = self._joint_group_bone_lengths[group_idx, len(joint_group) - 2] # duplicate the last bone length for the tip joint
            t_matrix = np.zeros([len(joint_group), len(self._joint_names)], dtype=np.float32)
            for j_idx, j_name in enumerate(joint_group[:-1]):
                t_matrix[j_idx, self._joint_names.index(j_name)] = -1.0
                t_matrix[j_idx, self._joint_names.index(joint_group[j_idx + 1])] = 1.0
                self._joint_rest_t_axes[self._joint_names.index(j_name)] = self._joint_group_rest_t_axis[group_idx, j_idx]
            t_matrix[-1] = t_matrix[-2]
            self._joint_group_t_matrix[group_name] = t_matrix
            for j_idx, j_name in enumerate(joint_group):
                self._joint2group_idx[self._joint_names.index(j_name)] = group_idx
                self._joint2in_group_idx[self._joint_names.index(j_name)] = j_idx
                self._joint_bone_lengths[self._joint_names.index(j_name)] = self._joint_group_bone_lengths[group_idx, j_idx]

        self._joint_group_t_matrix['all'] = np.concatenate([self._joint_group_t_matrix[group_name] for group_name in self._joint_groups.keys()], axis=0)
        self._joint_group_t_matrix['body'] = np.concatenate([self._joint_group_t_matrix[group_name] for group_name in self._joint_groups.keys() if 'Hand' not in group_name], axis=0)
        self._group_name2joint_indices['all'] = np.concatenate([self._group_name2joint_indices[group_name] for group_name in self._joint_groups.keys()], axis=0).tolist()
        self._group_name2joint_indices['body'] = np.concatenate([self._group_name2joint_indices[group_name] for group_name in self._joint_groups.keys() if 'Hand' not in group_name], axis=0).tolist()

    def get_joint_mesh(self, joint_names: List[str], ret_face_idx_map: bool = False, exclude_joint_names: List[str] = [], ret_tensor: bool = False):
        joint_names = [j_name.split(':')[-1] for j_name in joint_names]
        lbs_weights = self._lbs_weights.copy()
        joint_indices = [self._joint_names.index(joint_name) for joint_name in joint_names]
        exclude_joint_indices = [self._joint_names.index(joint_name) for joint_name in exclude_joint_names]
        verts = self.verts.detach().cpu().numpy()
        faces = self.faces.copy()

        vert_mask = (lbs_weights[:, joint_indices].sum(axis=-1) > 0.1) # (num_verts, )
        exclude_vert_mask = (lbs_weights[:, exclude_joint_indices].sum(axis=-1) > 0.1) # (num_verts, )
        vert_mask = vert_mask & (~exclude_vert_mask)
        face_mask = np.all(vert_mask[faces], axis=-1) # (num_faces, )
        vert_ori2new_idx = np.where(vert_mask, np.cumsum(vert_mask, axis=0) - 1, -1)
        part_verts = verts[:, :, vert_mask]
        part_faces = vert_ori2new_idx[faces[face_mask]]
        if ret_tensor:
            part_verts = torch.from_numpy(part_verts).to(self._joint_rotations.device)
            part_faces = torch.from_numpy(part_faces).to(self._joint_rotations.device)
        if ret_face_idx_map:
            face_ori2new_idx = np.where(face_mask, np.cumsum(face_mask, axis=0) - 1, -1)
            face_new2ori_idx = np.arange(face_mask.shape[0])[face_mask]
            return part_verts, part_faces, face_ori2new_idx, face_new2ori_idx
        else:
            return part_verts, part_faces


    @property
    def joint_rotations(self):
        if len(self._joint_names) != len(self._standard_joint_names):
            joint_map = self._joint_map.to(self._joint_rotations.device)
            return self._joint_rotations[:, :, joint_map]
        else:
            return self._joint_rotations


    @joint_rotations.setter
    def joint_rotations(self, rot: torch.Tensor):
        if rot.shape[-1] == 6:
            rot = rotation_6d_to_matrix(rot)
        assert (len(rot.shape) == 4 and rot.shape[-1] == 4) or (len(rot.shape) == 5 and rot.shape[-2:] == (3, 3))
        if len(self._joint_names) != len(self._standard_joint_names) and rot.shape[2] == len(self._standard_joint_names):
            B, T = rot.shape[:2]
            joint_map = self._joint_map.to(rot.device).unsqueeze(-1).unsqueeze(-1).expand(B, T, -1, 3, 3) # (B, T, num_standard_joints, 3, 3)
            self._joint_rotations = torch.scatter(torch.zeros(B, T, len(self._joint_names), 3, 3, dtype=rot.dtype, device=rot.device), 2, joint_map, rot) # (B, T, num_joints, 3, 3)
        else:
            self._joint_rotations = rot


    @property
    def root_locations(self):
        return self._root_locations


    @root_locations.setter
    def root_locations(self, loc: torch.Tensor):
        assert len(loc.shape) == 3 and loc.shape[-1] == 3
        self._root_locations = loc


    @property
    def verts(self):
        raise NotImplementedError()

    @property
    def faces(self):
        raise NotImplementedError()

    @property
    def joints(self):
        raise NotImplementedError()


    def fk(self):
        raise NotImplementedError()


    def _get_sensor_data(self, num_ring_per_bone: int, num_point_per_ring: int):
        if (num_ring_per_bone, num_point_per_ring) in self._sensor_cache:
            return self._sensor_cache[(num_ring_per_bone, num_point_per_ring)]

        lbs_weights = torch.from_numpy(self._lbs_weights.copy()).to(dtype=torch.float32) # (num_verts, num_joints)
        faces = torch.from_numpy(self.faces.astype(int)).to(dtype=torch.long)
        rest_verts = torch.from_numpy(self._rest_verts).to(dtype=torch.float32) # (num_verts, 3)
        sensor_data = {}
        for group_idx, (group_name, joint_group) in enumerate(self._joint_groups.items()):
            t_value = torch.linspace(0, len(joint_group) - 1, num_ring_per_bone * (len(joint_group) - 1) + 1, dtype=torch.float32)[:-1].contiguous() # (num_ring_per_bone, )
            phi = torch.linspace(0, 2*torch.pi, num_point_per_ring + 1, dtype=torch.float32)[:-1].contiguous() # (num_point_per_ring, )
            t_local = torch.stack(torch.meshgrid(t_value, phi, indexing='ij'), dim=-1) # (num_ring_per_bone, num_point_per_ring, 2)
            cos_sin_phi = torch.stack([torch.cos(t_local[..., 1]), torch.sin(t_local[..., 1])], dim=-1) # (num_ring_per_bone, num_point_per_ring, 2)
            t_local = torch.cat([t_local[..., :1], cos_sin_phi], dim=-1) # (num_ring_per_bone, num_point_per_ring, 3)
            index_tri, bary = self.tlocal_to_bary(t_local, torch.full(t_local.shape[:-1], group_idx, dtype=torch.long)) # (num_ring_per_bone, num_point_per_ring, 3)
            _, _, _, face_part2body_idx = self.get_joint_mesh(joint_group, ret_face_idx_map=True)
            sensor_mask = torch.isin(index_tri, torch.from_numpy(face_part2body_idx))
            index_tri = torch.clamp(index_tri, 0)
            sensor_weights = torch.einsum('...ij, ...i -> ...j', lbs_weights[faces[index_tri]], bary) # (num_ring_per_bone, num_point_per_ring, num_joints)
            sensor_weights = sensor_weights[..., self._group_name2joint_indices[group_name]] # (num_ring_per_bone, num_point_per_ring, num_joints_in_group)
            sensor_weights = sensor_weights / (sensor_weights.sum(dim=-1, keepdim=True) + 1e-8)
            triangles = rest_verts[faces[index_tri]] # (num_ring_per_bone, num_point_per_ring, 3, 3)
            sensor_locations = torch.einsum('...ij, ...i -> ...j', triangles, bary) # (num_ring_per_bone, num_point_per_ring, 3)
            sensor_data[group_name] = {
                'index_tri': index_tri,
                'bary': bary,
                'sensor_mask': sensor_mask,
                'sensor_weights': sensor_weights,
                'sensor_locations': sensor_locations,
                'sensor_t_local': t_local,
                'sensor_group_idx': torch.full(t_local.shape[:-1], group_idx, dtype=torch.long)
            }


        def pseudo_block_diag(matrices: List[torch.Tensor]):
            '''
            matrices: list of tensors with shape (M_n, N_n)
            return: (M_0+M_1+...+M_n, N_0+N_1+...+N_n)
            '''
            assert len(matrices) > 0
            if len(matrices) == 1:
                return matrices[0]
            prev_mat = pseudo_block_diag(matrices[:-1])
            cur_mat = matrices[-1]
            prev_shape = prev_mat.shape
            cur_shape = cur_mat.shape
            prev_mat = torch.cat([prev_mat, torch.zeros(prev_shape[0], cur_shape[1], dtype=prev_mat.dtype, device=prev_mat.device)], dim=-1)
            cur_mat = torch.cat([torch.zeros(cur_shape[0], prev_shape[1], dtype=cur_mat.dtype, device=cur_mat.device), cur_mat], dim=-1)
            return torch.cat([prev_mat, cur_mat], dim=0)


        all_index_tri = torch.cat([sensor_data[group_name]['index_tri'].reshape(-1) for group_name in self._joint_groups.keys()], dim=0) # (num_sensors, 3)
        all_bary = torch.cat([sensor_data[group_name]['bary'].reshape(-1, 3) for group_name in self._joint_groups.keys()], dim=0) # (num_sensors, 3)
        all_sensor_mask = torch.cat([sensor_data[group_name]['sensor_mask'].reshape(-1) for group_name in self._joint_groups.keys()], dim=0)
        all_sensor_weights = pseudo_block_diag([torch.flatten(sensor_data[group_name]['sensor_weights'], 0, -2) for group_name in self._joint_groups.keys()]) # (num_sensors, num_joints)
        all_sensor_t_local = torch.cat([sensor_data[group_name]['sensor_t_local'].reshape(-1, 3) for group_name in self._joint_groups.keys()], dim=0) # (num_sensors, 2)
        all_sensor_group_idx = torch.cat([sensor_data[group_name]['sensor_group_idx'].reshape(-1) for group_name in self._joint_groups.keys()], dim=0) # (num_sensors, )

        body_index_tri = torch.cat([sensor_data[group_name]['index_tri'].reshape(-1) for group_name in self._joint_groups.keys() if 'Hand' not in group_name], dim=0) # (num_body_sensors, 3)
        body_bary = torch.cat([sensor_data[group_name]['bary'].reshape(-1, 3) for group_name in self._joint_groups.keys() if 'Hand' not in group_name], dim=0)
        body_sensor_mask = torch.cat([sensor_data[group_name]['sensor_mask'].reshape(-1) for group_name in self._joint_groups.keys() if 'Hand' not in group_name], dim=0)
        body_sensor_weights = pseudo_block_diag([torch.flatten(sensor_data[group_name]['sensor_weights'], 0, -2) for group_name in self._joint_groups.keys() if 'Hand' not in group_name]) # (num_body_sensors, num_joints_in_group)
        body_sensor_t_local = torch.cat([sensor_data[group_name]['sensor_t_local'].reshape(-1, 3) for group_name in self._joint_groups.keys() if 'Hand' not in group_name], dim=0) # (num_body_sensors, 2)
        body_sensor_group_idx = torch.cat([sensor_data[group_name]['sensor_group_idx'].reshape(-1) for group_name in self._joint_groups.keys() if 'Hand' not in group_name], dim=0) # (num_body_sensors, )

        sensor_data['all'] = {
            'index_tri': all_index_tri,
            'bary': all_bary,
            'sensor_mask': all_sensor_mask,
            'sensor_weights': all_sensor_weights,
            'sensor_t_local': all_sensor_t_local,
            'sensor_group_idx': all_sensor_group_idx
        }
        sensor_data['body'] = {
            'index_tri': body_index_tri,
            'bary': body_bary,
            'sensor_mask': body_sensor_mask,
            'sensor_weights': body_sensor_weights,
            'sensor_t_local': body_sensor_t_local,
            'sensor_group_idx': body_sensor_group_idx
        }

        self._sensor_cache[(num_ring_per_bone, num_point_per_ring)] = sensor_data

        return sensor_data


    def sensors(self, num_ring_per_bone: int, num_point_per_ring: int, joint_group: str = 'all'):
        sensor_data = self._get_sensor_data(num_ring_per_bone, num_point_per_ring)
        verts, joints = self.fk() # (B, T, num_verts, 3), (B, T, num_joints, 3)
        B, T = verts.shape[:2]
        index_tri = sensor_data[joint_group]['index_tri'].to(verts.device) # (*, 3)
        bary = sensor_data[joint_group]['bary'].to(verts.device, verts.dtype) # (*, 3)
        sensor_mask = sensor_data[joint_group]['sensor_mask'].to(verts.device) # (*, 3)
        faces = torch.from_numpy(self.faces.astype(int)).to(device=verts.device, dtype=torch.long) # (num_faces, 3)
        triangles = verts[:, :, faces[index_tri]] # (B, T, *, 3, 3)
        points = torch.einsum('bt...ij, ...i -> bt...j', triangles, bary)
        root_loc = self.root_locations.reshape((B, T) + (1,) * (points.dim() - 3) + (3,)).to(points.device) # (B, T, 1, 3)
        points = points + root_loc # (B, T, *, 3)
        cur_meshes = Meshes(verts.reshape(B*T, -1, 3), faces.expand(B*T, -1, -1))
        cur_mesh_vertex_normals = cur_meshes.verts_normals_packed().reshape(B, T, -1, 3) # (B, T, num_verts, 3)
        sensor_normals = torch.einsum('bt...ij, ...i -> bt...j', cur_mesh_vertex_normals[:, :, faces[index_tri]], bary) # (B, T, *, 3)
        sensor_normals = F.normalize(sensor_normals, dim=-1)

        group_t_matrix = torch.from_numpy(self._joint_group_t_matrix[joint_group]).to(device=verts.device, dtype=verts.dtype) # (num_joints_in_group, num_joints)
        group_t_axes = torch.einsum('ij, btjk -> btik', group_t_matrix, joints) # (B, T, num_joints_in_group, 3)
        group_t_axes = F.normalize(group_t_axes, dim=-1)
        sensor_weights = sensor_data[joint_group]['sensor_weights'].to(verts.device, verts.dtype) # (*, num_joints_in_group)
        sensor_t = torch.einsum('btij, ...i -> bt...j', group_t_axes, sensor_weights) # (B, T, *, 3)
        sensor_t = sensor_t - torch.linalg.vecdot(sensor_t, sensor_normals, dim=-1).unsqueeze(-1) * sensor_normals # (B, T, *, 3)
        sensor_t = F.normalize(sensor_t, dim=-1)
        sensor_s = torch.linalg.cross(sensor_t, sensor_normals, dim=-1) # (B, T, *, 3)
        sensor_tns = torch.stack([sensor_t, sensor_normals, sensor_s], dim=-1) # (B, T, *, 3, 3)
        return points, sensor_tns, sensor_mask


    def sensor_mask(self, num_ring_per_bone: int, num_point_per_ring: int, joint_group: str = 'all'):
        sensor_data = self._get_sensor_data(num_ring_per_bone, num_point_per_ring)
        return sensor_data[joint_group]['sensor_mask']


    def bary_to_points(self, bary: torch.Tensor, index_tri: torch.Tensor):
        '''
        bary: (*, 3)
        index_tri: (*,)
        return: (B, T, *, 3)
        '''
        assert bary.shape[:-1] == index_tri.shape
        verts = self.verts.to(bary.device) # (B, T, num_verts, 3)
        bary = bary.to(verts.dtype)
        faces = torch.from_numpy(self.faces.astype(int)).to(device=bary.device, dtype=torch.long) # (num_faces, 3)
        triangles = verts[:, :, faces[index_tri]].to(bary.device) # (B, T, *, 3, 3)
        points = torch.einsum('bt...ij, ...i -> bt...j', triangles, bary) # (B, T, *, 3)
        return points


    def tlocal_to_bary(self, t_local: torch.Tensor, group_idx: torch.Tensor):
        '''
        t_local: (*, 3)
        group_idx: (*,)
        return: (*,), (*, 3)
        '''
        assert t_local.shape[-1] == 3 and t_local.shape[:-1] == group_idx.shape
        rest_mesh = Trimesh(self._rest_verts, self.faces, process=False)

        t_value = t_local[..., 0]
        j_idx_in_group = torch.clamp(torch.floor(t_value), 0).to(torch.long) # (*, ) j_idx_in_group can be negative, clamp to 0
        t_value = t_value - j_idx_in_group # (*, )
        cos_phi, sin_phi = t_local[..., 1], t_local[..., 2] # (*, )
        joints = torch.from_numpy(self._joint_loc).to(t_local.device) # (num_joints, 3)
        joint_group_indices = torch.from_numpy(self._joint_group_indices).to(t_local.device) # (num_joint_groups, 10)
        index_joints = torch.gather(joint_group_indices[group_idx], -1, j_idx_in_group.unsqueeze(-1)).squeeze(-1) # (*, )
        joint_group_bone_lengths = torch.from_numpy(self._joint_group_bone_lengths).to(t_local.device) # (num_joint_groups, 9)
        bone_lengths = torch.gather(joint_group_bone_lengths[group_idx], -1, j_idx_in_group.unsqueeze(-1)).squeeze(-1) # (*, )
        t_value = t_value * bone_lengths # (*, )
        joint_group_rest_t_axis = torch.from_numpy(self._joint_group_rest_t_axis).to(t_local.device) # (num_joint_groups, 9, 3)
        t_axis = torch.gather(joint_group_rest_t_axis[group_idx], -2, j_idx_in_group.unsqueeze(-1).unsqueeze(-1).expand(j_idx_in_group.shape + (1, 3))).squeeze(-2)
        b_axis = torch.from_numpy(self._b_axis).to(t_local.device).expand(t_value.shape + (3,)) # (*, 3)
        b_axis = b_axis - torch.sum(b_axis * t_axis, dim=-1, keepdim=True) * t_axis # (*, 3)
        b_axis = F.normalize(b_axis, dim=-1) # (*, 3)
        s_axis = torch.linalg.cross(t_axis, b_axis) # (*, 3)
        ray_origins = joints[index_joints] + t_value.unsqueeze(-1) * t_axis # (*, 3)
        ray_dirs = cos_phi.unsqueeze(-1) * b_axis +sin_phi.unsqueeze(-1) * s_axis # (*, 3)

        ori_shape = t_local.shape[:-1]
        intersector = RayMeshIntersector(rest_mesh)
        ray_origins = ray_origins.detach().reshape(-1, 3).cpu().numpy()
        ray_dirs = ray_dirs.detach().reshape(-1, 3).cpu().numpy()
        hit_locations, hit_index_ray, hit_index_tri = intersector.intersects_location(ray_origins, ray_dirs, multiple_hits=False) # Enable multiple_hits=True if needed
        hit_distances = np.linalg.norm(hit_locations - ray_origins[hit_index_ray], axis=-1)
        farmost_mask = np.zeros(hit_locations.shape[0], dtype=bool)
        face_part2body_idx = {}
        for group_name, joint_group in self._joint_groups.items():
            _, _, _, cur_face_part2body_idx = self.get_joint_mesh(joint_group, ret_face_idx_map=True)
            face_part2body_idx[group_name] = cur_face_part2body_idx
        for ori_idx in range(ray_origins.shape[0]):
            if np.sum(hit_index_ray == ori_idx) > 1:
                cur_hit_distances = hit_distances.copy()
                cur_hit_distances[hit_index_ray != ori_idx] = -1.0
                cur_group_name = list(self._joint_groups.keys())[group_idx.flatten()[ori_idx].item()]
                cur_hit_distances[~np.isin(hit_index_tri, face_part2body_idx[cur_group_name])] = -1.0
                farmost_mask[cur_hit_distances.argmax()] = True
            elif np.sum(hit_index_ray == ori_idx) == 1:
                farmost_mask[hit_index_ray == ori_idx] = True
        hit_locations, hit_index_ray, hit_index_tri = hit_locations[farmost_mask], hit_index_ray[farmost_mask], hit_index_tri[farmost_mask]
        hit_bary = trimesh.triangles.points_to_barycentric(rest_mesh.triangles[hit_index_tri], hit_locations)
        hit_bary = torch.from_numpy(hit_bary).to(device=t_local.device, dtype=t_local.dtype)
        hit_index_ray = torch.from_numpy(hit_index_ray).to(device=t_local.device, dtype=torch.long)
        hit_index_tri = torch.from_numpy(hit_index_tri).to(device=t_local.device, dtype=torch.long)
        bary = torch.zeros(ray_origins.shape, dtype=t_local.dtype, device=t_local.device)
        bary[hit_index_ray] = hit_bary
        bary = bary.reshape(ori_shape + (3,))
        index_tri = -torch.ones(ray_origins.shape[:-1], dtype=torch.long, device=t_local.device)
        index_tri[hit_index_ray] = hit_index_tri.to(torch.long)
        index_tri = index_tri.reshape(ori_shape)

        return index_tri, bary


    def bary_to_tlocal(self, index_tri: torch.Tensor, bary: torch.Tensor):
        '''
        index_tri: (*,)
        bary: (*, 3)
        return: (*, 2), (*,)
        '''
        assert index_tri.shape == bary.shape[:-1] and bary.shape[-1] == 3

        lbs_weights = torch.from_numpy(self._lbs_weights.copy()).to(device=bary.device, dtype=bary.dtype) # (num_verts, num_joints)
        not_in_group_j_indices = list(set(range(lbs_weights.shape[-1])) - set(self._group_name2joint_indices['all']))
        lbs_weights[:, not_in_group_j_indices] = 0.0
        faces = torch.from_numpy(self.faces.astype(int)).to(device=bary.device, dtype=torch.long) # (num_faces, 3)
        verts = torch.from_numpy(self._rest_verts).to(device=bary.device, dtype=bary.dtype) # (num_verts, 3)
        triangles = verts[faces[index_tri]] # (*, 3, 3)
        points = torch.einsum('...ij, ...i -> ...j', triangles, bary) # (*, 3)
        points_weights = torch.einsum('...ij, ...i -> ...j', lbs_weights[faces[index_tri]], bary) # (*, num_joints)
        joints = torch.from_numpy(self._joint_loc).to(device=bary.device, dtype=bary.dtype) # (num_joints, 3)
        points2joints = points_weights.argmax(dim=-1) # (*,)
        v = points - joints[points2joints] # (*, 3)
        t_axis = torch.from_numpy(self._joint_rest_t_axes).to(device=bary.device, dtype=bary.dtype) # (num_joints, 3)
        t_axis = t_axis[points2joints] # (*, 3)
        b_axis = torch.from_numpy(self._b_axis).to(device=bary.device, dtype=bary.dtype).expand(t_axis.shape[:-1] + (-1,)) # (*, 3)
        b_axis = b_axis - torch.sum(b_axis * t_axis, dim=-1, keepdim=True) * t_axis # (*, 3)
        b_axis = F.normalize(b_axis, dim=-1) # (*, 3)
        s_axis = torch.linalg.cross(t_axis, b_axis) # (*, 3)
        bone_lengths = torch.from_numpy(self._joint_bone_lengths).to(device=bary.device, dtype=bary.dtype)[points2joints] # (*,)
        joints2group_idx = torch.from_numpy(self._joint2group_idx).to(device=bary.device, dtype=torch.long) # (num_joints,)
        index_group = joints2group_idx[points2joints] # (*,)
        joints2in_group_idx = torch.from_numpy(self._joint2in_group_idx).to(device=bary.device, dtype=torch.long)
        joints2in_group_idx = joints2in_group_idx[points2joints] # (*,)
        t_value = torch.linalg.vecdot(v, t_axis, dim=-1) / bone_lengths # (*,)
        t_value = t_value + joints2in_group_idx # (*,)
        phi = torch.atan2(torch.linalg.vecdot(v, s_axis, dim=-1), torch.linalg.vecdot(v, b_axis, dim=-1)) # (*,)
        t_local = torch.stack([t_value, phi], dim=-1) # (*, 2)
        return t_local, index_group


    def relative_tns_coordinates(self, num_ring_per_bone: int = 5, num_point_per_ring: int = 10, joint_group: str = 'all'):
        sensor_locations, sensor_tns, sensor_mask = self.sensors(num_ring_per_bone, num_point_per_ring, joint_group)
        B, T = sensor_locations.shape[:2]
        sensor_locations = sensor_locations.reshape(B, T, -1, 3) # (B, T, n_sensor, 3)
        sensor_tns = sensor_tns.reshape(B, T, -1, 3, 3) # (B, T, n_sensor, 3, 3)
        sensor_mask = sensor_mask.reshape(-1) # (n_sensor)
        sensor_tns_inverted = sensor_tns.transpose(-1, -2).unsqueeze(3) # (B, T, n_sensor, 1, 3, 3)
        relative_sensor_coordinates = (sensor_locations.unsqueeze(2) - sensor_locations.unsqueeze(3)).unsqueeze(-1) # (B, T, n_sensor, n_sensor, 3, 1)
        relative_sensor_coordinates = torch.matmul(sensor_tns_inverted, relative_sensor_coordinates) # (B, T, n_sensor, n_sensor, 3, 1)
        relative_sensor_coordinates = relative_sensor_coordinates.squeeze(-1) # (B, T, n_sensor, n_sensor, 3)
        return relative_sensor_coordinates, sensor_mask


    @cached_property
    def limb_radius(self):
        num_ring_per_bone = 5
        num_point_per_ring = 10
        radiuses = {}
        sensor_data = self._get_sensor_data(num_ring_per_bone, num_point_per_ring)
        for group_name in self._joint_groups.keys():
            sensor_mask = sensor_data[group_name]['sensor_mask'] # (num_ring, num_point_per_ring)
            whole_ring_mask = torch.all(sensor_mask, dim=-1) # (num_ring, )
            sensor_locations = sensor_data[group_name]['sensor_locations'][whole_ring_mask] # (num_whole_ring, num_point_per_ring, 3)
            radiuses[group_name] = torch.linalg.norm(sensor_locations[:, :num_point_per_ring//2] - sensor_locations[:, num_point_per_ring//2:], dim=-1).mean().item()
        return radiuses


class MixamoBodyArmature(BaseBodyArmature):
    _joint_groups = {
        'LeftArm': ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandMiddle1'],
        'RightArm': ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandMiddle1'],
        'LeftLeg': ['LeftUpLeg', 'LeftLeg', 'LeftFoot'],
        'RightLeg': ['RightUpLeg', 'RightLeg', 'RightFoot'],
        'Torso': ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck'],
        'Head': ['Neck', 'Head', 'HeadTop_End'],
        'LeftHandMiddle': ['LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4'],
        'RightHandMiddle': ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4'],
        'LeftHandRing': ['LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4'],
        'RightHandRing': ['RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4'],
        'LeftHandPinky': ['LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4'],
        'RightHandPinky': ['RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4'],
        'LeftHandIndex': ['LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4'],
        'RightHandIndex': ['RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4'],
        'LeftHandThumb': ['LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4'],
        'RightHandThumb': ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4']
    }
    _standard_joint_names = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End')
    def __init__(self, joint_names: List[str], joint_parents: np.ndarray, verts: np.ndarray, faces: np.ndarray, lbs_weights: np.ndarray, joint_loc: np.ndarray):
        joint_names = [j_name.split(':')[-1] for j_name in joint_names]
        self._faces = faces
        self._lbs_weights = lbs_weights
        self._b_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32) # in rest pose, z-axis is the forward direction

        super().__init__(joint_names, joint_loc, joint_parents, verts)


    @property
    def verts(self):
        skin_mesh = SkinnableMesh(self._rest_verts, self._joint_loc, self._joint_parents, self._lbs_weights)
        posed_verts = skin_mesh.skin(self._joint_rotations) + self._root_locations.unsqueeze(2).to(self._joint_rotations.device)
        return posed_verts


    @property
    def faces(self):
        return self._faces


    def penetration_ratio(self):
        head_verts, head_faces = self.get_joint_mesh(self._joint_groups['Head'], ret_tensor=False)
        body_verts, body_faces = self.get_joint_mesh(self._joint_groups['Torso'], ret_tensor=False)
        left_arm_joint_names = self._joint_groups['LeftArm'][1:-1]
        for group_name in ['LeftHandMiddle', 'LeftHandRing', 'LeftHandPinky', 'LeftHandIndex', 'LeftHandThumb']:
            left_arm_joint_names += self._joint_groups[group_name]
        right_arm_joint_names = self._joint_groups['RightArm'][1:-1]
        for group_name in ['RightHandMiddle', 'RightHandRing', 'RightHandPinky', 'RightHandIndex', 'RightHandThumb']:
            right_arm_joint_names += self._joint_groups[group_name]
        left_arm_verts, left_arm_faces = self.get_joint_mesh(left_arm_joint_names, ret_tensor=False)
        right_arm_verts, right_arm_faces = self.get_joint_mesh(right_arm_joint_names, ret_tensor=False)
        left_leg_verts, left_leg_faces = self.get_joint_mesh(self._joint_groups['LeftLeg'], ret_tensor=False)
        right_leg_verts, right_leg_faces = self.get_joint_mesh(self._joint_groups['RightLeg'], ret_tensor=False)
        B, T = head_verts.shape[:2]
        head_prs, body_prs, leg_pr = [], [], []
        for b in range(B):
            h_verts = head_verts[b]
            b_verts = body_verts[b]
            arm_verts = np.concatenate([left_arm_verts[b], right_arm_verts[b]], axis=1)
            for t in range(T):
                head_sdf = SDF(h_verts[t], head_faces)
                body_sdf = SDF(b_verts[t], body_faces)
                left_leg_sdf = SDF(left_leg_verts[b, t], left_leg_faces)
                right_leg_sdf = SDF(right_leg_verts[b, t], right_leg_faces)
                head_contains = head_sdf.contains(arm_verts[t])
                body_contains = body_sdf.contains(arm_verts[t])
                head_prs.append(head_contains.mean())
                body_prs.append(body_contains.mean())
                left_leg_contains = left_leg_sdf.contains(right_leg_verts[b, t])
                right_leg_contains = right_leg_sdf.contains(left_leg_verts[b, t])
                leg_pr.append((left_leg_contains.mean() + right_leg_contains.mean()) / 2)
        return np.mean(head_prs), np.mean(body_prs), np.mean(leg_pr)



    @property
    def joints(self):
        skin_mesh = SkinnableMesh(self._rest_verts, self._joint_loc, self._joint_parents, self._lbs_weights)
        _, posed_joints = skin_mesh.skin(self._joint_rotations, ret_joint=True)
        posed_joints = posed_joints + self._root_locations.unsqueeze(2)
        return posed_joints


    def fk(self):
        skin_mesh = SkinnableMesh(self._rest_verts, self._joint_loc, self._joint_parents, self._lbs_weights)
        posed_verts, posed_joints = skin_mesh.skin(self._joint_rotations, ret_joint=True)
        posed_joints = posed_joints + self._root_locations.unsqueeze(2).to(posed_joints.device)
        return posed_verts, posed_joints


def export_armature(armature: BaseBodyArmature, p: str, factor: float = 1.0, f: int = 0):
    verts = armature.verts.detach()[0, f].cpu().numpy().squeeze()
    mesh = Trimesh(vertices=verts * factor, faces=armature.faces)
    with open(p, 'w') as f:
        f.write(export_obj(mesh))


def build_armature(motion_data: dict):
    joint_names = motion_data['vgrp_label']
    joint_names = [j_name.split(':')[-1] for j_name in joint_names]
    joint_parents = motion_data['vgrp_parents']
    verts = motion_data['verts'].squeeze()
    faces = motion_data['faces']
    lbs_weights = motion_data['lbs_weights']
    joint_loc = motion_data['vgrp_cors'].squeeze()
    armature = MixamoBodyArmature(joint_names, joint_parents, verts, faces, lbs_weights, joint_loc)
    return armature
