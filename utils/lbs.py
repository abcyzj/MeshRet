from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from utils.rotation_conversions import quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion


def rigid_transform(
    rot_mat: torch.Tensor, # (N, J, 3, 3)
    joint_locations: torch.Tensor, # (N, J, 3)
    parents: Union[torch.Tensor, np.ndarray] # (J)
):
    '''
    rot_mat: (N, J, 3, 3)
    joint_locations: (N, J, 3)
    parents: (J)

    Returns
    posed_joint_global: (N, J, 3)
    pose_transform_in_g: (N, J, 4, 4)
    '''
    j_in_p = joint_locations.clone().unsqueeze(-1) # (N, J, 3, 1) joint in parent frame
    j_in_p = torch.cat([j_in_p[:, [0]], j_in_p[:, 1:] - j_in_p[:, parents[1:]]], dim=1)

    jf_in_p = torch.cat([F.pad(rot_mat, [0, 0, 0, 1]), F.pad(j_in_p, [0, 0, 0, 1], value=1.0)], dim=3) # (N, J, 4, 4) joint frame in parent frame

    jf_in_g = [jf_in_p[:, 0]]
    for j_idx, p_idx in enumerate(parents):
        if p_idx == -1:
            assert j_idx == 0
            continue
        jf_in_g.append(jf_in_g[p_idx] @ jf_in_p[:, j_idx])
    jf_in_g = torch.stack(jf_in_g, dim=1) # (N, J, 4, 4) joint frame in global frame

    posed_joint_global = jf_in_g[:, :, :3, 3]

    N, J = joint_locations.shape[:2]
    gf_in_rest_j = torch.eye(4, 3, dtype=joint_locations.dtype, device=joint_locations.device).repeat(N, J, 1, 1) # (N, J, 4, 3)
    gf_in_rest_j = torch.cat([gf_in_rest_j, F.pad(-joint_locations.clone().unsqueeze(-1), [0, 0, 0, 1], value=1)], dim=3) # (N, J, 4, 4) global frame in rest pose joint frame, the orientation of rest joint frame is the same as the global frame

    pose_transform_in_g = jf_in_g @ gf_in_rest_j # (N, J, 4, 4) transform in global frame

    return posed_joint_global, pose_transform_in_g



def lbs(
    pose_mat: Union[torch.Tensor, np.ndarray],  # (*, J, 3, 3)
    joint_locations: Union[torch.Tensor, np.ndarray],  # (*, J, 3)
    parents: Union[torch.Tensor, np.ndarray],  # (J)
    verts: Union[torch.Tensor, np.ndarray],  # (*, V, 3)
    lbs_weights: Union[torch.Tensor, np.ndarray],  # (*, V, J)
    device: torch.device = torch.device('cpu')
):
    '''
    Return
    posed_joint_global: (*, J, 3)
    posed_v: (*, V, 3)
    '''
    if isinstance(pose_mat, np.ndarray):
        pose_mat = torch.from_numpy(pose_mat).to(device)
    if isinstance(joint_locations, np.ndarray):
        joint_locations = torch.from_numpy(joint_locations).to(device)
    if isinstance(verts, np.ndarray):
        verts = torch.from_numpy(verts).to(device)
    if isinstance(lbs_weights, np.ndarray):
        lbs_weights = torch.from_numpy(lbs_weights).to(device)

    ori_shape = pose_mat.shape[:-3]
    J = joint_locations.shape[-2]
    V = verts.shape[-2]
    pose_mat = pose_mat.reshape(-1, J, 3, 3)
    joint_locations = joint_locations.reshape(-1, J, 3)
    verts = verts.reshape(-1, V, 3)
    lbs_weights = lbs_weights.reshape(-1, V, J)

    posed_joint_global, pose_transform_in_g = rigid_transform(pose_mat, joint_locations, parents)

    weighted_transform_in_g = torch.einsum('nvj,njmk->nvmk', lbs_weights, pose_transform_in_g) # (N, V, 4, 4)
    v_homo = F.pad(verts, [0, 1], value=1.0).unsqueeze(-1) # (N, V, 4, 1)
    posed_v_homo = weighted_transform_in_g @ v_homo # (N, V, 4, 1)
    posed_v = posed_v_homo.squeeze(-1)[:, :, :3] # (N, V, 3)

    return posed_joint_global.reshape(ori_shape + (J, 3)), posed_v.reshape(ori_shape + (V, 3))


def sensor_lbs(
    pose_mat: torch.Tensor, # (*, J, 3, 3)
    joint_locations: torch.Tensor,  # (*, J, 3)
    parents: torch.Tensor,  # (J)
    sensor_locations: torch.Tensor, # (*, S, 3)
    sensor_tns: torch.Tensor, # (*, S, 3, 3)
    sensor_weights: torch.Tensor # (*, S, J)
):
    '''
    Return
    posed_joints_global: (*, J, 3)
    posed_sensor_locations: (*, S, 3)
    posed_sensor_tns: (*, S, 3, 3)
    '''
    ori_shape = pose_mat.shape[:-3]
    J = joint_locations.shape[-2]
    S = sensor_locations.shape[-2]
    pose_mat = pose_mat.reshape(-1, J, 3, 3)
    joint_locations = joint_locations.reshape(-1, J, 3)
    sensor_locations = sensor_locations.reshape(-1, S, 3)
    sensor_tns = sensor_tns.reshape(-1, S, 3, 3)
    sensor_weights = sensor_weights.reshape(-1, S, J)

    posed_joint_global, pose_transform_in_g = rigid_transform(pose_mat, joint_locations, parents)

    weighted_transform_in_g = torch.einsum('nsj,njmk->nsmk', sensor_weights, pose_transform_in_g) # (N, S, 4, 4)
    s_homo = F.pad(sensor_locations, [0, 1], value=1.0).unsqueeze(-1) # (N, S, 4, 1)
    posed_s_homo = weighted_transform_in_g @ s_homo # (N, S, 4, 1)
    posed_s = posed_s_homo.squeeze(-1)[..., :3]
    sensor_dir = sensor_tns.transpose(-1, -2).unsqueeze(-1) # (N, S, 3, 3, 1)
    weighted_transform_in_g = weighted_transform_in_g[..., :3, :3].unsqueeze(-3) # (N, S, 1, 3, 3)
    posed_sensor_dir = weighted_transform_in_g @ sensor_dir # (N, S, 3, 3, 1)
    posed_sensor_tns = posed_sensor_dir.squeeze(-1).transpose(-1, -2)

    return posed_joint_global.reshape(ori_shape + (J, 3)), posed_s.reshape(ori_shape + (S, 3)), posed_sensor_tns.reshape(ori_shape + (S, 3, 3)), pose_transform_in_g[..., :3, :3].reshape(ori_shape + (J, 3, 3))


class SkinnableMesh:
    def __init__(self, verts: Union[torch.Tensor, np.ndarray], joint_cors: Union[torch.Tensor, np.ndarray], parents: Union[torch.Tensor, np.ndarray], weight: Union[torch.Tensor, np.ndarray]):
        if isinstance(verts, np.ndarray):
            self.verts = torch.from_numpy(verts.copy())
        else:
            self.verts = verts.clone()
        if isinstance(joint_cors, np.ndarray):
            self.joint_locations = torch.from_numpy(joint_cors.copy())
        else:
            self.joint_locations = joint_cors.clone()
        if isinstance(parents, np.ndarray):
            self.parents = torch.from_numpy(parents.copy())
        else:
            self.parents = parents.clone()
        if isinstance(weight, np.ndarray):
            self.lbs_weights = torch.from_numpy(weight.copy())
        else:
            self.lbs_weights = weight.clone()

    def skin(self, pose: Union[torch.Tensor, np.ndarray], ret_joint: bool = False):
        '''
        pose: (*, J, 3, 3) or (*, J, 4)
        '''
        ret_numpy = False
        if isinstance(pose, np.ndarray):
            ret_numpy = True
            pose = torch.from_numpy(pose)

        if pose.shape[-1] == 4:
            pose = quaternion_to_matrix(pose)

        ori_shape = pose.shape[:-3]
        pose = pose.reshape((-1,) + pose.shape[-3:])

        joint_locations = self.joint_locations.repeat(ori_shape + (1, 1)).to(pose.device, pose.dtype)
        parents = self.parents.to(pose.device)
        verts = self.verts.repeat(ori_shape + (1, 1)).to(pose.device, pose.dtype)
        lbs_weights = self.lbs_weights.repeat(ori_shape + (1, 1)).to(pose.device, pose.dtype)
        posed_joints, posed_verts = lbs(pose, joint_locations, parents, verts, lbs_weights)

        posed_verts = posed_verts.reshape(ori_shape + posed_verts.shape[-2:])
        if ret_numpy:
            posed_joints = posed_joints.detach().cpu().numpy()
            posed_verts = posed_verts.detach().cpu().numpy()

        if ret_joint:
            return posed_verts, posed_joints
        else:
            return posed_verts


class SkinnableSensor:
    def __init__(
            self,
            sensor_locations: Union[np.ndarray, torch.Tensor],
            sensor_tns: Union[np.ndarray, torch.Tensor],
            joint_locations: Union[np.ndarray, torch.Tensor],
            parents: Union[np.ndarray, torch.Tensor],
            sensor_weights: Union[np.ndarray, torch.Tensor]
        ):
        if isinstance(sensor_locations, np.ndarray):
            sensor_locations = torch.from_numpy(sensor_locations)
        if isinstance(sensor_tns, np.ndarray):
            sensor_tns = torch.from_numpy(sensor_tns)
        if isinstance(joint_locations, np.ndarray):
            joint_locations = torch.from_numpy(joint_locations)
        if isinstance(sensor_weights, np.ndarray):
            sensor_weights = torch.from_numpy(sensor_weights)
        self.sensor_locations = sensor_locations
        self.sensor_tns = sensor_tns
        self.joint_locations = joint_locations
        self.parents = parents
        self.sensor_weights = sensor_weights

    def skin(self, pose: torch.Tensor, ret_joint: bool = False, ret_pose_global: bool = False):
        '''
        pose: (*, J, 3, 3) or (*, J, 4) or (*, J, 6)
        '''
        ori_pose = pose
        if pose.shape[-1] == 4:
            pose = quaternion_to_matrix(pose)
        elif pose.shape[-1] == 6:
            pose = rotation_6d_to_matrix(pose)

        ori_shape = pose.shape[:-3]
        pose = pose.reshape((-1,) + pose.shape[-3:])
        if len(self.joint_locations.shape) == 2:
            joint_locations = self.joint_locations.repeat(ori_shape + (1, 1)).to(pose.device).reshape((-1,) + self.joint_locations.shape[-2:])
        else:
            joint_locations = self.joint_locations.to(pose.device)
        if len(self.sensor_weights.shape) == 2:
            sensor_weights = self.sensor_weights.repeat(ori_shape + (1, 1)).to(pose.device).reshape((-1,) + self.sensor_weights.shape[-2:])
        else:
            sensor_weights = self.sensor_weights.to(pose.device)

        posed_joints, posed_sensors, posed_sensor_tns, pose_global = sensor_lbs(pose, joint_locations, self.parents.to(pose.device), self.sensor_locations.to(pose.device), self.sensor_tns.to(pose.device), sensor_weights)

        if ori_pose.shape[-1] == 4:
            pose_global = matrix_to_quaternion(pose_global)
        elif ori_pose.shape[-1] == 6:
            pose_global = matrix_to_rotation_6d(pose_global)

        if ret_joint:
            return posed_sensors.reshape(ori_shape + posed_sensors.shape[-2:]), posed_sensor_tns.reshape(ori_shape + posed_sensor_tns.shape[-3:]), posed_joints.reshape(ori_shape + posed_joints.shape[-2:])
        elif ret_pose_global:
            return posed_sensors.reshape(ori_shape + posed_sensors.shape[-2:]), posed_sensor_tns.reshape(ori_shape + posed_sensor_tns.shape[-3:]), pose_global.reshape(ori_shape + pose_global.shape[-2:])
        else:
            return posed_sensors.reshape(ori_shape + posed_sensors.shape[-2:]), posed_sensor_tns.reshape(ori_shape + posed_sensor_tns.shape[-3:])


class Rotation2MixamoVerts:
    def __init__(self, device):
        self.device = device

    def __call__(self, x, mask, pose_rep, verts, joint_loc, parents, lbs_weights):
        if mask is None:
            mask = torch.ones((x.shape[0],), dtype=bool, device=x.device)

        if pose_rep == 'rot6d':
            rotations = rotation_6d_to_matrix(x[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        skin_mesh = SkinnableMesh(verts, joint_loc, parents, lbs_weights)
        posed_verts = skin_mesh.skin(rotations)
        return posed_verts
