from argparse import ArgumentParser, Namespace
import pickle
from pathlib import Path
from copy import deepcopy

import torch
from tqdm import tqdm

from utils.body_armatures import build_armature


def get_limb_radius(points, mask, num_ring_per_bone, num_point_per_ring):
    points = points.reshape(num_ring_per_bone, num_point_per_ring, 3)
    mask = mask.reshape(num_ring_per_bone, num_point_per_ring)
    ring_mask = mask.all(dim=-1)
    radius = torch.norm(points[:, :num_point_per_ring//2] - points[:, num_point_per_ring//2:], dim=-1)[ring_mask].mean()
    return radius.item()


def get_sensor_data(motion_data, num_ring_per_bone, num_point_per_ring):
    armature = build_armature(motion_data)
    rest_pose = torch.zeros(1, 1, len(armature._joint_names), 4)
    rest_pose[:, :, :, 0] = 1.0
    root_locations = torch.zeros(1, 1, 3)
    armature.joint_rotations = rest_pose
    armature.root_locations = root_locations
    points, sensor_tns, sensor_mask = armature.sensors(num_ring_per_bone, num_point_per_ring, 'all')
    armature_sensor_data = armature._get_sensor_data(num_ring_per_bone, num_point_per_ring)['all']
    index_tri = armature_sensor_data['index_tri']
    bary = armature_sensor_data['bary']
    sensor_t_local = armature_sensor_data['sensor_t_local']
    sensor_group_idx = armature_sensor_data['sensor_group_idx']
    faces = torch.from_numpy(armature.faces.astype(int)).to(dtype=torch.long)
    lbs_weights = torch.from_numpy(armature._lbs_weights.copy()).to(dtype=torch.float32)
    sensor_weights = torch.einsum('...ij, ...i -> ...j', lbs_weights[faces[index_tri]], bary)
    sensor_tns[:, :, ~sensor_mask] = 0.0
    sensor_t_local[~sensor_mask] = -1.0
    sensor_group_idx[~sensor_mask] = 16 # 16 is the index of the 'none' group
    sensor_data = {}
    sensor_data['sensor_locations'] = points.squeeze(0, 1).detach().cpu().numpy()
    sensor_data['sensor_tns'] = sensor_tns.squeeze(0, 1).detach().cpu().numpy()
    sensor_data['sensor_mask'] = sensor_mask.detach().cpu().numpy()
    sensor_data['sensor_weights'] = sensor_weights.detach().cpu().numpy()
    sensor_data['sensor_t_local'] = sensor_t_local.detach().cpu().numpy()
    sensor_data['sensor_group_idx'] = sensor_group_idx.detach().cpu().numpy()
    num_point_per_bone = num_ring_per_bone * num_point_per_ring
    left_arm_radius = get_limb_radius(points[0, 0, num_point_per_bone*2:num_point_per_bone*3], sensor_mask[num_point_per_bone*2:num_point_per_bone*3], num_ring_per_bone, num_point_per_ring)
    left_hand_radius = get_limb_radius(points[0, 0, num_point_per_bone*3:num_point_per_bone*4], sensor_mask[num_point_per_bone*3:num_point_per_bone*4], num_ring_per_bone, num_point_per_ring)
    left_leg_radius = get_limb_radius(points[0, 0, num_point_per_bone*9:num_point_per_bone*10], sensor_mask[num_point_per_bone*9:num_point_per_bone*10], num_ring_per_bone, num_point_per_ring)
    right_arm_radius = get_limb_radius(points[0, 0, num_point_per_bone*6:num_point_per_bone*7], sensor_mask[num_point_per_bone*6:num_point_per_bone*7], num_ring_per_bone, num_point_per_ring)
    right_hand_radius = get_limb_radius(points[0, 0, num_point_per_bone*7:num_point_per_bone*8], sensor_mask[num_point_per_bone*7:num_point_per_bone*8], num_ring_per_bone, num_point_per_ring)
    right_leg_radius = get_limb_radius(points[0, 0, num_point_per_bone*11:num_point_per_bone*12], sensor_mask[num_point_per_bone*11:num_point_per_bone*12], num_ring_per_bone, num_point_per_ring)
    sensor_data['limb_radius'] = {
        'LeftArm': left_arm_radius,
        'LeftHand': left_hand_radius,
        'LeftLeg': left_leg_radius,
        'RightArm': right_arm_radius,
        'RightHand': right_hand_radius,
        'RightLeg': right_leg_radius
    }
    return sensor_data


def main(args: Namespace):
    correct_spine = (Path(args.data_dir).stem == 'beat2')
    for motion_f in tqdm(list(Path(args.data_dir).glob('*_motion.pkl'))):
        with open(motion_f, 'rb') as f:
            motion_data = pickle.load(f)
        tmp_motion_data = deepcopy(motion_data)
        sensor_configs = [(4, 4), (4, 8)]
        if correct_spine:
            tmp_motion_data['vgrp_cors'][..., :4, 0] = motion_data['vgrp_cors'][..., :4, 0].mean()
            tmp_motion_data['vgrp_cors'][..., :4, 2] = motion_data['vgrp_cors'][..., :4, 2].mean()
        motion_data['sensor_data'] = {}
        for (num_ring_per_bone, num_point_per_ring) in sensor_configs:
            motion_data['sensor_data'][(num_ring_per_bone, num_point_per_ring)] = get_sensor_data(tmp_motion_data, num_ring_per_bone, num_point_per_ring)
        with open(motion_f, 'wb') as f:
            pickle.dump(motion_data, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='artifact/datasets/scanret')
    args = parser.parse_args()
    main(args)
