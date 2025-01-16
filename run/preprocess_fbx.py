import pickle
import shlex
import subprocess
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np

from utils.lbs import SkinnableMesh

STANDARD_JOINT_NAMES = ('mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head', 'mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand', 'mixamorig:LeftHandThumb1', 'mixamorig:LeftHandThumb2', 'mixamorig:LeftHandThumb3', 'mixamorig:LeftHandIndex1', 'mixamorig:LeftHandIndex2', 'mixamorig:LeftHandIndex3', 'mixamorig:LeftHandMiddle1', 'mixamorig:LeftHandMiddle2', 'mixamorig:LeftHandMiddle3', 'mixamorig:LeftHandRing1', 'mixamorig:LeftHandRing2', 'mixamorig:LeftHandRing3', 'mixamorig:LeftHandPinky1', 'mixamorig:LeftHandPinky2', 'mixamorig:LeftHandPinky3', 'mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm', 'mixamorig:RightHand', 'mixamorig:RightHandThumb1', 'mixamorig:RightHandThumb2', 'mixamorig:RightHandThumb3', 'mixamorig:RightHandIndex1', 'mixamorig:RightHandIndex2', 'mixamorig:RightHandIndex3', 'mixamorig:RightHandMiddle1', 'mixamorig:RightHandMiddle2', 'mixamorig:RightHandMiddle3', 'mixamorig:RightHandRing1', 'mixamorig:RightHandRing2', 'mixamorig:RightHandRing3', 'mixamorig:RightHandPinky1', 'mixamorig:RightHandPinky2', 'mixamorig:RightHandPinky3', 'mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 'mixamorig:LeftFoot', 'mixamorig:LeftToeBase', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg', 'mixamorig:RightFoot', 'mixamorig:RightToeBase')
STANDARD_JOINT_PARENTS = (-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 9, 10, 11, 9, 13, 14, 9, 16, 17, 9, 19, 20, 9, 22, 23, 3, 25, 26, 27, 28, 29, 30, 28, 32, 33, 28, 35, 36, 28, 38, 39, 28, 41, 42, 0, 44, 45, 46, 0, 48, 49, 50)
EXTENDED_JOINT_NAMES = ('mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head', 'mixamorig:HeadTop_End', 'mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand', 'mixamorig:LeftHandThumb1', 'mixamorig:LeftHandThumb2', 'mixamorig:LeftHandThumb3', 'mixamorig:LeftHandThumb4', 'mixamorig:LeftHandIndex1', 'mixamorig:LeftHandIndex2', 'mixamorig:LeftHandIndex3', 'mixamorig:LeftHandIndex4', 'mixamorig:LeftHandMiddle1', 'mixamorig:LeftHandMiddle2', 'mixamorig:LeftHandMiddle3', 'mixamorig:LeftHandMiddle4', 'mixamorig:LeftHandRing1', 'mixamorig:LeftHandRing2', 'mixamorig:LeftHandRing3', 'mixamorig:LeftHandRing4', 'mixamorig:LeftHandPinky1', 'mixamorig:LeftHandPinky2', 'mixamorig:LeftHandPinky3', 'mixamorig:LeftHandPinky4', 'mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm', 'mixamorig:RightHand', 'mixamorig:RightHandThumb1', 'mixamorig:RightHandThumb2', 'mixamorig:RightHandThumb3', 'mixamorig:RightHandThumb4', 'mixamorig:RightHandIndex1', 'mixamorig:RightHandIndex2', 'mixamorig:RightHandIndex3', 'mixamorig:RightHandIndex4', 'mixamorig:RightHandMiddle1', 'mixamorig:RightHandMiddle2', 'mixamorig:RightHandMiddle3', 'mixamorig:RightHandMiddle4', 'mixamorig:RightHandRing1', 'mixamorig:RightHandRing2', 'mixamorig:RightHandRing3', 'mixamorig:RightHandRing4', 'mixamorig:RightHandPinky1', 'mixamorig:RightHandPinky2', 'mixamorig:RightHandPinky3', 'mixamorig:RightHandPinky4', 'mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 'mixamorig:LeftFoot', 'mixamorig:LeftToeBase', 'mixamorig:LeftToe_End', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg', 'mixamorig:RightFoot', 'mixamorig:RightToeBase', 'mixamorig:RightToe_End')
EXTENDED_JOINT_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 10, 11, 12, 13, 10, 15, 16, 17, 10, 19, 20, 21, 10, 23, 24, 25, 10, 27, 28, 29, 3, 31, 32, 33, 34, 35, 36, 37, 34, 39, 40, 41, 34, 43, 44, 45, 34, 47, 48, 49, 34, 51, 52, 53, 0, 55, 56, 57, 58, 0, 60, 61, 62, 63)


def extract_mesh_data(mesh_data: dict):
    if tuple(mesh_data['vgrp_label']) == EXTENDED_JOINT_NAMES:
        return mesh_data
    new_mesh_data = {k: v.copy() for k, v in mesh_data.items()}
    new_mesh_data['vgrp_label'] = np.asarray(EXTENDED_JOINT_NAMES)
    new_mesh_data['vgrp_parents'] = np.asarray(EXTENDED_JOINT_PARENTS)
    new_mesh_data['vgrp_rest_mats'] = np.tile(np.eye(3, dtype=mesh_data['vgrp_rest_mats'].dtype), [len(EXTENDED_JOINT_NAMES), 1, 1])
    new_mesh_data['vgrp_cors'] = np.zeros([len(EXTENDED_JOINT_NAMES), 3], dtype=mesh_data['vgrp_cors'].dtype)
    new_mesh_data['weight'] = np.zeros([mesh_data['weight'].shape[0], len(EXTENDED_JOINT_NAMES)], dtype=mesh_data['weight'].dtype)
    new_mesh_data['t_pose'] = np.tile(np.eye(3, dtype=mesh_data['t_pose'].dtype), [len(EXTENDED_JOINT_NAMES), 1, 1])
    for j, j_name in enumerate(EXTENDED_JOINT_NAMES):
        try:
            if isinstance(mesh_data['vgrp_label'], np.ndarray):
                old_j_idx = mesh_data['vgrp_label'].tolist().index(j_name)
            else:
                old_j_idx = mesh_data['vgrp_label'].index(j_name)
        except ValueError:
            continue
        new_mesh_data['vgrp_rest_mats'][j] = mesh_data['vgrp_rest_mats'][old_j_idx]
        new_mesh_data['vgrp_cors'][j] = mesh_data['vgrp_cors'][old_j_idx]
        new_mesh_data['weight'][:, j] = mesh_data['weight'][:, old_j_idx]
        new_mesh_data['t_pose'][j] = mesh_data['t_pose'][old_j_idx]
    for old_j, old_j_name in enumerate(mesh_data['vgrp_label']):
        if old_j_name in EXTENDED_JOINT_NAMES:
            continue
        old_j_ances = mesh_data['vgrp_parents'][old_j]
        while old_j_ances != -1 and mesh_data['vgrp_label'][old_j_ances] not in EXTENDED_JOINT_NAMES:
            old_j_ances = mesh_data['vgrp_parents'][old_j_ances]
        if old_j_ances == -1:
            continue
        new_j = EXTENDED_JOINT_NAMES.index(mesh_data['vgrp_label'][old_j_ances])
        new_mesh_data['weight'][:, new_j] += mesh_data['weight'][:, old_j]
    return new_mesh_data


def extract_motion_data(motion_data: dict):
    if tuple(motion_data['bone_names']) == EXTENDED_JOINT_NAMES:
        return motion_data

    old_motion_data = {k: v.copy() for k, v in motion_data.items()}
    new_motion_data = {k: v.copy() for k, v in motion_data.items()}
    new_motion_data['bone_names'] = np.asarray(EXTENDED_JOINT_NAMES)
    new_motion_data['bone_rest_mats'] = np.tile(np.eye(4, dtype=old_motion_data['bone_rest_mats'].dtype), [len(EXTENDED_JOINT_NAMES), 1, 1])
    new_motion_data['bone_pose_mats'] = np.tile(np.eye(4, dtype=old_motion_data['bone_pose_mats'].dtype), [old_motion_data['bone_pose_mats'].shape[0], len(EXTENDED_JOINT_NAMES), 1, 1])
    for j, j_name in enumerate(EXTENDED_JOINT_NAMES):
        try:
            old_j_idx = old_motion_data['bone_names'].tolist().index(j_name)
        except ValueError:
            continue
        new_motion_data['bone_rest_mats'][j] = old_motion_data['bone_rest_mats'][old_j_idx]
        new_motion_data['bone_pose_mats'][:, j] = old_motion_data['bone_pose_mats'][:, old_j_idx]
    return new_motion_data


def fbx2mesh(fbx_path: Path, output_path: Path):
    if output_path.exists():
        print('Skip processing mesh for {}'.format(fbx_path))
        return
    cmd = 'blender -b -P blender_scripts/fbx2mesh.py -- --input {} --output {}'.format(fbx_path.as_posix(), output_path.as_posix())
    cmd = shlex.split(cmd)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Finish processing mesh for {}'.format(fbx_path))


def fix_mesh_joint_labels(mesh_data):
    mesh_data = {k: mesh_data[k] for k in mesh_data.keys()}
    joint_names = mesh_data['vgrp_label'].copy().tolist()
    if 'mixamorig:Spine3' in joint_names:
        joint_names[joint_names.index('mixamorig:Spine')] = 'mixamorig:SpineNull'
        joint_names[joint_names.index('mixamorig:Spine1')] = 'mixamorig:Spine'
        joint_names[joint_names.index('mixamorig:Spine2')] = 'mixamorig:Spine1'
        joint_names[joint_names.index('mixamorig:Spine3')] = 'mixamorig:Spine2'

    if 'mixamorig:HeadEnd' in joint_names:
        joint_names[joint_names.index('mixamorig:HeadEnd')] = 'mixamorig:HeadTop_End'

    mesh_data['vgrp_label'] = joint_names
    return mesh_data

def fbx2motion(fbx_path_list: List[Path], mesh_path: Path, output_path: Path):
    if output_path.exists():
        print('Skip processing motion for {}'.format(mesh_path))
        return
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        out_motion_list = []
        with Pool(32) as p:
            for fbx_path in fbx_path_list:
                out_motion_path = temp_dir.joinpath(fbx_path.stem + '.npz')
                out_motion_list.append(out_motion_path)
                cmd = 'blender -b -P blender_scripts/fbx2motion.py -- --input_fbx {} --output {}'.format(shlex.quote(fbx_path.as_posix()), shlex.quote(out_motion_path.as_posix()))
                cmd = shlex.split(cmd)
                p.apply_async(subprocess.run, args=(cmd,), kwds={'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL})
            if len(out_motion_list) > 0:
                p.close()
                p.join()

        mesh_data = np.load(mesh_path)
        mesh_data = fix_mesh_joint_labels(mesh_data)
        mesh_data = extract_mesh_data(mesh_data)
        verts = mesh_data['verts']
        vgrp_cors = mesh_data['vgrp_cors']
        vgrp_parents = mesh_data['vgrp_parents']
        vgrp_label = mesh_data['vgrp_label']
        if isinstance(vgrp_label, np.ndarray):
            vgrp_label = vgrp_label.tolist()
        vgrp_rest_mats = mesh_data['vgrp_rest_mats']
        lbs_weights = mesh_data['weight']
        t_pose = mesh_data['t_pose'][np.newaxis, ...]
        t_pose_global = t_pose.copy()
        for j in range(len(vgrp_label)):
            j_parent = vgrp_parents[j]
            if j_parent == -1:
                continue
            t_pose_global[:, j] = t_pose_global[:, j_parent] @ t_pose[:, j]
        t_pose_global_inv = t_pose_global.transpose([0, 1, 3, 2])
        skin_mesh = SkinnableMesh(verts, vgrp_cors, vgrp_parents, lbs_weights)
        new_verts, new_vgrp_cors = skin_mesh.skin(t_pose, ret_joint=True)
        skin_mesh = SkinnableMesh(new_verts.squeeze(0), new_vgrp_cors.squeeze(0), vgrp_parents, lbs_weights)

        motion_poses = {}
        motion_translations = {}
        for out_motion_path in out_motion_list:
            if not out_motion_path.exists():
                print('Cannot find motion file for {}'.format(out_motion_path))
                continue
            motion_data = np.load(out_motion_path)
            motion_data = extract_motion_data(motion_data)
            motion_id = out_motion_path.stem
            joint_names = motion_data['bone_names'].tolist()
            try:
                assert joint_names == vgrp_label
            except AssertionError:
                print('Joint names mismatch for {} and {}'.format(mesh_path, out_motion_path))
                continue
            joint_rotations = motion_data['bone_pose_mats'][:, :, :3, :3]
            root_locations = motion_data['bone_pose_mats'][:, 0:1, :3, 3]
            pose = np.tile(np.eye(3, dtype=joint_rotations.dtype), [joint_rotations.shape[0], len(vgrp_parents), 1, 1])
            for j in range(len(vgrp_label)):
                j_rest_mat_inv = vgrp_rest_mats[j][np.newaxis, ...].transpose([0, 2, 1])
                j_pose_mat = joint_rotations[:, j]
                j_t_pose_inv = t_pose[:, j].transpose([0, 2, 1])
                if vgrp_parents[j] == -1:
                    pose[:, j] = j_pose_mat @ j_rest_mat_inv @ j_t_pose_inv
                else:
                    j_parent = vgrp_parents[j]
                    parent_rest_mat = vgrp_rest_mats[j_parent][np.newaxis, ...]
                    parent_pose_mat_inv = joint_rotations[:, j_parent].transpose([0, 2, 1])
                    pose[:, j] = parent_rest_mat @ parent_pose_mat_inv @ j_pose_mat @ j_rest_mat_inv
                    pose[:, j] = t_pose_global[:, j_parent] @ pose[:, j] @ t_pose_global_inv[:, j]
            motion_poses[motion_id] = pose
            motion_translations[motion_id] = root_locations

        with open(output_path, 'wb') as f:
            pickle.dump({
                'verts': verts,
                'faces': mesh_data['faces'],
                'vgrp_label': vgrp_label,
                'vgrp_cors': vgrp_cors,
                'vgrp_parents': vgrp_parents,
                'lbs_weights': lbs_weights,
                'motion_poses': motion_poses,
                'motion_translations': motion_translations
            }, f)
        print('Finish processing motion for {}'.format(mesh_path))


def main(args: Namespace):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    fbx2mesh_args = []
    fbx2motion_args = []
    for character_dir in input_dir.iterdir():
        if not character_dir.is_dir():
            continue
        character_name = character_dir.name
        mesh_fbx_path = character_dir / f'{character_name}.fbx'
        if not mesh_fbx_path.exists():
            print(f'Cannot find mesh file for {character_name}')
            continue
        mesh_output_path = output_dir / f'{character_name}.npz'
        fbx2mesh_args.append((mesh_fbx_path, mesh_output_path))
        motion_fbx_paths = []
        for f in character_dir.glob('*.fbx'):
            if f.stem != character_name:
                motion_fbx_paths.append(f)
        motion_output_path = output_dir / f'{character_name}_motion.pkl'
        fbx2motion_args.append((motion_fbx_paths, mesh_output_path, motion_output_path))


    with Pool(args.num_process) as p:
        p.starmap(fbx2mesh, fbx2mesh_args)

    for args in fbx2motion_args:
        fbx2motion(*args)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_process', type=int, default=16)
    args = parser.parse_args()
    main(args)
