import json
import pickle
from functools import cache
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.body_armatures import MixamoBodyArmature
from utils.rotation_conversions import matrix_to_rotation_6d


def mret_collate(batch: list):
    root_translation = torch.stack([torch.as_tensor(b['root_translation']) for b in batch]).to(torch.float32)
    motion = torch.stack([torch.as_tensor(b['inp']) for b in batch]).to(torch.float32)
    root_translation[:, 1:] = root_translation[:, 1:] - root_translation[:, :1]
    B, T = motion.shape[:2]
    root_translation[:, 1:] = (motion[:, :1, 0].transpose(-1, -2) @ root_translation[:, 1:].reshape(B, T-1, 3, 1)).reshape(B, T-1, 1, 3)
    root_translation[:, 0] = 0.0
    motion[:, 1:, 0] = motion[:, :1, 0].transpose(-1, -2) @ motion[:, 1:, 0]
    motion[:, 0, 0] = torch.eye(3, dtype=motion.dtype).unsqueeze(0).repeat(motion.shape[0], 1, 1)
    motion = matrix_to_rotation_6d(motion)
    mask = torch.stack([torch.as_tensor(b['mask']) for b in batch]).unsqueeze(-1).unsqueeze(-1) # (B, T, 1, 1)
    is_intra = torch.cat([b['is_intra'] for b in batch], dim=0)

    src_static = character_collate([b['src_static'] for b in batch])
    tgt_static = character_collate([b['tgt_static'] for b in batch])

    if batch[0]['gt'] is not None:
        gt = torch.stack([torch.as_tensor(b['gt']) for b in batch]).to(torch.float32)
        gt[:, 1:, 0] = gt[:, :1, 0].transpose(-1, -2) @ gt[:, 1:, 0]
        gt[:, 0, 0] = torch.eye(3, dtype=gt.dtype).unsqueeze(0).repeat(gt.shape[0], 1, 1)
        gt = matrix_to_rotation_6d(gt)
    else:
        gt = None

    meta = [b['meta'] for b in batch]

    return {
        'y': {
            'mask': mask,
            'is_intra': is_intra,
            'src_static': src_static,
            'tgt_static': tgt_static
        },
        'x': motion,
        'gt': gt,
        'root_translation': root_translation,
        'meta': meta
    }


def character_collate(batch: list):
    joint_locations = torch.stack([torch.as_tensor(b['joint_locations']).squeeze(0) for b in batch]).to(torch.float32)
    body_heights = joint_locations[:, 6, 1] - joint_locations[:, 58, 1] # from HeadTop_End to LeftToeBase
    normalized_joint_locations = (joint_locations - joint_locations[:, 0:1]) / body_heights.unsqueeze(-1).unsqueeze(-1)
    parents = torch.as_tensor(batch[0]['parents']) # Parents are the same for all samples in a batch
    sensor_locations = torch.stack([torch.as_tensor(b['sensor_locations']) for b in batch]).to(torch.float32)
    normalized_sensor_locations = (sensor_locations - joint_locations[:, 0:1]) / body_heights.unsqueeze(-1).unsqueeze(-1)
    sensor_tns = torch.stack([torch.as_tensor(b['sensor_tns']) for b in batch]).to(torch.float32)
    sensor_mask = torch.stack([torch.as_tensor(b['sensor_mask']) for b in batch])
    sensor_weights = torch.stack([torch.as_tensor(b['sensor_weights']) for b in batch]).to(torch.float32)
    sensor_t_local = torch.stack([torch.as_tensor(b['sensor_t_local']) for b in batch]).to(torch.float32)
    sensor_group_idx = torch.stack([torch.as_tensor(b['sensor_group_idx']) for b in batch])
    limb_radius = {k: torch.stack([torch.as_tensor(b['limb_radius'][k]) for b in batch]) for k in batch[0]['limb_radius'].keys()}
    normalized_limb_radius = {k: v / body_heights for k, v in limb_radius.items()}
    verts = [torch.as_tensor(b['verts']).squeeze(0).to(torch.float32) for b in batch]
    faces = [torch.as_tensor(b['faces'].astype(np.int32)) for b in batch]
    lbs_weights = [torch.as_tensor(b['lbs_weights']).to(torch.float32) for b in batch]
    num_body_sensor = sensor_mask.shape[-1] // 48 * 18
    cond = {
        'joint_locations': joint_locations,
        'body_heights': body_heights,
        'parents': parents,
        'normalized_joint_locations': normalized_joint_locations,
        'normalized_sensor_locations': normalized_sensor_locations,
        'normalized_body_sensor_locations': normalized_sensor_locations[:, :num_body_sensor],
        'body_sensor_locations': sensor_locations[:, :num_body_sensor],
        'body_sensor_tns': sensor_tns[:, :num_body_sensor],
        'body_sensor_mask': sensor_mask[:, :num_body_sensor],
        'body_sensor_weights': sensor_weights[:, :num_body_sensor],
        'body_sensor_t_local': sensor_t_local[:, :num_body_sensor],
        'body_sensor_group_idx': sensor_group_idx[:, :num_body_sensor],
        'sensor_locations': sensor_locations,
        'sensor_tns': sensor_tns,
        'sensor_mask': sensor_mask,
        'sensor_weights': sensor_weights,
        'sensor_t_local': sensor_t_local,
        'sensor_group_idx': sensor_group_idx,
        'limb_radius': limb_radius,
        'normalized_limb_radius': normalized_limb_radius,
        'verts': verts,
        'faces': faces,
        'lbs_weights': lbs_weights
    }
    return cond


class MRet(Dataset):
    def __init__(self,
                 data_dir: Union[str, List[str]] = 'artifact/datasets/scanret',
                 seq_len: int = 60,
                 sample_stride: int = 10,
                 split: str = 'sc+am',
                 pose_rep: str = 'rot6d',
                 is_training: bool = True,
                 num_ring_per_bone: int = 4,
                 num_point_per_ring: int = 8,
                 only_contact: bool = False,
                 test_penetration: bool = False,
                 paired_gt: bool = False
                 ):
        super().__init__()
        if isinstance(data_dir, str):
            self.data_dir = [Path(data_dir)]
        elif isinstance(data_dir, list):
            self.data_dir = [Path(d) for d in data_dir]
        else:
            raise ValueError(f'Invalid data_dir type: {type(data_dir)}')
        self.seq_len = seq_len
        self.sample_stride = sample_stride
        self.split = split
        self.pose_rep = pose_rep
        self.is_training = is_training
        self.num_ring_per_bone = num_ring_per_bone
        self.num_point_per_ring = num_point_per_ring
        self.only_contact = only_contact
        self.test_penetration = test_penetration
        self.paired_gt = paired_gt
        self._load_data_keys()


    def _load_data_keys(self):
        data_files = []
        split_files = []
        for d in self.data_dir:
            data_files.extend(d.glob('*_motion.pkl'))
            split_files.append(d / 'split.json')
        rng = np.random.RandomState(522)
        character_names = set([f.stem[:-7] for f in data_files])
        char2file = {f.stem[:-7]: f for f in data_files}
        motion_ids = set()
        self._motion_data = {}
        for c_name in character_names:
            with open(char2file[c_name], 'rb') as f:
                motion_data = pickle.load(f)
            self._motion_data[c_name] = motion_data
            motion_ids.update(motion_data['motion_poses'].keys())
        unseen_chars = set()
        unseen_motions = set()
        for f in split_files:
            with open(f, 'r') as f:
                split_data = json.load(f)
            unseen_chars.update(split_data['uc'])
            unseen_motions.update(split_data['um'])
        char_split, motion_split = self.split.split('+')
        if char_split == 'sc':
            character_names = list(character_names - unseen_chars)
        elif char_split == 'uc':
            character_names = list(unseen_chars)
        elif char_split == 'ac':
            character_names = list(character_names)
        else:
            raise ValueError(f'Invalid character split: {char_split}')
        if motion_split == 'sm':
            motion_ids = list(motion_ids - unseen_motions)
        elif motion_split == 'um':
            motion_ids = list(unseen_motions)
        elif motion_split == 'am':
            motion_ids = list(motion_ids)
        else:
            raise ValueError(f'Invalid motion split: {motion_split}')

        character_names = sorted(character_names)
        char2dir_name = {f.stem[:-7]: f.parent.stem for f in data_files}
        dir_name2count = {d.stem: 0 for d in self.data_dir}
        for c_name in character_names:
            motion_data = self._motion_data[c_name]
            match self.pose_rep:
                case 'rot6d':
                    for i in motion_data['motion_poses'].keys():
                        motion_data['motion_poses'][i] = torch.from_numpy(motion_data['motion_poses'][i]) # FIXME This is a temporary fix
                case 'rotmat':
                    for i in motion_data['motion_poses'].keys():
                        motion_data['motion_poses'][i] = torch.from_numpy(motion_data['motion_poses'][i])
                case _:
                    raise ValueError(f'Invalid pose representation: {self.pose_rep}')

            dir_name2count[char2dir_name[c_name]] += sum([cur_clip.shape[0] for cur_clip in motion_data['motion_poses'].values()])

        intra_data_keys = []
        max_frames_per_dir = max(dir_name2count.values())
        super_sampled_c_names = []
        for c_name in character_names:
            # Super sampling to make sure each directory has roughly same number of clips
            motion_data = self._motion_data[c_name]
            dir_name = char2dir_name[c_name]
            dir_num_frames = dir_name2count[dir_name]
            super_sampling_ratio = max(1, max_frames_per_dir // dir_num_frames)
            cur_motion_ids = [i for i in motion_data['motion_poses'].keys() if i in motion_ids]
            for motion_id in cur_motion_ids:
                data_len = motion_data['motion_poses'][motion_id].shape[0]
                for i in range(0, data_len - self.seq_len + 1, self.sample_stride):
                    intra_data_keys.extend([(c_name, motion_id, c_name, i)] * super_sampling_ratio)
            super_sampled_c_names.extend([c_name] * super_sampling_ratio)

        c_name2tgt_c_names = {} # For each character, we sample from all other characters except itself
        for c_name in character_names:
            c_name2tgt_c_names[c_name] = [c for c in super_sampled_c_names if c != c_name]
        cross_data_keys = []
        for c_name in super_sampled_c_names:
            motion_data = self._motion_data[c_name]
            cur_motion_ids = [i for i in motion_data['motion_poses'].keys() if i in motion_ids]
            for motion_id in cur_motion_ids:
                data_len = motion_data['motion_poses'][motion_id].shape[0]

                for i in range(0, data_len - self.seq_len + 1, self.sample_stride):
                    tgt_c_name = rng.choice(c_name2tgt_c_names[c_name])
                    cross_data_keys.append((c_name, motion_id, tgt_c_name, i))

        if self.is_training:
            self.data_keys = intra_data_keys + cross_data_keys
            rng.shuffle(self.data_keys)
        else:
            self.data_keys = cross_data_keys

        # for penetration test
        if not self.is_training and self.test_penetration:
            scanret_ids = [31,1,45,4,43,9,41,47,72,3,83,12]
            self.data_keys = list(filter(lambda x: x[1] in scanret_ids, self.data_keys))
            if not self.paired_gt:
                self.data_keys = list(filter(lambda x: x[2][:2] != 'QY', self.data_keys))
            else:
                take_num = 64*3
                if len(self.data_keys) > take_num:
                    indices = rng.choice(len(self.data_keys), take_num, replace=False).tolist()
                    self.data_keys = [self.data_keys[i] for i in indices]


    def __len__(self):
        return len(self.data_keys)


    def __getitem__(self, idx):
        src_c_name, m_id, tgt_c_name, f_start = self.data_keys[idx]
        f_end = f_start + self.seq_len
        src_motion_data, tgt_motion_data = self._motion_data[src_c_name], self._motion_data[tgt_c_name]
        src_sensor_data = src_motion_data['sensor_data'][self.num_ring_per_bone, self.num_point_per_ring]
        tgt_sensor_data = tgt_motion_data['sensor_data'][self.num_ring_per_bone, self.num_point_per_ring]
        pose = src_motion_data['motion_poses'][m_id]
        pose = pose[f_start:f_end]
        pose = torch.as_tensor(pose)
        translation = src_motion_data['motion_translations'][m_id][f_start:f_end]
        translation = torch.as_tensor(translation)
        mask = np.ones((self.seq_len, ), dtype=bool)
        if pose.shape[0] < self.seq_len:
            pose = torch.cat([pose, torch.zeros((self.seq_len - pose.shape[0],) + pose.shape[1:], dtype=pose.dtype)], dim=0)
            mask[f_end:] = False
        if self.paired_gt:
            pose_gt = tgt_motion_data['motion_poses'][m_id]
            pose_gt = pose_gt[f_start:f_end]
            pose_gt = torch.as_tensor(pose_gt)
            if pose_gt.shape[0] < self.seq_len:
                pose_gt = torch.cat([pose_gt, torch.zeros((self.seq_len - pose_gt.shape[0],) + pose_gt.shape[1:], dtype=pose_gt.dtype)], dim=0)
        else:
            pose_gt = None
        ret = {
            'meta': (src_c_name, m_id, tgt_c_name, f_start),
            'inp': pose,
            'gt': pose_gt,
            'root_translation': translation,
            'mask': mask,
            'is_intra': torch.ones(1, dtype=torch.bool) if src_c_name == tgt_c_name else torch.zeros(1, dtype=torch.bool),
            'src_static': {
                'joint_locations': src_motion_data['vgrp_cors'],
                'parents': src_motion_data['vgrp_parents'],
                'sensor_locations': src_sensor_data['sensor_locations'],
                'sensor_tns': src_sensor_data['sensor_tns'],
                'sensor_mask': src_sensor_data['sensor_mask'],
                'sensor_weights': src_sensor_data['sensor_weights'],
                'sensor_t_local': src_sensor_data['sensor_t_local'],
                'sensor_group_idx': src_sensor_data['sensor_group_idx'],
                'limb_radius': src_sensor_data['limb_radius'],
                'verts': src_motion_data['verts'],
                'faces': src_motion_data['faces'],
                'lbs_weights': src_motion_data['lbs_weights']
            },
            'tgt_static': {
                'joint_locations': tgt_motion_data['vgrp_cors'],
                'parents': tgt_motion_data['vgrp_parents'],
                'sensor_locations': tgt_sensor_data['sensor_locations'],
                'sensor_tns': tgt_sensor_data['sensor_tns'],
                'sensor_mask': tgt_sensor_data['sensor_mask'],
                'sensor_weights': tgt_sensor_data['sensor_weights'],
                'sensor_t_local': tgt_sensor_data['sensor_t_local'],
                'sensor_group_idx': tgt_sensor_data['sensor_group_idx'],
                'limb_radius': tgt_sensor_data['limb_radius'],
                'verts': tgt_motion_data['verts'],
                'faces': tgt_motion_data['faces'],
                'lbs_weights': tgt_motion_data['lbs_weights']
            }
        }

        return ret


class MRetDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: Union[str, List[str]] = 'artifact/datasets/scanret',
                 batch_size: int = 64,
                 seq_len: int = 60,
                 sample_stride: int = 10,
                 num_ring_per_bone: int = 4,
                 num_point_per_ring: int = 8,
                 split: str = 'sc+am',
                 pose_rep: str = 'rot6d',
                 num_workers: int = 8,
                 only_contact: bool = False,
                 test_penetration: bool = False,
                 paired_gt: bool = False
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sample_stride = sample_stride
        self.num_ring_per_bone = num_ring_per_bone
        self.num_point_per_ring = num_point_per_ring
        self.split = split
        self.pose_rep = pose_rep
        self.num_workers = num_workers
        self.only_contact = only_contact
        self.test_penetration = test_penetration
        self.paired_gt = paired_gt

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_data = MRet(data_dir=self.data_dir, seq_len=self.seq_len, sample_stride=self.sample_stride, split=self.split, pose_rep=self.pose_rep, is_training=True, num_ring_per_bone=self.num_ring_per_bone, num_point_per_ring=self.num_point_per_ring, only_contact=self.only_contact)
        if stage == 'test' or stage is None:
            self.test_data = MRet(data_dir=self.data_dir, seq_len=self.seq_len, sample_stride=self.sample_stride, split=self.split, pose_rep=self.pose_rep, is_training=False, num_ring_per_bone=self.num_ring_per_bone, num_point_per_ring=self.num_point_per_ring, only_contact=self.only_contact, test_penetration=self.test_penetration, paired_gt=self.paired_gt)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=mret_collate, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=mret_collate, num_workers=self.num_workers)


def get_relative_coords(group_pairs: list, posed_sensor_locations: torch.Tensor, posed_sensor_tns: torch.Tensor, sensor_t_local: torch.Tensor, sensor_group_emb: torch.Tensor, sensor_mask: torch.Tensor, num_max: int = 4, sparse_mode: str = 'both'):
    '''
    posed_sensor_locations: (B, T, N, 3)
    posed_sensor_tns: (B, T, N, 3, 3)
    sensor_t_local: (B, N, 2)
    sensor_group_emb: (B, N, E)
    sensor_mask: (B, N)
    '''
    def relative_coords_by_group(group_bone_ranges: dict, obs_groups: Tuple[str], tgt_groups: Tuple[str]):
        '''
        group_bone_ranges: dict of {group_name: slice}
        obs_groups: list of group names
        tgt_groups: list of group names
        '''
        T = posed_sensor_locations.shape[1]
        obs_bone_ranges = [group_bone_ranges[g] for g in obs_groups]
        tgt_bone_ranges = [group_bone_ranges[g] for g in tgt_groups]
        obs_coords = torch.cat([posed_sensor_locations[:, :, r, :] for r in obs_bone_ranges], dim=-2)
        tgt_coords = torch.cat([posed_sensor_locations[:, :, r, :] for r in tgt_bone_ranges], dim=-2)
        N, M = obs_coords.shape[-2], tgt_coords.shape[-2]
        obs_tns = torch.cat([posed_sensor_tns[:, :, r, :, :] for r in obs_bone_ranges], dim=-3) # (B, T, N, 3, 3)
        tgt_tns = torch.cat([posed_sensor_tns[:, :, r, :, :] for r in tgt_bone_ranges], dim=-3) # (B, T, M, 3, 3)
        obs_tns_inverted = obs_tns.transpose(-1, -2).unsqueeze(-3) # (B, T, N, 1, 3, 3)
        obs_sensor_mask = torch.cat([sensor_mask[:, r] for r in obs_bone_ranges], dim=1) # (B, N)
        tgt_sensor_mask = torch.cat([sensor_mask[:, r] for r in tgt_bone_ranges], dim=1) # (B, M)
        obs_sensor_mask = obs_sensor_mask.unsqueeze(2).unsqueeze(1).expand(-1, T, -1, M) # (B, T, N, M)
        tgt_sensor_mask = tgt_sensor_mask.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1)
        relative_coords = (tgt_coords.unsqueeze(2) - obs_coords.unsqueeze(3)).unsqueeze(-1)
        relative_coords = torch.matmul(obs_tns_inverted, relative_coords).squeeze(-1) # (B, T, N, M, 3)
        obs_t_local = torch.cat([sensor_t_local[:, r, :] for r in obs_bone_ranges], dim=-2) # (B, N, 2)
        tgt_t_local = torch.cat([sensor_t_local[:, r, :] for r in tgt_bone_ranges], dim=-2) # (B, M, 2)
        obs_t_local = obs_t_local.unsqueeze(2).unsqueeze(1).expand(-1, T, -1, M, -1) # (B, T, N, M, 2)
        tgt_t_local = tgt_t_local.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1, -1) # (B, T, N, M, 2)
        obs_grp_embed = torch.cat([sensor_group_emb[:, r, :] for r in obs_bone_ranges], dim=-2)
        obs_grp_embed = obs_grp_embed.unsqueeze(2).unsqueeze(1).expand(-1, T, -1, M, -1) # (B, T, N, M, E)
        tgt_grp_embed = torch.cat([sensor_group_emb[:, r, :] for r in tgt_bone_ranges], dim=-2)
        tgt_grp_embed = tgt_grp_embed.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1, -1) # (B, T, N, M, E)
        obs_normal = obs_tns[..., 1]
        obs_back_mask = torch.sum(relative_coords * obs_normal.unsqueeze(-2), dim=-1) < 0 # (B, T, N, M)
        tgt_normal = tgt_tns[..., 1]
        tgt_back_mask = torch.sum(relative_coords * tgt_normal.unsqueeze(-3), dim=-1) > 0 # (B, T, N, M)
        relative_dist = torch.norm(relative_coords, dim=-1) # (B, T, N, M)
        # relative_dist[obs_back_mask | tgt_back_mask] = 1e8
        invalid_mask = ~obs_sensor_mask | ~tgt_sensor_mask
        if sparse_mode == 'both' or sparse_mode == 'close':
            cur_num_sensor = num_max if sparse_mode == 'close' else num_max // 2
            close_dir_indices = relative_dist.masked_fill(invalid_mask, 1e8).topk(cur_num_sensor, dim=-1, largest=False).indices
            close_dir_mask = torch.scatter(torch.zeros_like(relative_dist, dtype=torch.bool), -1, close_dir_indices, 1)
        else:
            close_dir_mask = torch.zeros_like(relative_dist, dtype=torch.bool)
        if sparse_mode == 'both' or sparse_mode == 'far':
            cur_num_sensor = num_max if sparse_mode == 'close' else num_max // 2
            far_dir_indices = relative_dist.masked_fill(invalid_mask, 0.0).topk(cur_num_sensor, dim=-1, largest=True).indices
            far_dir_mask = torch.scatter(torch.zeros_like(relative_dist, dtype=torch.bool), -1, far_dir_indices, 1)
        else:
            far_dir_mask = torch.zeros_like(relative_dist, dtype=torch.bool)
        # sparse_mask = close_dir_mask & ~obs_back_mask & ~tgt_back_mask & obs_sensor_mask & tgt_sensor_mask # (B, T, N, M)
        sparse_mask = (close_dir_mask | far_dir_mask) & obs_sensor_mask & tgt_sensor_mask # (B, T, N, M)
        sparse_indices = sparse_mask.to(torch.int8).topk(num_max, dim=-1, largest=True).indices # (B, T, N, num_max)
        relative_dir = F.normalize(relative_coords, dim=-1)
        relative_coords = torch.cat([relative_dir, relative_dist.unsqueeze(-1), obs_t_local, obs_grp_embed, tgt_t_local, tgt_grp_embed], dim=-1) # (B, T, N, M, 15)
        sparse_relative_coords = torch.gather(relative_coords, -2, sparse_indices.unsqueeze(-1).expand(-1, -1, -1, -1, relative_coords.shape[-1])) # (B, T, N, 15)
        sparse_mask = torch.gather(sparse_mask, -1, sparse_indices)
        sparse_relative_coords = torch.where(sparse_mask.unsqueeze(-1), sparse_relative_coords, -1.0)
        sparse_indices = torch.where(sparse_mask, sparse_indices, -1)
        sparse_dist = sparse_relative_coords[..., 3].clone()
        # sparse_dist = torch.where(sparse_mask, sparse_dist, float('inf')) # Now we do not use weighted scheme, so we do not need to mask out the invalid distances
        return sparse_relative_coords, sparse_indices, sparse_dist

    num_bone = 18 # MixamoBodyArmature has 18 bones in body groups
    num_sensor = posed_sensor_locations.shape[-2]
    num_sensor_per_bone = num_sensor // num_bone
    bone_ranges = get_bone_ranges(num_sensor_per_bone)
    relative_coords = []
    sparse_indices = []
    sparse_dist = []
    for g in group_pairs:
        g_coords = []
        g_indices = []
        g_dist = []
        for p in g:
            p_coords, p_indices, p_dist = relative_coords_by_group(bone_ranges, *p)
            g_coords.append(p_coords)
            g_indices.append(p_indices)
            g_dist.append(p_dist)
        relative_coords.append(torch.cat(g_coords, dim=-2))
        sparse_indices.append(torch.stack(g_indices, dim=-1))
        sparse_dist.append(torch.cat(g_dist, dim=-1))
    return relative_coords, sparse_indices, sparse_dist


def get_rest_sensor_feat_by_group(rest_sensor_feat: torch.Tensor):
    '''
    rest_sensor_feat: (B, N, D)
    '''
    bone_ranges = get_bone_ranges(rest_sensor_feat.shape[1]//18) # MixamoBodyArmature has 18 bones in body groups
    grouped_feat = []
    for g_name in ['LeftArm', 'RightArm', 'LeftLeg', 'RightLeg', 'Torso', 'Head']:
        grouped_feat.append(rest_sensor_feat[:, bone_ranges[g_name], :])
    return grouped_feat


@cache
def get_bone_ranges(num_sensor_per_bone: int):
    joint_group_bone_ranges = {}
    bone_cnt = 0
    for group_name, joint_group in MixamoBodyArmature._joint_groups.items():
        joint_group_bone_ranges[group_name] = slice(bone_cnt * num_sensor_per_bone, (bone_cnt + len(joint_group) - 1) * num_sensor_per_bone)
        for bone_idx in range(len(joint_group)):
            joint_group_bone_ranges[f'{group_name}_{bone_idx}'] = slice((bone_cnt + bone_idx) * num_sensor_per_bone, (bone_cnt + bone_idx + 1) * num_sensor_per_bone)
        bone_cnt += len(joint_group) - 1
    return joint_group_bone_ranges


def indices2relative_coords(group_pairs: list, posed_sensor_locations: torch.Tensor, posed_sensor_tns: torch.Tensor, sensor_mask: torch.Tensor, indices: torch.Tensor):
    '''
    posed_sensor_locations: (B, T, N, 3)
    posed_sensor_tns: (B, T, N, 3, 3)
    sensor_mask: (B, N)
    indices: (B, T, N, num_max)
    '''

    def get_relative_coords_by_indices(group_bone_ranges: dict, obs_groups: Tuple[str], tgt_groups: Tuple[str], indices: torch.Tensor):
        '''
        group_bone_ranges: dict of {group_name: slice}
        obs_groups: list of group names
        tgt_groups: list of group names
        indices: (B, T, N, num_max)
        '''
        T = posed_sensor_locations.shape[1]
        obs_bone_ranges = [group_bone_ranges[g] for g in obs_groups]
        tgt_bone_ranges = [group_bone_ranges[g] for g in tgt_groups]
        obs_coords = torch.cat([posed_sensor_locations[:, :, r, :] for r in obs_bone_ranges], dim=-2) # (B, T, N, 3)
        tgt_coords = torch.cat([posed_sensor_locations[:, :, r, :] for r in tgt_bone_ranges], dim=-2) # (B, T, M, 3)
        N, M = obs_coords.shape[-2], tgt_coords.shape[-2]
        obs_tns = torch.cat([posed_sensor_tns[:, :, r, :, :] for r in obs_bone_ranges], dim=-3) # (B, T, N, 3, 3)
        obs_tns_inverted = obs_tns.transpose(-1, -2).unsqueeze(-3) # (B, T, N, 1, 3, 3)
        obs_sensor_mask = torch.cat([sensor_mask[:, r] for r in obs_bone_ranges], dim=1) # (B, N)
        tgt_sensor_mask = torch.cat([sensor_mask[:, r] for r in tgt_bone_ranges], dim=1) # (B, M)
        obs_sensor_mask = obs_sensor_mask.unsqueeze(2).unsqueeze(1).expand(-1, T, -1, M) # (B, T, N, M)
        tgt_sensor_mask = tgt_sensor_mask.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1) # (B, T, N, M)
        relative_coords = (tgt_coords.unsqueeze(2) - obs_coords.unsqueeze(3)).unsqueeze(-1) # (B, T, N, M, 3, 1)
        relative_coords = torch.matmul(obs_tns_inverted, relative_coords).squeeze(-1) # (B, T, N, M, 3)
        sparse_mask = (indices != -1) # (B, T, N, num_max)
        indices = torch.where(sparse_mask, indices, 0)
        sele_coords = torch.gather(relative_coords, -2, indices.unsqueeze(-1).expand(-1, -1, -1, -1, 3)) # (B, T, N, num_max, 3)
        sele_obs_sensor_mask = torch.gather(obs_sensor_mask, -1, indices)
        sele_tgt_sensor_mask = torch.gather(tgt_sensor_mask, -1, indices)
        sele_mask = sele_obs_sensor_mask & sele_tgt_sensor_mask & sparse_mask
        sele_coords = torch.where(sele_mask.unsqueeze(-1), sele_coords, -1.0)
        return sele_coords, sele_mask

    bone_ranges = get_bone_ranges(posed_sensor_locations.shape[-2]//18) # MixamoBodyArmature has 18 bones in body groups
    relative_coords = []
    sparse_masks = []
    for g_idx, g in enumerate(group_pairs):
        g_coords = []
        g_masks = []
        for p_idx, p in enumerate(g):
            p_coords, p_mask = get_relative_coords_by_indices(bone_ranges, *p, indices[g_idx][..., p_idx])
            g_coords.append(p_coords)
            g_masks.append(p_mask)
        relative_coords.append(torch.cat(g_coords, dim=-2))
        sparse_masks.append(torch.cat(g_masks, dim=-1))

    return relative_coords, sparse_masks


def get_end_effector_velocity(end_effectors: list, posed_sensor_locations: torch.Tensor, sensor_mask: torch.Tensor):
    '''
    posed_sensor_locations: (B, T, N, 3)
    sensor_mask: (B, N)
    '''
    num_sensor_per_bone = posed_sensor_locations.shape[-2] // 18 # MixamoBodyArmature has 18 bones in body groups
    bone_ranges = get_bone_ranges(num_sensor_per_bone)
    T = posed_sensor_locations.shape[1]
    sensor_mask = sensor_mask.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, 3)
    posed_sensor_locations = torch.where(sensor_mask, posed_sensor_locations, 0.0)
    ef_velocities = []
    for g in end_effectors:
        g_range = bone_ranges[g]
        ef_loc = posed_sensor_locations[:, :, g_range, :]
        ef_vel = torch.diff(ef_loc, dim=1) # (B, T-1, num_sensor_per_bone, 3)
        ef_vel = torch.cat([ef_vel, ef_vel[:, -1:, :, :]], dim=1) # (B, T, num_sensor_per_bone, 3)
        ef_velocities.append(ef_vel)
    return torch.cat(ef_velocities, dim=-2)


def get_end_effector_pose_global(pose_global: torch.Tensor):
    '''
    pose_global: (B, T, J, 6)
    '''
    if pose_global.shape[2] == 25:
        bone_names = [n for n in MixamoBodyArmature._standard_joint_names if 'Hand' not in n or 'Hand' == n[-4:]]
    else:
        bone_names = MixamoBodyArmature._standard_joint_names
    end_bone_names = ['Head', 'LeftArm', 'RightArm', 'LeftLeg', 'RightLeg']
    end_pose_global = torch.cat([pose_global[:, :, bone_names.index(b)] for b in end_bone_names], dim=2)
    return end_pose_global


def all_pose_to_body_pose(pose_all: torch.Tensor):
    '''
    pose_all: (B, T, J, 6)
    parents_all: (J)
    '''
    body_bone_names = [n for n in MixamoBodyArmature._standard_joint_names if 'Hand' not in n or 'Hand' == n[-4:]]
    body_pose = torch.stack([pose_all[:, :, MixamoBodyArmature._standard_joint_names.index(b)] for b in body_bone_names], dim=2)
    return body_pose


def all_static_to_body_static(static: dict):
    body_bone_names = [n for n in MixamoBodyArmature._standard_joint_names if 'Hand' not in n or 'Hand' == n[-4:]]
    body_parents = [static['parents'][MixamoBodyArmature._standard_joint_names.index(b)] for b in body_bone_names]
    body_parents = [body_bone_names.index(MixamoBodyArmature._standard_joint_names[p]) if p != -1 else -1 for p in body_parents]
    body_parents = torch.as_tensor(body_parents)
    body_joint_locations = torch.stack([static['joint_locations'][:, MixamoBodyArmature._standard_joint_names.index(b)] for b in body_bone_names], dim=1)
    normalized_body_joint_locations = torch.stack([static['normalized_joint_locations'][:, MixamoBodyArmature._standard_joint_names.index(b)] for b in body_bone_names], dim=1)
    sensor_weights = torch.stack([static['sensor_weights'][..., MixamoBodyArmature._standard_joint_names.index(b)] for b in body_bone_names], dim=-1)
    sensor_weights = sensor_weights / (sensor_weights.sum(dim=-1, keepdim=True) + 1e-8)
    body_sensor_weights = torch.stack([static['body_sensor_weights'][..., MixamoBodyArmature._standard_joint_names.index(b)] for b in body_bone_names], dim=-1)
    body_sensor_weights = body_sensor_weights / (body_sensor_weights.sum(dim=-1, keepdim=True) + 1e-8)
    static.update({
        'parents': body_parents,
        'joint_locations': body_joint_locations,
        'normalized_joint_locations': normalized_body_joint_locations,
        'sensor_weights': sensor_weights,
        'body_sensor_weights': body_sensor_weights
    })
    return static


def body_pose_to_all_pose(ref_pose_all: torch.Tensor, body_pose: torch.Tensor):
    '''
    ref_pose_all: (B, T, J, 6)
    body_pose: (B, T, J', 6)
    '''
    body_bone_names = [n for n in MixamoBodyArmature._standard_joint_names if 'Hand' not in n or 'Hand' == n[-4:]]
    body_bone_indices = [MixamoBodyArmature._standard_joint_names.index(b) for b in body_bone_names]
    body_bone_indices = torch.as_tensor(body_bone_indices).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(ref_pose_all.shape[0], ref_pose_all.shape[1], -1, body_pose.shape[-1])
    pose_all = ref_pose_all.scatter(2, body_bone_indices.to(body_pose.device), body_pose)
    return pose_all
