import os
import pickle
import shlex
from copy import deepcopy
from pathlib import Path
from subprocess import Popen
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from pytorch_lightning.cli import LightningArgumentParser
from trimesh import Trimesh
from trimesh.sample import sample_surface
import pyvista as pv

import utils.BVH as BVH
from data_loaders.mret import (MRetDataModule, all_pose_to_body_pose,
                               all_static_to_body_static,
                               body_pose_to_all_pose, get_bone_ranges, indices2relative_coords)
from model.retnet import RetNet
from run.train_retnet import RetNetCLI
from utils.Animation import Animation
from utils.body_armatures import MixamoBodyArmature
from utils.lbs import Rotation2MixamoVerts, SkinnableSensor
from utils.Quaternions_old import Quaternions
from utils.rotation_conversions import matrix_to_quaternion, rotation_6d_to_matrix


class GenCLI(RetNetCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint')
        parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
        parser.add_argument('--output_dir', type=str, default='output', help='Output directory')

    def before_instantiate_classes(self):
        self.config.data['batch_size'] = self.config.num_samples


def main():
    cli = GenCLI(RetNet, MRetDataModule, run=False)
    cli.model = RetNet.load_from_checkpoint(cli.config.ckpt_path, map_location=cli.model.device)
    cli.model.freeze()
    cli.datamodule.setup('test')
    dataloader = cli.datamodule.test_dataloader()
    dataloader.dataset.data_keys =  [ # Specify the data keys to be used for demo
        ('QY_0713_LiYan_049', 3, 'Mousey', 0),
        ('QY_0713_LiYan_049', 31, 'YBot', 0),
        ('QY_0713_LiYan_049', 45, 'Ortiz', 0),
        ('QY_0713_LiYan_049', 4, 'Aj', 0),
        ('QY_0713_LiYan_049', 43, 'Mousey', 0),
        ('QY_0713_LiYan_049', 9, 'Aj', 0),
        ('QY_0713_LiYan_049', 9, 'Kaya', 0),
        ('QY_0713_LiYan_049', 47, 'Aj', 0),
        ('QY_0713_LiYan_049', 47, 'Mousey', 0),
        ('QY_0713_LiYan_049', 72, 'Mousey', 0),
    ]

    output_dir = Path(cli.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(dataloader))

    with torch.inference_mode():
        x, y = batch['x'], batch['y']
        if cli.config.model.only_body:
            x_body = all_pose_to_body_pose(x)
            y_body = deepcopy(y)
            y_body['src_static'] = all_static_to_body_static(y_body['src_static'])
            y_body['tgt_static'] = all_static_to_body_static(y_body['tgt_static'])
            num_clips = cli.config.data.seq_len // 30
            x_hat = []
            for i in range(num_clips):
                x_hat.append(cli.model(x_body[:, i*30:i*30+30], y_body))
            x_hat = torch.cat(x_hat, dim=1)
            for i in range(1, num_clips):
                for k in range(3):
                    m_1 = x_hat[:, i*30-k:i*30-k+2].mean(dim=1)
                    m_2 = x_hat[:, i*30+k:i*30+k+2].mean(dim=1)
                    x_hat[:, i*30-k] = m_1
                    x_hat[:, i*30+k+1] = m_2
            x_hat = body_pose_to_all_pose(x, x_hat)
        else:
            x_hat = cli.model(x, y)

    rot2verts = Rotation2MixamoVerts(x_hat.device)
    seq_mask = y['mask'].reshape(x.shape[0], x.shape[1])
    tgt_static = y['tgt_static']
    tgt_rest_verts = tgt_static['verts']
    tgt_faces = tgt_static['faces']
    tgt_joint_loc = tgt_static['joint_locations']
    parents = tgt_static['parents']
    tgt_lbs_weights = tgt_static['lbs_weights']
    src_static = y['src_static']
    src_rest_verts = src_static['verts']
    src_faces = src_static['faces']
    src_joint_loc = src_static['joint_locations']
    src_lbs_weights = src_static['lbs_weights']
    root_translation = batch['root_translation']
    joint_names = MixamoBodyArmature._standard_joint_names
    joint_names = ['mixamorig:' + j for j in joint_names]

    for clip_idx, sample in enumerate(x_hat):
        posed_verts = rot2verts(sample, seq_mask[clip_idx], cli.model.data_rep, tgt_rest_verts[clip_idx], tgt_joint_loc[clip_idx], parents, tgt_lbs_weights[clip_idx])
        posed_verts += root_translation[clip_idx]
        export_video(posed_verts, tgt_faces[clip_idx], output_dir / f'sample_{clip_idx}.mp4')
        export_bvh(joint_names, sample, tgt_joint_loc[clip_idx], root_translation[clip_idx], parents, output_dir / f'sample_{clip_idx}.bvh')
        posed_verts = rot2verts(x[clip_idx], seq_mask[clip_idx], cli.model.data_rep, src_rest_verts[clip_idx], src_joint_loc[clip_idx], parents, src_lbs_weights[clip_idx])
        posed_verts += root_translation[clip_idx]
        export_video(posed_verts, src_faces[clip_idx], output_dir / f'sample_{clip_idx}_src.mp4')
        export_bvh(joint_names, x[clip_idx], src_joint_loc[clip_idx], root_translation[clip_idx], parents, output_dir / f'sample_{clip_idx}_src.bvh')
        posed_verts = rot2verts(x[clip_idx], seq_mask[clip_idx], cli.model.data_rep, tgt_rest_verts[clip_idx], tgt_joint_loc[clip_idx], parents, tgt_lbs_weights[clip_idx])
        posed_verts += root_translation[clip_idx]
        export_video(posed_verts, tgt_faces[clip_idx], output_dir / f'sample_{clip_idx}_copy.mp4')
        export_bvh(joint_names, x[clip_idx], tgt_joint_loc[clip_idx], root_translation[clip_idx], parents, output_dir / f'sample_{clip_idx}_copy.bvh')


def export_video(posed_verts, faces, path):
    with NamedTemporaryFile('wb', delete=False) as temp_f:
        pickle.dump((posed_verts, faces), temp_f)
        temp_f.close()
        p = Popen(shlex.split(f'python -m run.play_mesh --input {temp_f.name} --output {path}'), stdout=None, stderr=None)
        p.wait()
        os.unlink(temp_f.name)


def export_bvh(joint_names, rotations, joint_loc, root_translations, parents, output):
    rotations = matrix_to_quaternion(rotation_6d_to_matrix(rotations))
    rotations = Quaternions(rotations.detach().cpu().numpy())
    root_positions = root_translations.detach().cpu().numpy()
    offsets = joint_loc.detach().cpu().numpy().copy()
    for idx, p in enumerate(parents):
        if p == -1:
            continue
        offsets[idx] = joint_loc[idx] - joint_loc[p]
    positions = np.concatenate([root_positions, np.repeat(offsets[np.newaxis], root_positions.shape[0], axis=0)[:, 1:]], axis=1)
    orients = Quaternions.id(offsets.shape[0])
    anim = Animation(rotations, positions, orients, offsets, parents)
    BVH.save(output, anim, joint_names, 1/30)


if __name__ == '__main__':
    main()
