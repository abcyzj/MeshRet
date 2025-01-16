from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.cli import LightningArgumentParser
from tqdm import tqdm

import utils.BVH as BVH
from data_loaders.mret import MRetDataModule
from model.retnet import RetNet
from run.train_retnet import RetNetCLI
from utils.Animation import Animation
from utils.body_armatures import MixamoBodyArmature
from utils.Quaternions_old import Quaternions
from utils.rotation_conversions import (matrix_to_quaternion,
                                        rotation_6d_to_matrix)


class ExportCLI(RetNetCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--output_dir', type=Path, required=True)
        parser.add_argument('--ckpt_path', type=str)


def main(dataloader: torch.utils.data.DataLoader, output_dir):
    joint_names = ['mixamorig:' + n for n in MixamoBodyArmature._standard_joint_names]
    for batch in tqdm(dataloader):
        for clip_meta, sample, root_positions, joint_loc, tgt_joint_loc in zip(batch['meta'], batch['x'], batch['root_translation'], batch['y']['src_static']['joint_locations'], batch['y']['tgt_static']['joint_locations']):
            rotation = rotation_6d_to_matrix(sample)
            rotation = matrix_to_quaternion(rotation).numpy()
            rotation = Quaternions(rotation)
            joint_loc = joint_loc.reshape(-1, 3)
            offsets = joint_loc.numpy().copy()
            parents = batch['y']['src_static']['parents']
            for idx, p in enumerate(parents):
                if p == -1:
                    continue
                offsets[idx] = joint_loc[idx] - joint_loc[p]
            root_positions = root_positions.numpy()
            positions = np.concatenate([root_positions, np.repeat(offsets[np.newaxis], root_positions.shape[0], axis=0)[:, 1:]], axis=1)
            orients = Quaternions.id(offsets.shape[0])
            anim = Animation(rotation, positions, orients, offsets, parents)
            output = output_dir / (','.join([str(n) for n in clip_meta]) + '.bvh')
            BVH.save(output.as_posix(), anim, joint_names, 1/30)

            tgt_char = clip_meta[2]
            tgt_offsets = tgt_joint_loc.reshape(-1, 3).numpy().copy()
            for idx, p in enumerate(parents):
                if p == -1:
                    continue
                tgt_offsets[idx] = tgt_joint_loc[idx] - tgt_joint_loc[p]
            tgt_positions = np.concatenate([root_positions, np.repeat(offsets[np.newaxis], root_positions.shape[0], axis=0)[:, 1:]], axis=1)
            anim = Animation(rotation, tgt_positions, orients, tgt_offsets, parents)
            output = output_dir / (','.join([str(n) for n in clip_meta]) + '_tgt.bvh')
            BVH.save(output.as_posix(), anim, joint_names, 1/30)


if __name__ == '__main__':
    cli = ExportCLI(RetNet, MRetDataModule, run=False)
    cli.model.freeze()
    cli.datamodule.setup('test')
    output_dir = cli.config.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    dataloader = cli.datamodule.test_dataloader()
    main(dataloader, output_dir)
