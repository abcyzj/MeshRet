from copy import deepcopy
from pathlib import Path

import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
from data_loaders.mret import MRetDataModule
from model.retnet import RetNet
from run.train_retnet import RetNetCLI
import utils.BVH as BVH
from utils.body_armatures import MixamoBodyArmature
from utils.lbs import rigid_transform
from utils.rotation_conversions import quaternion_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from data_loaders.mret import all_pose_to_body_pose, all_static_to_body_static


class ComputeMetricsCLI(RetNetCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--input_dir', type=Path, required=True)
        parser.add_argument('--ckpt_path', type=str)


def main(model: RetNet, dataloader: torch.utils.data.DataLoader, input_dir: Path, test_penetration: bool):
    all_precisions = []
    for batch in tqdm(dataloader):
        x, y = batch['x'], batch['y']
        x_hat = []
        for sample_idx, meta in enumerate(batch['meta']):
            src_c_name, m_id, tgt_c_name, f_start = meta
            input_file = input_dir / f'{src_c_name},{m_id},{tgt_c_name},{f_start}.bvh'
            if not input_file.exists():
                x_hat.append(x[sample_idx].clone())
                continue
            anim, bvh_joint_names, _ = BVH.load(input_file)
            bvh_rotations = anim.rotations.qs.copy()
            bvh_rotations = torch.from_numpy(bvh_rotations).to(x.device, x.dtype)
            bvh_rotations = matrix_to_rotation_6d(quaternion_to_matrix(bvh_rotations))
            cur_x_hat = x[sample_idx].clone()
            for bvh_j_idx, j_name in enumerate(bvh_joint_names):
                if j_name not in MixamoBodyArmature._standard_joint_names:
                    continue
                j_idx = MixamoBodyArmature._standard_joint_names.index(j_name)
                cur_x_hat[:, j_idx] = bvh_rotations[:, bvh_j_idx]
            x_hat.append(cur_x_hat)
        x_hat = torch.stack(x_hat, dim=0)
        x_body = all_pose_to_body_pose(x)
        x_body_hat = all_pose_to_body_pose(x_hat)
        y_body = deepcopy(y)
        y_body['src_static'] = all_static_to_body_static(y_body['src_static'])
        y_body['tgt_static'] = all_static_to_body_static(y_body['tgt_static'])
        cur_contact_precision = model.contact_precision(x_body, x_body_hat, y_body['src_static'], y_body['tgt_static'])
        all_precisions.append(cur_contact_precision)

        parents = y['tgt_static']['parents']
        tgt_rest_verts = y['tgt_static']['verts']
        tgt_faces = y['tgt_static']['faces']
        tgt_lbs_weights = y['tgt_static']['lbs_weights']
        normalized_tgt_joint_loc = y['tgt_static']['normalized_joint_locations']
        root_translation = batch['root_translation']
        mse, local_mse = [], []
        if batch['gt'] is not None:
            def get_local_positions(joint_locs, parents):
                local_positions = joint_locs.clone()
                for i in range(1, joint_locs.shape[1]):
                    local_positions[:, i] = joint_locs[:, i] - joint_locs[:, parents[i]]
                return local_positions
            gt = batch['gt']
            joint_loc = normalized_tgt_joint_loc.unsqueeze(1).expand(-1, x.shape[1], -1, -1).flatten(0, 1) # (-1, J, 3)
            rot_hat = rotation_6d_to_matrix(x_hat).flatten(0, 1) # (-1, J, 3, 3)
            rot_gt = rotation_6d_to_matrix(gt).flatten(0, 1) # (-1, J, 3, 3)
            rot_copy = rotation_6d_to_matrix(x).flatten(0, 1) # (-1, J, 3, 3)
            joints_hat, _ = rigid_transform(rot_hat, joint_loc, parents)
            joints_gt, _ = rigid_transform(rot_gt, joint_loc, parents)
            cur_mse = (joints_hat - joints_gt).norm(dim=-1).mean()
            cur_local_mse = (get_local_positions(joints_hat, parents) - get_local_positions(joints_gt, parents)).norm(dim=-1).mean()
            mse.append(cur_mse)
            local_mse.append(cur_local_mse)

        if test_penetration:
            tgt_joint_loc = y['tgt_static']['joint_locations']
            pr = []
            for clip_idx, sample in enumerate(x_hat):
                armature = MixamoBodyArmature(MixamoBodyArmature._standard_joint_names, parents, tgt_rest_verts[clip_idx].detach().cpu().numpy(), tgt_faces[clip_idx].detach().cpu().numpy(), tgt_lbs_weights[clip_idx].detach().cpu().numpy(), tgt_joint_loc[clip_idx].detach().cpu().numpy())
                armature.joint_rotations = sample.unsqueeze(0)
                armature.root_locations = root_translation[clip_idx].reshape(1, -1, 3)
                pr.append(armature.penetration_ratio())

    all_precisions = torch.as_tensor(all_precisions)
    for i in range(all_precisions.shape[1]):
        print(f'Contact precision {i}: {all_precisions[:, i].mean().item()}')

    if len(mse) > 0:
        print(f'MSE: {torch.stack(mse).mean().item()}')
        print(f'Local MSE: {torch.stack(local_mse).mean().item()}')

    if test_penetration:
        pr = torch.as_tensor(pr)
        print(f'Head penetration ratio: {pr[:, 0].mean().item()}')
        print(f'Body penetration ratio: {pr[:, 1].mean().item()}')
        print(f'Leg penetration ratio: {pr[:, 2].mean().item()}')


if __name__ == '__main__':
    cli = ComputeMetricsCLI(RetNet, MRetDataModule, run=False)
    cli.model.freeze()
    cli.datamodule.setup('test')
    main(cli.model, cli.datamodule.test_dataloader(), cli.config.input_dir, cli.config.model.test_penetration)
