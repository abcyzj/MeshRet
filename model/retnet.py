from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.cli import instantiate_class
from torchvision.ops import MLP

from data_loaders.mret import (all_pose_to_body_pose,
                               all_static_to_body_static,
                               body_pose_to_all_pose,
                               get_end_effector_pose_global,
                               get_end_effector_velocity, get_relative_coords,
                               get_rest_sensor_feat_by_group,
                               indices2relative_coords)
from model.pointnet import PointNet, SensorNet
from utils.body_armatures import MixamoBodyArmature
from utils.lbs import SkinnableSensor, rigid_transform
from utils.rotation_conversions import rotation_6d_to_matrix


class RetNet(pl.LightningModule):
    def __init__(self,
                 njoints: int,
                 nfeats: int,
                 optim_init: dict,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 data_rep: str = 'rot6d',
                 arch: str = 'trans_enc_dec',
                 num_ring_per_bone: int = 4,
                 num_point_per_ring: int = 8,
                 sensor_grp_emb_dim: int = 8,
                 num_sensor_grp: int = 16,
                 cond_mode: str = 'tns',
                 num_sensor_max: int = 4,
                 lambda_dir: float = 10.0,
                 lambda_efv: float = 5.0,
                 lambda_spine: float = 10.0,
                 lambda_gan: float = 2.0,
                 lambda_pair_softmin: float = 100.0,
                 lambda_group_softmin: float = 10.0,
                 lambda_weighted_dir: float = 5.0,
                 lambda_contact: float = 5.0,
                 lambda_ef_pose_global: float = 5.0,
                 lambda_dm: float = 0.0,
                 group_pairs: list = None,
                 end_effectors: list = None,
                 dis_margin: float = 0.3,
                 update_dis_every_n_step: int = 1,
                 limb_bound: tuple = (1.0, 2.0),
                 far_cos_bound: float = 0.5,
                 only_body: bool = False,
                 test_penetration: bool = False,
                 sparse_mode: str = 'both'
                 ):
        super().__init__()

        self.automatic_optimization = False

        self.save_hyperparameters()

        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

        self.optim_init = optim_init

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        self.cond_mode = cond_mode
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.num_ring_per_bone = num_ring_per_bone
        self.num_point_per_ring = num_point_per_ring
        self.sensor_grp_emb_dim = sensor_grp_emb_dim
        self.num_sensor_max = num_sensor_max

        self.lambda_dir = lambda_dir
        self.lambda_efv = lambda_efv
        self.lambda_spine = lambda_spine
        self.lambda_gan = lambda_gan
        self.lambda_pair_softmin = lambda_pair_softmin
        self.lambda_group_softmin = lambda_group_softmin
        self.lambda_weighted_dir = lambda_weighted_dir
        self.lambda_contact = lambda_contact
        self.lambda_ef_pose_global = lambda_ef_pose_global
        self.dis_margin = dis_margin
        self.update_dis_every_n_step = update_dis_every_n_step
        self.lambda_dm = lambda_dm

        self.group_pairs = group_pairs
        self.end_effectors = end_effectors

        self.limb_bound = limb_bound
        self.far_cos_bound = far_cos_bound
        self.only_body = only_body
        self.test_penetration = test_penetration
        self.sparse_mode = sparse_mode

        self.G = nn.ModuleList([])
        if self.arch == 'trans_enc_dec':
            print('TRANS_ENC_DEC init')
            self.sensor_grp_emb = nn.Embedding(num_sensor_grp+ 1, self.sensor_grp_emb_dim) # num_sensor_grp is the index of the 'none' group
            sensor_pair_dim = 3+1+3+3+self.sensor_grp_emb_dim*2
            self.sensorEnc = nn.ModuleList([
                SensorNet(input_dim=sensor_pair_dim, latent_dim=self.latent_dim) for _ in range(len(self.group_pairs))
            ])
            num_point_per_bone = self.num_ring_per_bone * self.num_point_per_ring
            self.ef_fc = MLP(num_point_per_bone*5*3, [self.latent_dim, self.latent_dim], nn.LayerNorm, dropout=self.dropout)
            self.root_rot_fc = MLP(6, [self.latent_dim, self.latent_dim], nn.LayerNorm, dropout=self.dropout) # TODO remove hardcoding
            self.sensor_fc = nn.Linear(self.latent_dim*len(self.group_pairs), self.latent_dim)
            rest_sensor_dim = 3+9+3+self.sensor_grp_emb_dim
            self.restPointEnc = nn.ModuleList([
                PointNet(input_dim=rest_sensor_dim, latent_dim=self.latent_dim) for _ in range(6)
            ])
            self.rest_fc = MLP(6*self.latent_dim, [self.latent_dim*2, self.latent_dim], nn.LayerNorm, dropout=self.dropout)
            self.seqTrans = nn.Transformer(d_model=self.latent_dim,
                                           nhead=self.num_heads,
                                           num_encoder_layers=self.num_layers,
                                           num_decoder_layers=self.num_layers,
                                           dim_feedforward=self.ff_size,
                                           dropout=self.dropout,
                                           activation=self.activation,
                                           batch_first=True)
            self.G.extend([
                self.sensor_grp_emb,
                self.sensorEnc,
                self.ef_fc,
                self.root_rot_fc,
                self.sensor_fc,
                self.restPointEnc,
                self.rest_fc,
                self.seqTrans
            ])
        else:
            raise ValueError('Please choose correct architecture.')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        self.G.append(self.output_process)

        self.D = MotionDis(self.only_body)


    def forward(self, x: torch.Tensor, y: dict):
        nframes = x.shape[1]

        if 'tns' == self.cond_mode:
            src_rest_senosr_feat, src_relative_coords, _, src_ef_vel, src_root_rots, *_ = self.get_cond_inp({'src_motion': x, **y['src_static']})
            tgt_rest_sensor_feat, *_ = self.get_cond_inp({'src_motion': x, **y['tgt_static']})
            if 'src_rest_sensor_feat' in y: # for the case of test
                src_rest_senosr_feat = y['src_rest_sensor_feat']
            if 'tgt_rest_sensor_feat' in y: # for the case of test
                tgt_rest_sensor_feat = y['tgt_rest_sensor_feat']
            if 'relative_coords' in y: # for the case of test
                src_relative_coords = y['src_relative_coords']
            if 'ef_vel' in y: # for the case of test
                src_ef_vel = y['src_ef_vel']
            if 'root_rots' in y: # for the case of test
                src_root_rots = y['src_root_rots']

        ori_x = x
        x = self.input_process(x)

        if self.arch == 'trans_enc_dec':
            src_rest_sensor_emb = torch.cat([enc(coords.unsqueeze(1)) for enc, coords in zip(self.restPointEnc, src_rest_senosr_feat)], dim=-1)  # [bs, 1, d*6]
            src_rest_sensor_emb = self.rest_fc(src_rest_sensor_emb)  # [bs, 1, d]
            tgt_rest_sensor_emb = torch.cat([enc(coords.unsqueeze(1)) for enc, coords in zip(self.restPointEnc, tgt_rest_sensor_feat)], dim=-1)  # [bs, 1, d*6]
            tgt_rest_sensor_emb = self.rest_fc(tgt_rest_sensor_emb)  # [bs, 1, d]
            sensor_emb = torch.cat([enc(coords) for enc, coords in zip(self.sensorEnc, src_relative_coords)], dim=-1) # [bs, seqlen, d*4]
            temporal_emb = self.sensor_fc(sensor_emb) + self.ef_fc(src_ef_vel.flatten(-2)) + self.root_rot_fc(src_root_rots) # [bs, seqlen, d]
            yseq = torch.cat((tgt_rest_sensor_emb, temporal_emb), axis=1)  # [bs, seqlen+1, d]
            yseq = self.sequence_pos_encoder(yseq, batch_first=True)  # [bs, seqlen+1, d]
            align_mask = ~torch.eye(nframes+1, device=x.device, dtype=torch.bool)
            align_mask[0, :] = False
            xseq = torch.cat((src_rest_sensor_emb, x), axis=1)  # [bs, seqlen+1, d]
            xseq = self.sequence_pos_encoder(xseq, batch_first=True)  # [bs, seqlen+1, d]
            output = self.seqTrans(yseq, xseq, memory_mask=align_mask)[:, 1:]  # [bs, seqlen, d]

        output = self.output_process(output)  # [bs, nframes, njoints, nfeats]
        output = torch.cat([ori_x[:, :, :1], output[:, :, 1:]], dim=2) # the model do not predict root rotation
        return output


    def get_cond_inp(self, y, sparse_mode=None):
        nframes = y['src_motion'].shape[1]
        sensor_tns = y['body_sensor_tns']
        sensor_t_local = y['body_sensor_t_local'] # [bs, #sensors, 2]
        sensor_group_emb = self.sensor_grp_emb(y['body_sensor_group_idx']) # [bs, #sensors, d]
        normalized_sensor_locations = y['normalized_body_sensor_locations']
        rest_sensor_feat = torch.cat((normalized_sensor_locations, torch.flatten(sensor_tns, -2), sensor_group_emb, sensor_t_local), axis=-1) # [bs, #sensors, 3+9+grp_emb_dim]

        sensor_tns = sensor_tns.unsqueeze(1).expand(-1, nframes, -1, -1, -1)
        sensor_weights = y['body_sensor_weights'].unsqueeze(1).expand(-1, nframes, -1, -1)
        normalized_joint_locations = y['normalized_joint_locations'].unsqueeze(1).expand(-1, nframes, -1, -1)
        normalized_sensor_locations = normalized_sensor_locations.unsqueeze(1).expand(-1, nframes, -1, -1)
        skin_sensor = SkinnableSensor(normalized_sensor_locations, sensor_tns, normalized_joint_locations, y['parents'], sensor_weights)
        posed_sensor_locations, posed_sensor_tns, pose_global = skin_sensor.skin(y['src_motion'], ret_pose_global=True)
        sensor_mask = y['body_sensor_mask']

        if sparse_mode is None:
            sparse_mode = self.sparse_mode
        relative_coords, sparse_indices, sparse_dist = get_relative_coords(self.group_pairs, posed_sensor_locations, posed_sensor_tns, sensor_t_local, sensor_group_emb, sensor_mask, self.num_sensor_max, sparse_mode)
        grouped_rest_sensor_feat = get_rest_sensor_feat_by_group(rest_sensor_feat)
        ef_vel = get_end_effector_velocity(self.end_effectors, posed_sensor_locations, sensor_mask)

        root_rots = y['src_motion'][:, :, 0]
        ef_pose_global = get_end_effector_pose_global(pose_global)
        return grouped_rest_sensor_feat, relative_coords, sparse_indices, ef_vel, root_rots, sparse_dist, ef_pose_global


    def configure_optimizers(self):
        opt_G = instantiate_class(self.G.parameters(), self.optim_init)
        opt_D = instantiate_class(self.D.parameters(), self.optim_init)
        return opt_G, opt_D


    def training_step(self, batch: dict, batch_idx: int):
        x, y = batch['x'], batch['y']
        if self.only_body:
            x = all_pose_to_body_pose(x)
            y['src_static'] = all_static_to_body_static(y['src_static'])
            y['tgt_static'] = all_static_to_body_static(y['tgt_static'])
        x_hat = self(x, y)

        def sum_flat(tensor):
            return tensor.sum(dim=list(range(1, len(tensor.shape))))

        def masked_l2(a, b, mask):
            mse_loss = (a - b) ** 2
            mse_loss = sum_flat(mse_loss * mask.float())
            n_entries = a.shape[-1] * a.shape[-2]
            non_zero_elements = sum_flat(mask) * n_entries
            mse_loss = mse_loss / (non_zero_elements + 1e-8)
            return mse_loss.mean()

        def masked_cosine(a, b, mask, upper_bound=1.0):
            '''
            a: (bs, seqlen, J, num_max, 3)
            b: (bs, seqlen, J, num_max, 3)
            mask: (bs, seqlen, J, num_max)
            '''
            cos_sim = F.cosine_similarity(a, b, dim=-1)
            cos_sim = torch.where(cos_sim > upper_bound, torch.ones_like(cos_sim), cos_sim)
            loss = 1 - cos_sim
            loss = sum_flat(loss * mask.float())
            non_zero_elements = sum_flat(mask)
            cosine_loss = loss / (non_zero_elements + 1e-8)
            return cosine_loss.mean()
        
        def masked_bce_loss(a, b, mask):
            bce_loss = nn.BCELoss(reduction='none')
            loss = bce_loss(a, b)
            loss = torch.sum(loss * mask.float()) # Frame-level, not batch_level
            non_zero_elements = torch.sum(mask)
            bce_loss = loss / non_zero_elements
            return bce_loss

        def maked_mean(x, mask, dim_start):
            '''
            x may contain inf value
            '''
            x = x.masked_fill(~mask, 0.0)
            x = x.sum(dim=list(range(dim_start, len(x.shape))))
            n_entries = mask.float().sum(dim=list(range(dim_start, len(mask.shape))))
            x = x / (n_entries + 1e-8)
            return x
        
        def masked_dm_sim(a, b, mask):
            a = a.masked_fill(~mask, 0.0)
            b = b.masked_fill(~mask, 0.0)
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
            b = b / (b.sum(dim=-1, keepdim=True) + 1e-8)
            return masked_l2(a, b, mask)

        opt_G, opt_D = self.optimizers()
        seq_mask = y['mask'] # (B, T, 1, 1)
        is_intra = y['is_intra'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (B, 1, 1, 1)
        adv_mask = seq_mask.float().squeeze([-2, -1]) # (B, T)

        terms = {}

        # Optimize D
        d_x = self.D(x)
        d_x_hat = self.D(x_hat.detach())
        err_D_real = masked_bce_loss(d_x, torch.ones_like(d_x), adv_mask)
        err_D_fake = masked_bce_loss(d_x_hat, torch.zeros_like(d_x_hat), adv_mask)
        terms['err_D'] = err_D_real + err_D_fake
        if (batch_idx % self.update_dis_every_n_step) == 0 and  adv_mask.sum() > 0 and d_x_hat[adv_mask.bool()].max() > self.dis_margin: # Only update D when the fake data is not good enough
            opt_D.zero_grad()
            self.manual_backward(terms['err_D'])
            opt_D.step()


        # Optimize G
        d_x_hat = self.D(x_hat)
        terms['err_G'] = masked_bce_loss(d_x_hat, torch.ones_like(d_x_hat), adv_mask)

        terms['rot_mse'] = masked_l2(x, x_hat, seq_mask)

        if self.lambda_dir > 0. or self.lambda_efv > 0. or self.lambda_weighted_dir > 0. or self.lambda_contact > 0. or self.lambda_dm > 0.:
            src_static, tgt_static = y['src_static'], y['tgt_static']
            with torch.no_grad():
                _, src_relative_coords, src_sparse_indices, src_ef_vel, _, src_sparse_dist, src_ef_pose_global = self.get_cond_inp({'src_motion': x, **src_static})
            bs, nframes = src_relative_coords[0].shape[:2]
            tgt_parents = tgt_static['parents']
            tgt_sensor_tns = tgt_static['body_sensor_tns'].unsqueeze(1).expand(-1, nframes, -1, -1, -1)
            tgt_sensor_weights = tgt_static['body_sensor_weights'].unsqueeze(1).expand(-1, nframes, -1, -1)
            tgt_normalized_sensor_locations = tgt_static['normalized_body_sensor_locations'].unsqueeze(1).expand(-1, nframes, -1, -1)
            tgt_normalized_joint_locations = tgt_static['normalized_joint_locations'].unsqueeze(1).expand(-1, nframes, -1, -1)
            skin_sensor = SkinnableSensor(tgt_normalized_sensor_locations, tgt_sensor_tns, tgt_normalized_joint_locations, tgt_parents, tgt_sensor_weights)
            posed_sensor_locations, posed_sensor_tns, pred_pose_global = skin_sensor.skin(x_hat, ret_pose_global=True)
            pred_ef_pose_global = get_end_effector_pose_global(pred_pose_global)
            tgt_sensor_mask = tgt_static['body_sensor_mask']

            if self.lambda_efv > 0.:
                pred_efv = get_end_effector_velocity(self.end_effectors, posed_sensor_locations, tgt_sensor_mask)
                terms['efv_mse'] = masked_l2(src_ef_vel, pred_efv, seq_mask)

            if self.lambda_dir > 0. or self.lambda_weighted_dir > 0. or self.lambda_contact > 0. or self.lambda_dm > 0.:
                pred_relative_coords, pred_sparse_mask = indices2relative_coords(self.group_pairs, posed_sensor_locations, posed_sensor_tns, tgt_sensor_mask, src_sparse_indices)

                if self.lambda_dir > 0. or self.lambda_contact > 0. or self.lambda_dm > 0.:
                    terms['far_dir_cosine'] = 0.0
                    terms['middle_dir_cosine'] = 0.0
                    terms['contact'] = 0.0
                    terms['dm'] = 0.0

                    for gt_coords, gt_dist, pred_coords, cur_sparse_mask, cur_group_pair in zip(src_relative_coords, src_sparse_dist, pred_relative_coords, pred_sparse_mask, self.group_pairs):
                        gt_dir = gt_coords[..., :3]
                        cur_obs_limb_name = cur_group_pair[0][0][0].split('_')[0]
                        src_limb_radius = y['src_static']['normalized_limb_radius'][cur_obs_limb_name].reshape(-1, 1, 1, 1)
                        tgt_limb_radius = y['tgt_static']['normalized_limb_radius'][cur_obs_limb_name].reshape(-1, 1, 1, 1)
                        close_mask = gt_dist < src_limb_radius * self.limb_bound[0]
                        far_mask = gt_dist >  src_limb_radius * self.limb_bound[1]
                        middle_mask = ~close_mask & ~far_mask
                        if self.lambda_dir > 0.:
                            terms['far_dir_cosine'] = terms['far_dir_cosine'] + masked_cosine(gt_dir, pred_coords, far_mask & cur_sparse_mask & seq_mask, upper_bound=self.far_cos_bound)
                            terms['middle_dir_cosine'] = terms['middle_dir_cosine'] + masked_cosine(gt_dir, pred_coords, middle_mask & cur_sparse_mask & seq_mask)
                        if self.lambda_contact > 0.:
                            pred_dist = pred_coords.norm(dim=-1)
                            terms['contact'] = terms['contact'] + masked_l2(gt_dist / src_limb_radius, pred_dist / tgt_limb_radius, close_mask & cur_sparse_mask & seq_mask)
                        if self.lambda_dm > 0.:
                            pred_dist = pred_coords.norm(dim=-1)
                            terms['dm'] = terms['dm'] + masked_dm_sim(gt_dist / src_limb_radius, pred_dist / tgt_limb_radius, cur_sparse_mask & seq_mask)
                    terms['dir_cosine'] = terms['far_dir_cosine'] + terms['middle_dir_cosine']


                if self.lambda_weighted_dir > 0.:
                    group_mean_dist = []
                    for gt_dist, cur_sparse_mask in zip(src_sparse_dist, pred_sparse_mask):
                        group_mean_dist.append(maked_mean(gt_dist, cur_sparse_mask, 2))
                    group_mean_dist = torch.stack(group_mean_dist, dim=-1) # (B, T, #groups)
                    group_dist_weight = F.softmin(group_mean_dist * self.lambda_group_softmin, dim=-1).permute(2, 0, 1) # (#groups, B, T)
                    terms['dir_weighted'] = 0.0
                    for cur_group_dist_weight, gt_coords, pred_coords, gt_dist, cur_sparse_mask in zip(group_dist_weight, src_relative_coords, pred_relative_coords, src_sparse_dist, pred_sparse_mask):
                        gt_dir = gt_coords[..., :3]
                        pair_dist_weight = F.softmin(gt_dist.flatten(2)*self.lambda_pair_softmin, dim=-1).reshape(gt_dist.shape)
                        pair_dist_weight = torch.where(cur_sparse_mask, pair_dist_weight, 0.0)
                        w = (cur_sparse_mask & seq_mask).float() * pair_dist_weight * cur_group_dist_weight.unsqueeze(-1).unsqueeze(-1)
                        terms['dir_weighted'] = terms['dir_weighted'] + masked_cosine(gt_dir, pred_coords, w)

        if self.lambda_ef_pose_global > 0.:
            terms['ef_pose_global'] = masked_l2(src_ef_pose_global, pred_ef_pose_global, seq_mask.squeeze(-1))


        if self.lambda_spine > 0.:
            spine_idx = torch.tensor([1, 2, 3], dtype=torch.long, device=x.device)
            pred_spine_rot = x_hat[:, :, spine_idx]
            gt_spine_rot = x[:, :, spine_idx]
            terms['spine_mse'] = masked_l2(gt_spine_rot, pred_spine_rot, seq_mask)

        for k, v in terms.items():
            self.log(f'train/{k}', v, sync_dist=True)

        loss = terms['rot_mse'] + (self.lambda_dir * terms.get('dir_cosine', 0.0)) + \
            (self.lambda_efv * terms.get('efv_mse', 0.0)) + \
            (self.lambda_spine * terms.get('spine_mse', 0.0)) + \
            (self.lambda_gan * terms.get('err_G', 0.0)) + \
            (self.lambda_weighted_dir * terms.get('dir_weighted', 0.0)) + \
            (self.lambda_ef_pose_global * terms.get('ef_pose_global', 0.0)) + \
            (self.lambda_contact * terms.get('contact', 0.0)) + \
            (self.lambda_dm * terms.get('dm', 0.0))

        opt_G.zero_grad()
        self.manual_backward(loss)
        opt_G.step()


    def test_step(self, batch: dict, batch_idx: int):
        x, y = batch['x'], batch['y']
        if self.only_body:
            x_body = all_pose_to_body_pose(x)
            y_body = deepcopy(y)
            y_body['src_static'] = all_static_to_body_static(y_body['src_static'])
            y_body['tgt_static'] = all_static_to_body_static(y_body['tgt_static'])
        x_body_hat = self(x_body, y_body)
        
        if not self.test_penetration and batch['gt'] is None:
            our_contact_precision = self.contact_precision(x_body, x_body_hat, y_body['src_static'], y_body['tgt_static'])
            our_arm_contact_precision = torch.mean(our_contact_precision[2:4]) # Contact precision between arms and torso
            self.log(f'test/contact_precision', our_arm_contact_precision, sync_dist=True, batch_size=x.shape[0])

        x_hat = body_pose_to_all_pose(x, x_body_hat)
        sample_pr = []
        parents = y['tgt_static']['parents']
        tgt_rest_verts = y['tgt_static']['verts']
        tgt_faces = y['tgt_static']['faces']
        tgt_lbs_weights = y['tgt_static']['lbs_weights']
        normalized_tgt_joint_loc = y['tgt_static']['normalized_joint_locations']
        root_translation = batch['root_translation']

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
            mse = (joints_hat - joints_gt).norm(dim=-1).mean()
            local_mse = (get_local_positions(joints_hat, parents) - get_local_positions(joints_gt, parents)).norm(dim=-1).mean()
            self.log('test/mse', mse, sync_dist=True, batch_size=x.shape[0])
            self.log('test/local_mse', local_mse, sync_dist=True, batch_size=x.shape[0])

        if self.test_penetration:
            tgt_joint_loc = y['tgt_static']['joint_locations']
            for clip_idx, sample in enumerate(x_hat):
                armature = MixamoBodyArmature(MixamoBodyArmature._standard_joint_names, parents, tgt_rest_verts[clip_idx].detach().cpu().numpy(), tgt_faces[clip_idx].detach().cpu().numpy(), tgt_lbs_weights[clip_idx].detach().cpu().numpy(), tgt_joint_loc[clip_idx].detach().cpu().numpy())
                armature.joint_rotations = sample.unsqueeze(0)
                armature.root_locations = root_translation[clip_idx].reshape(1, -1, 3)
                sample_pr.append(armature.penetration_ratio())
                armature.joint_rotations = x[clip_idx].unsqueeze(0)

            self.log('test/penetration_ratio', np.mean([p[1] for p in sample_pr]), sync_dist=True, batch_size=x.shape[0])

    def contact_precision(self, motion_A, motion_B, static_A, static_B):
        def masked_l2(a, b, mask):
            b = torch.where(b < a, a, b) # if the predicted distance is smaller than the ground truth, loss is zero
            mse_loss = (a - b) ** 2
            mse_loss = (mse_loss * mask.float()).sum()
            non_zero_elements = mask.sum()
            mse_loss = mse_loss / (non_zero_elements + 1e-8)
            return mse_loss.mean()
        T = motion_A.shape[1]
        skin_sensor_B = SkinnableSensor(static_B['normalized_body_sensor_locations'].unsqueeze(1).expand(-1, T, -1, -1), static_B['body_sensor_tns'].unsqueeze(1).expand(-1, T, -1, -1, -1), static_B['normalized_joint_locations'].unsqueeze(1).expand(-1, T, -1, -1), static_B['parents'], static_B['body_sensor_weights'].unsqueeze(1).expand(-1, T, -1, -1))
        sensor_mask_B = static_B['body_sensor_mask']
        posed_sensor_locations_B, posed_sensor_tns_B = skin_sensor_B.skin(motion_B)
        with torch.no_grad():
            _, src_relative_coords, src_sparse_indices, src_ef_vel, _, src_sparse_dist, src_ef_pose_global = self.get_cond_inp({'src_motion': motion_A, **static_A})
            pred_relative_coords, pred_sparse_mask = indices2relative_coords(self.group_pairs, posed_sensor_locations_B, posed_sensor_tns_B, sensor_mask_B, src_sparse_indices)
        contact_precision = []
        for gt_dist, pred_coords, cur_sparse_mask, cur_group_pair in zip(src_sparse_dist, pred_relative_coords, pred_sparse_mask, self.group_pairs):
            pred_dist = pred_coords.norm(dim=-1)
            cur_obs_limb_name = cur_group_pair[0][0][0].split('_')[0]
            src_limb_radius = static_A['normalized_limb_radius'][cur_obs_limb_name].reshape(-1, 1, 1, 1)
            tgt_limb_radius = static_B['normalized_limb_radius'][cur_obs_limb_name].reshape(-1, 1, 1, 1)
            close_mask = gt_dist < src_limb_radius * 0.1
            contact_precision.append(masked_l2(gt_dist/tgt_limb_radius, pred_dist/tgt_limb_radius, close_mask & cur_sparse_mask))
        contact_precision = torch.stack(contact_precision, dim=0)
        return contact_precision


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, batch_first=False):
        # not used in the final model
        if batch_first:
            x = x + self.pe.transpose(0, 1)[:, :x.shape[1]]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, njoints, nfeats = x.shape
        x = x.reshape(bs, nframes, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        bs, nframes = output.shape[:2]
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(bs, nframes, self.njoints, self.nfeats)
        return output


class MotionDis(nn.Module):
    def __init__(self, only_body=False, dropout=0.2):
        super(MotionDis, self).__init__()

        njoints = 25 if only_body else 65

        self.seq = nn.Sequential(
            OrderedDict(
                [
                    ('dropout0', nn.Dropout(dropout)),
                    ('h0', nn.Conv1d(6*njoints, 16, kernel_size=3, padding='same', stride=1)),
                    ('acti0', nn.LeakyReLU(0.2)),
                    ('h1', nn.Conv1d(16, 32, kernel_size=3, padding='same', stride=1)),
                    ('bn1', nn.BatchNorm1d(32)),
                    ('acti1', nn.LeakyReLU(0.2)),
                    ('h2', nn.Conv1d(32, 64, kernel_size=3, padding='same', stride=1)),
                    ('bn2', nn.BatchNorm1d(64)),
                    ('acti2', nn.LeakyReLU(0.2)),
                    ('h3', nn.Conv1d(64, 64, kernel_size=3, padding='same', stride=1)),
                    ('bn3', nn.BatchNorm1d(64)),
                    ('acti3', nn.LeakyReLU(0.2)),
                    ('h4', nn.Conv1d(64, 1, kernel_size=1, padding='same', stride=1)),
                    ('sigmoid', nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        # x: B, T, J, 6
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1).transpose(1, 2)
        y = self.seq(x)
        return y.view(B, T)
