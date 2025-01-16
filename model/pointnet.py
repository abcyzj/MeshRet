from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
       x = (x - x.mean(dim=self.dim, keepdim=True)) / torch.sqrt(x.var(dim=self.dim, keepdim=True)+self.eps)
       return x


class STPointNetBlock(nn.Module):
    def __init__(self, input_dim, layer_dims, layernorm=True, global_feat=False,
                 transposed=False):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.layernorm = layernorm
        self.global_feat = global_feat
        self.transposed = transposed
        self.activation = nn.ReLU()

        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, input_dim)

        self.conv_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx], layer_dims[idx + 1], 1))
        if layernorm:
            self.ln = LayerNorm(dim=1)

        if not global_feat:
            self.last_conv = nn.Conv1d(layer_dims[-1]*2, layer_dims[-1]*2, 1)

    def forward(self, x):
        if self.transposed:
            batch_size, window_size, num_points = x.size(0), x.size(2), x.size(3)
            x = x.view(batch_size, self.input_dim, window_size*num_points)
        else:
            batch_size, window_size, num_points = x.size(0), x.size(1), x.size(2)
            x = x.permute(0, 3, 1, 2).view(batch_size, self.input_dim, window_size*num_points)

        x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False:
            local_features = x.view(batch_size, -1, window_size, num_points)

        for idx in range(1, len(self.conv_layers) - 1):
            x = self.activation(self.conv_layers[idx](x))

        x = self.conv_layers[-1](x)

        x = x.view(-1, self.layer_dims[-1], window_size, num_points)
        x = torch.max(x, 3)[0]

        if self.global_feat:
            if self.layernorm:
                return self.ln(x)
            return x

        x = x.view(-1, self.layer_dims[-1], window_size, 1).repeat(1, 1, 1, num_points)

        x = torch.cat((x, local_features), dim=1)
        x = x.view(batch_size, -1, window_size*num_points) + self.last_conv(x.view(batch_size, -1, window_size*num_points))
        x = x.view(batch_size, -1, window_size, num_points)

        if self.layernorm:
            return self.ln(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)


class PointNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, layer_dims: List[int] = [32, 64, 128, 256]):
        super().__init__()

        self.blocks = nn.ModuleList([
            STPointNetBlock(input_dim=input_dim, layer_dims=[layer_dims[0], layer_dims[0], layer_dims[0]], transposed=False),
            STPointNetBlock(input_dim=layer_dims[0]*2, layer_dims=[layer_dims[1], layer_dims[1], layer_dims[1]], transposed=True),
            STPointNetBlock(input_dim=layer_dims[1]*2, layer_dims=[layer_dims[2], layer_dims[2], layer_dims[2]], transposed=True),
            STPointNetBlock(input_dim=layer_dims[2]*2, layer_dims=[layer_dims[3], layer_dims[3], layer_dims[3]], transposed=True, global_feat=True)
        ])

        self.latent_fc = nn.Conv1d(layer_dims[3], latent_dim, 1)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = F.relu(block(x))
        x = self.latent_fc(x).permute(0, 2, 1)
        return x


class SensorNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.per_sensor_pointnet = PointNet(input_dim=input_dim, latent_dim=16, layer_dims=[8, 8, 8, 16])
        self.sensor_pointnet = PointNet(input_dim=16, latent_dim=latent_dim, layer_dims=[32, 64, 128, 256])


    def forward(self, x: torch.Tensor):
        '''
        x: (B, T, S, N, D) or (B, T, S, D)
        '''
        if len(x.size()) == 4:
            batch_size, window_size, num_sensors, input_dim = x.size()
            num_points = 1
        else:
            batch_size, window_size, num_sensors, num_points, input_dim = x.size()
        x = x.view(batch_size, window_size*num_sensors, num_points, input_dim)
        x = self.per_sensor_pointnet(x)
        x = x.view(batch_size, window_size, num_sensors, -1)
        x = self.sensor_pointnet(x)
        return x
