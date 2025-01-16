from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
import torch

from data_loaders.mret import MRetDataModule
from model.retnet import RetNet


class RetNetCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_optimizer_args(nested_key='optim', link_to='model.optim_init')
        parser.link_arguments('model.num_ring_per_bone', 'data.num_ring_per_bone')
        parser.link_arguments('model.num_point_per_ring', 'data.num_point_per_ring')
        parser.link_arguments('model.test_penetration', 'data.test_penetration')


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    RetNetCLI(RetNet, MRetDataModule)


if __name__ == '__main__':
    main()
