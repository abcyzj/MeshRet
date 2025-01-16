from argparse import ArgumentParser, Namespace
import pickle
import pyvista as pv
import torch
from tqdm import tqdm
from trimesh import Trimesh

pv.start_xvfb()

def play_mesh(posed_verts: torch.Tensor, faces: torch.Tensor, output: str):
    posed_verts = posed_verts.detach().cpu().numpy()
    posed_verts[..., 2]
    faces = faces.detach().cpu().numpy()
    plotter = pv.Plotter()
    mesh = pv.wrap(Trimesh(posed_verts[0], faces, process=False))
    plotter.add_mesh(mesh, color='w')
    plotter.open_movie(output)
    plotter.show(interactive_update=True)
    for i in tqdm(range(posed_verts.shape[0])):
        mesh.points = posed_verts[i]
        plotter.camera_position = 'xy'
        plotter.write_frame()
    plotter.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    with open(args.input, 'rb') as f:
        posed_verts, faces = pickle.load(f)
    play_mesh(posed_verts, faces, args.output)


if __name__ == '__main__':
    main()
