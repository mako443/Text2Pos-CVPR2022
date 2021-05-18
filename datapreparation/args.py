import argparse
from argparse import ArgumentParser
import os
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description='K360 data preparation')

    parser.add_argument('--path_in', type=str, default='./data/kitti360')
    parser.add_argument('--path_out', type=str, default='./data/k360')
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--cell_size', type=int, default=30)
    parser.add_argument('--cell_dist', type=int, default=30, help="The minimum distance between two cells")

    parser.add_argument('--pose_dist', type=int, default=30)
    parser.add_argument('--shift_poses', action='store_true')  

    args = parser.parse_args()
    
    assert osp.isdir(args.path_in)
    assert osp.isdir(osp.join(args.path_in, 'data_3d_semantics', args.scene_name)), f'Input folder not found {osp.join(args.path_in, "data_3d_semantics", args.scene_name)}'
    args.path_out = f'{args.path_out}_cs{args.cell_size}_cd{args.cell_dist}_pd{args.pose_dist}_sh{args.shift_poses}'

    print(f'Folders: {args.path_in} -> {args.path_out}')
    if not osp.isdir(args.path_out):
        print(f'Creating folders for {args.path_out}')
        os.mkdir(args.path_out)
        os.mkdir(osp.join(args.path_out, 'cells'))
        os.mkdir(osp.join(args.path_out, 'poses'))

    return args
