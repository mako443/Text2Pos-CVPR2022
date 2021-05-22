import argparse
from argparse import ArgumentParser
import os
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description='K360 data preparation')

    parser.add_argument('--path_in', type=str, default='./data/kitti360')
    parser.add_argument('--path_out', type=str, default='./data/k360')
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--num_mentioned', type=int, default=6)
    parser.add_argument('--describe_by', type=str, default='closest')

    parser.add_argument('--cell_size', type=int, default=30)
    parser.add_argument('--cell_dist', type=int, default=30, help="The minimum distance between two cells")
    parser.add_argument('--shift_cells', action='store_true')
    parser.add_argument('--grid_cells', action='store_true')

    parser.add_argument('--pose_dist', type=int, default=30)
    parser.add_argument('--pose_count', type=int, default=1)
    parser.add_argument('--shift_poses', action='store_true')  

    parser.add_argument('--describe_best_cell', action='store_true') 

    args = parser.parse_args()
    
    assert osp.isdir(args.path_in)
    assert osp.isdir(osp.join(args.path_in, 'data_3d_semantics', args.scene_name)), f'Input folder not found {osp.join(args.path_in, "data_3d_semantics", args.scene_name)}'
    
    if args.shift_cells:
        cells_text = 'Y'
    elif args.grid_cells:
        cells_text = 'G'
    else:
        cells_text = 'N'

    args.path_out = f'{args.path_out}_cs{args.cell_size}_cd{args.cell_dist}_sc{cells_text}_pd{args.pose_dist}_pc{args.pose_count}_sp{"Y" if args.shift_poses else "N"}_{args.describe_by}'
    if args.describe_best_cell:
        args.path_out += '_bestCell'

    print(f'Folders: {args.path_in} -> {args.path_out}')

    assert args.describe_by in ('closest', 'class', 'direction', 'random', 'all')
    assert args.shift_cells + args.grid_cells < 2 # Only of of them

    # Create dirs
    try:
        os.mkdir(args.path_out)
    except:
        pass
    try:
        os.mkdir(osp.join(args.path_out, 'cells'))
    except:
        pass
    try:
        os.mkdir(osp.join(args.path_out, 'poses'))
    except:
        pass


    return args
