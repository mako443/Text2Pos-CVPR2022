import argparse
from argparse import ArgumentParser
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description='PoseRefer models and ablations')

    # General
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_distractors', default='all')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='K360')
    parser.add_argument('--base_path', type=str, default='./data/kitti360_shifted')

    # Model
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--use_features', nargs='+', default=['class', 'color', 'position'])
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--variation', type=int, default=0)
    parser.add_argument('--lr_idx', type=int)

    # SuperGlue
    parser.add_argument('--sinkhorn_iters', type=int, default=50)
    parser.add_argument('--num_mentioned', type=int, default=6)
    parser.add_argument('--pad_size', type=int, default=16)

    # Cell retrieval
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--ranking_loss', type=str, default='pairwise')
    # parser.add_argument('--cell_size', type=float, default=60)
    # parser.add_argument('--cell_stride', type=float, default=40)

    # Object-encoder / PointNet
    parser.add_argument('--pointnet_layers', type=int, default=3)
    parser.add_argument('--pointnet_variation', type=int, default=0)
    parser.add_argument('--pointnet_numpoints', type=int, default=512)
    parser.add_argument('--pointnet_transform', type=int, default=0)
    parser.add_argument('--pointnet_path', type=str, default='./checkpoints/pointnet_K360_lr2_t1_p512.pth')
    parser.add_argument('--pointnet_freeze', action='store_true')
    parser.add_argument('--pointnet_embed', action='store_true')    
    parser.add_argument('--pointnet_features', type=int, default=2)

    # Others
    parser.add_argument('--epochs', type=int, default=16)
    # parser.add_argument('--alpha_obj_ref', type=float, default=2.0)
    # parser.add_argument('--alpha_target_class', type=float, default=100.0)
    # parser.add_argument('--alpha_obj_class', type=float, default=1.0)
    # parser.add_argument('--alpha_offset', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--scene_names', nargs='+', default=[])
    

    args = parser.parse_args()
    try:
        args.num_distractors = int(args.num_distractors)
    except:
        pass

    try:
        args.max_batches = int(args.max_batches)
    except:
        pass    
    
    args.dataset = args.dataset.upper()
    assert args.dataset in ('S3D', 'K360')
    
    assert args.ranking_loss in ('triplet', 'pairwise', 'hardest')
    for feat in args.use_features:
        assert feat in ['class', 'color', 'position'], "Unexpected feature"

    if args.pointnet_path:
        assert osp.isfile(args.pointnet_path)

    assert osp.isdir(args.base_path)

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)