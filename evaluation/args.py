import argparse
from argparse import ArgumentParser
import os
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description='PoseRefer models and ablations')

    parser.add_argument('--purpose', type=str)

    # Paths
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='K360')
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--path_coarse', type=str, help="The path to the Cell-Retrieval model")
    parser.add_argument('--path_fine', type=str, help="The path to the Hints-to-Objects matching model")


    # Options
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--threshs', type=int, nargs='+', default=[5, 10, 15]) # Possibly over-write here when it becomes a list of tuples
    parser.add_argument('--pad_size', type=int, default=16)

    parser.add_argument('--use_test_set', action='store_true', help="Run test-set instead of the validation set.")
    
    parser.add_argument('--no_pc_augment', action='store_true')
    parser.add_argument('--num_mentioned', type=int, default=6)

    parser.add_argument('--plot_retrievals', action='store_true', help="Plot 3 success and fail examples, then quit.")
    parser.add_argument('--plot_matches', action='store_true')
    parser.add_argument('--coarse_only', action='store_true')

    # Oracles
    parser.add_argument('--coarse_oracle', action='store_true', help="Use gt-retrievals")
    parser.add_argument('--fine_oracle', action='store_true', help="Use perfect in-cell locations")

    # Object-encoder / PointNet
    parser.add_argument('--pointnet_numpoints', type=int, default=256)

    # Various - don't change these!
    parser.add_argument('--ranking_loss', type=str, default='pairwise')
    parser.add_argument('--regressor_cell', type=str, default='pose') # Pose or best
    parser.add_argument('--regressor_learn', type=str, default='center') # Center or closest
    parser.add_argument('--regressor_eval', type=str, default='center') # Center or closest    


    args = parser.parse_args()

    assert osp.isfile(args.path_coarse)
    assert osp.isfile(args.path_fine)
    if args.coarse_oracle:
        assert max(args.top_k) == 1, "Coarse oracle can only retrieved one best cell."

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)