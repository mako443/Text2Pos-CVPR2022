import argparse
from argparse import ArgumentParser
import os
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description='PoseRefer models and ablations')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='K360')
    parser.add_argument('--path_retrieval', type=str, help="The path to the Cell-Retrieval model")
    parser.add_argument('--path_matching', type=str, help="The path to the Hints-to-Objects matching model")
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--pad_size', type=int, default=16)

    args = parser.parse_args()

    assert osp.isfile(args.path_retrieval)
    assert osp.isfile(args.path_matching)

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)