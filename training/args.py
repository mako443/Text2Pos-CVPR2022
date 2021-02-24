import argparse
from argparse import ArgumentParser

def parse_arguments():
    parser = argparse.ArgumentParser(description='PoseRefer models and ablations')

    parser.add_argument('--num_distractors', default='all')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--alpha_obj_ref', default=2.0)
    parser.add_argument('--alpha_target_class', default=5.0)
    parser.add_argument('--alpha_obj_class', default=1.0)
    parser.add_argument('--lr_gamma', type=float, default=1.0)

    args = parser.parse_args()
    try:
        args.num_distractors = int(args.num_distractors)
    except:
        pass

    try:
        args.max_batches = int(args.max_batches)
    except:
        pass    

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)