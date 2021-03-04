import argparse
from argparse import ArgumentParser

def parse_arguments():
    parser = argparse.ArgumentParser(description='PoseRefer models and ablations')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_distractors', default='all')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--use_color', action='store_true')
    
    # DGCNN
    parser.add_argument('--use_layers', type=str, default='all')
    parser.add_argument('--k', type=int, default=4)

    # SuperGlue
    parser.add_argument('--sinkhorn_iters', type=int, default=40)

    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--alpha_obj_ref', type=float, default=2.0)
    parser.add_argument('--alpha_target_class', type=float, default=100.0)
    parser.add_argument('--alpha_obj_class', type=float, default=1.0)
    parser.add_argument('--alpha_offset', type=float, default=0.01)
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