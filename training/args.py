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
    parser.add_argument('--dim_ff', type=int, default=2048)
    # parser.add_argument('--use_color', action='store_true')
    parser.add_argument('--use_features', nargs='+', default=['class', 'color', 'position'])
    parser.add_argument('--shuffle', action='store_true')
    
    # DGCNN
    parser.add_argument('--use_layers', type=str, default='all')
    parser.add_argument('--k', type=int, default=4)

    # SuperGlue
    parser.add_argument('--sinkhorn_iters', type=int, default=40)

    # Cell retrieval
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--ranking_loss', type=str, default='pairwise')
    parser.add_argument('--cell_size', type=float, default=65.0)
    parser.add_argument('--cell_stride', type=float, default=65.0/3)

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
    
    assert args.ranking_loss in ('triplet', 'pairwise', 'hardest')
    for feat in args.use_features:
        assert feat in ['class', 'color', 'position'], "Unexpected feature"

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)