import argparse


def build_args():
    parser = argparse.ArgumentParser('Multiview Deformable DETR (Shadow) Detector', add_help=False)
    parser.add_argument('--numbers', nargs='+', type=int, help='List of numbers')

    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--src_cams', default=[0, 1, 2, 3, 4, 5, 6], nargs='+', type=int)
    parser.add_argument('--src_nums', default=[1, 1, 1, 1, 1, 1, 1], nargs='+', type=int)
    parser.add_argument('--src_size', default=64, type=int)
    parser.add_argument('--tar_cams', default=[0, 1, 2, 3, 4, 5, 6], nargs='+', type=int)
    parser.add_argument('--tar_nums', default=[1, 1, 1, 1, 1, 1, 1], nargs='+', type=int)
    parser.add_argument('--tar_size', default=64, type=int)
    parser.add_argument('--src_sampler', default='uniformg', type=str, choices=['uniformg', 'uniforml', 'random', 'custom'])
    parser.add_argument('--tar_sampler', default='custom', type=str, choices=['uniformg', 'uniforml', 'random', 'custom'])
    parser.add_argument('--smoother', default=False, type=bool)
    parser.add_argument('--aggregator', default=None, type=str, choices=['mean', 'norm', None])
    parser.add_argument('--epsilon', default=None, type=float)
    parser.add_argument('--constrainer', default=None, type=str, choices=['l2', 'linf', None])
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--attack', default='CP', type=str, choices=['IP', 'OP', 'CP', 'SP', 'Clean'])
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--visualize', default=True, type=bool)

    return parser.parse_args()
