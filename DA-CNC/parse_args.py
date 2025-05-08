import argparse
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser(description='AdaGIn')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 1)')
    parser.add_argument('--source_dataset', type=str, default='citationv1_degree_0.4', help="Blog1 Blog2")
    parser.add_argument('--target_dataset', type=str, default='dblpv7_degree_0.4',help="dblpv7  acmv9  citationv1")
    parser.add_argument('--source_dataset_del', type=str, default='dblpv7_0.2', help="Blog1 Blog2")
    parser.add_argument('--target_dataset_del', type=str, default='acmv9_0.2', help="dblpv7  acmv9  citationv1")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--epochs1', type=int, default=50)
    parser.add_argument('--lr_cly', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--aggregator_class', type=str, default='mean')
    parser.add_argument('--n_samples', type=str, default='10,10')
    # parser.add_argument('--n_samples1', type=str, default='10,10,10')
    parser.add_argument('--output_dims', type=str, default='2048,128')
    parser.add_argument('--arch_cly', type=str, default="", help='node classifier architecture')
    parser.add_argument('--arch_disc', type=str, default="512-64-16", help='domain discriminator architecture')
    parser.add_argument('--is_social_net', action='store_true', help='whether to analysis the social networks')
    parser.add_argument('--is_blog', default=True, action='store_true', help='whether to analysis the blog networks, i.e., Blog1 and Blog2.')
    parser.add_argument('--dgi_param', type=float, default=0.1)
    parser.add_argument('--cdan_param', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--k', type=int, default=9, help='num of node neighbor')
    parser.add_argument('--instance_temperature', type=float, default=0.1)
    parser.add_argument('--cluster_temperature', type=float, default=0.1)
    parser.add_argument('--mi_param', type=float, default=0.001)
    args = parser.parse_args()

    return args
