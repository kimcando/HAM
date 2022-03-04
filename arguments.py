import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    # -- model
    parser.add_argument('--model', type=str, default='resnet8_gn', help='neural network used in training')
    parser.add_argument('--init_type', type=str, default='xavier', help='init type: xavier, layerwise')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')

    # -- algorithm
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='this version code provides fedavg, fedls, feddu')
    parser.add_argument('--smoothing', type=float, default=0.1, help='smoothing factor for fedls')
    parser.add_argument('--sub_n', type=int, default=5000, help='dummy data points for feddu ')

    # -- dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='target dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="./input/", help="Data directory")
    parser.add_argument('--sub_dataset', type=str, default='cifar100', help='dummy dataset used for training only for FedDU')
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')

    parser.add_argument('--input_size', type=int, default=32, help='target dataset used for training')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes == classifer output')

    parser.add_argument('--data_aug', action='store_true', help="use data augmentation")

    # -- optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.995, help='learning rate decay for exponential')
    parser.add_argument('--lr_decay_stepsize', nargs='+', default=[150, 225],
                        help='decay step; usage: --lr_decay stepsize 150 200 300')

    parser.add_argument('--momentum_avg', type=str2bool, default=True, help='momentum averaging')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")

    parser.add_argument('--nesterov', type=str2bool, default=True,
                        help='nesterov')

    # -- training details
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--comm_round', type=int, default=300, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')

    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')

    # -- config
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')

    parser.add_argument('--wandb_log', type=str2bool, required=False, default=True, help="Data directory")
    parser.add_argument('--exp_name', type=str, required=False, default="resnet8gn_single", help="wandb experiments name")

    parser.add_argument('--logdir', type=str, required=False, default="./log_test/", help='log directory path')
    parser.add_argument('--tbdir', type=str, required=False, default="./tb_test/", help='tensorboard directory path')
    parser.add_argument('--exp_flag', type=str, required=False, default="./errors_test/", help='flags for saving results')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    args = parser.parse_args()
    return args