import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # -- model
    parser.add_argument('--model', type=str, default='basemodel', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='target dataset used for training')
    parser.add_argument('--sub_dataset', type=str, default='cifar100', help='dummy dataset used for training only for FedDU')

    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--init_type', type=str, default='xavier', help='init type: xavier, layerwise')

    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')

    parser.add_argument('--alg', type=str, default='fedavg',
                        help='this version code provides fedavg, fedls, feddu')
    parser.add_argument('--comm_round', type=int, default=300, help='number of maximum communication round')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")

    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./log_test/", help='log directory path')
    parser.add_argument('--tbdir', type=str, required=False, default="./tb_test/", help='tensorboard directory path')
    parser.add_argument('--exp_flag', type=str, required=False, default="./errors_test/", help='flags for saving results')

    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')

    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')

    parser.add_argument('--smoothing', type=float, default=0.1, help='smoothing factor for fedls')
    parser.add_argument('--sub_n', type=int, default=5000, help='dummy data points for feddu ')
    parser.add_argument('--loss_type', type=str, default='kl01', help='loss type')

    parser.add_argument('--data_aug', action='store_true', help="use data augmentation")
    parser.add_argument('--lr_decay_gamma', type=float, default=0.995, help='learning rate decay')
    parser.add_argument('--momentum_avg', type=str2bool, default=True, help='momentum averaging')

    parser.add_argument('--lr_decay_stepsize', nargs='+', default=[150, 225],
                        help='decay step; usage: --lr_decay stepsize 150 200 300')
    parser.add_argument('--nesterov', type=str2bool, default=True,
                        help='nesterov')

    args = parser.parse_args()
    return args