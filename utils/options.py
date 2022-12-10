import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # pancreas_0 example
    '''
    --rounds=50 --num_user=9 --frac=1.0 --local_ep=5 --local_bs=32 --lr=0.001 --model=ClassicNN --dataset=pbmc
    --gpu=0 --all_client --lamda=1.0 --dim1=1136 --dim2=100
    
    # UWB example: change 'X' to 'x' in scDGN.py
    --rounds=50 --num_user=8 --frac=1.0 --local_ep=2 --local_bs=5 --lr=0.01 --model=ClassicNN --dataset=UWB
    --gpu=0 --all_client --lamda=1.0 --dim1=32 --dim2=16
    '''
    # federated arguments
    parser.add_argument('--rounds', type=int, default=50, help="rounds of training") # each communication between sever and client
    parser.add_argument('--num_users', type=int, default=4, help="number of users: K") # How many clients
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C") # Clients who participate in training for each round
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=99999, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=1.0, help="SGD momentum (default: 0.9)")
    parser.add_argument('--decay', type=float, default=0.0, help="weight decay")
    parser.add_argument('--lamda', type=float, default=1.0, help="Lamda in Ditto")
    # model arguments
    parser.add_argument('--model', type=str, default='ClassicNN', help='model name')
    parser.add_argument('--dim1', type=int, default=32, help='fc1_layer')
    parser.add_argument('--dim2', type=int, default=16, help='fc2_layer')
    # parser.add_argument('--dim3', type=int, default=64, help='dim_domain')
    # parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                     help='comma-separated kernel size to use for convolution')
    # parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    # parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    # parser.add_argument('--max_pool', type=str, default='True',
    #                     help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='pancreas_0', help="name of dataset")
    # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    # parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    return args

