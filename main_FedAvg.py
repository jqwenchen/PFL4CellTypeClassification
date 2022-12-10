import matplotlib.pyplot as plt
import copy
import time
import numpy as np
import json
import os
from torchvision import datasets, transforms
import torch
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import ClassicNN, scDGN
from models.Fed import Weight_Averaged_Aggregation
from models.validate import validate

if __name__ == '__main__':

    N_CELL_TYPES = {'scquery': 39, 'pbmc': 10, 'pancreas_0': 13, 'pancreas_1': 13, 'pancreas_2': 13, 'pancreas_3': 13,
                    'pancreas_4': 13, 'pancreas_5': 13, 'UWB': 2}

    N_GENES = {'scquery': 20499, 'pbmc': 3000, 'pancreas_0': 3000, 'pancreas_1': 3000, 'pancreas_2': 3000, 'pancreas_3': 3000,
               'pancreas_4': 3000, 'pancreas_5': 3000, 'UWB': 55}
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model

    if args.model == 'ClassicNN':
        net_glob = ClassicNN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
                             l_dim=N_CELL_TYPES[args.dataset]).to(args.device)

    elif args.model == 'scDGN':
        net_glob = scDGN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
                         dim_label=N_CELL_TYPES[args.dataset], dim_domain=args.dim3).to(args.device)
    else:
        net_glob = None
        exit('Error: unrecognized model')


    print(net_glob)

    net_glob.train()

    # FL training
    train_acc_arr = np.array([])
    train_loss_arr = np.array([])
    val_acc_arr = np.array([])
    val_loss_arr = np.array([])

    np.random.seed(42)
    for iter in range(args.rounds):
        # copy weights
        w_glob = net_glob.state_dict()

        # sample clients
        if args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]
            idxs_users = np.arange(args.num_users)
        else:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            w_locals = [w_glob for i in range(idxs_users.shape[0])]

        for idx, user_id in enumerate(idxs_users):
            dataset_train = datasets.scDGN('data/{}/'.format(args.dataset), user_id=user_id, mode='train',
                                           transform=transforms.ToTensor())

            local = LocalUpdate(args=args, dataset=dataset_train)
            begin = time.time()
            w = local.train(net=copy.deepcopy(net_glob).to(args.device), param=w_glob)
            end = time.time()
            w_locals[idx] = copy.deepcopy(w)
            print('Round {} user {} end training, training time is {}min'.format(iter, user_id, format((end-begin)/60, '.2f')))

        # get num of training samples per user

        if iter == 0:
            filename = os.listdir('data/{}/train/'.format(args.dataset))[0]
            with open('data/{}/train/{}'.format(args.dataset, filename), 'r') as inf:
                cdata = json.load(inf)
            num_samples_per_user = np.array(cdata['num_samples'])
        # aggregates weights
        w_glob = Weight_Averaged_Aggregation(w_locals, num_samples_per_user[idxs_users])

        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)

        # validate per round
        w_locals = [w_glob for i in range(args.num_users)]
        train_acc, train_loss = validate(args, net_glob, w_locals, mode='train')
        val_acc, val_loss = validate(args, net_glob, w_locals, mode='val')

        train_acc_arr = np.append(train_acc_arr, train_acc)
        train_loss_arr = np.append(train_loss_arr, train_loss)


        val_acc_arr = np.append(val_acc_arr, val_acc)
        val_loss_arr = np.append(val_loss_arr, val_loss)

        print('Round {} -> average train accuracy: {}%, average train loss: {}'.format(iter, format(train_acc, '.2f'),
                                                                                  format(train_loss, '.4f')))

        print('Round {} -> average val accuracy: {}%, average val loss: {}'.format(iter, format(val_acc, '.2f'),
                                                                                  format(val_loss, '.4f')))


    # report final results on test set
    w_locals = [w_glob for i in range(args.num_users)]
    test_acc, test_loss = validate(args, net_glob, w_locals, mode='test')
    print('Final -> average test accuracy: {}%, average test loss: {}'.format(format(test_acc, '.2f'),
                                                                            format(test_loss, '.4f')))


    # plot acc and loss curve
    plt.figure(figsize=(10, 10))
    for idx, y_label in enumerate(['Training Accuracy', 'Training Loss', 'Val Accuracy', 'Val Loss']):
        ax = plt.subplot(2, 2, idx+1)
        X = np.arange(1, train_acc_arr.shape[0] + 1)
        if idx == 0:
            ax.plot(X, train_acc_arr)
        elif idx == 1:
            ax.plot(X, train_loss_arr)
        elif idx == 2:
            ax.plot(X, val_acc_arr)
        elif idx == 3:
            ax.plot(X, val_loss_arr)

        ax.set(ylabel=y_label, xlabel='Global Rounds')

    plt.show()