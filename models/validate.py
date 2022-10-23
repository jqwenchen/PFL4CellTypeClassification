import numpy as np
from torchvision import datasets, transforms
from .test import test_img
def validate(args, net_glob, mode=None):
    nc_arr = np.array([])
    nall_arr = np.array([])
    acc_arr = np.array([])
    loss_arr = np.array([])
    for idx in range(args.num_users):
        if mode == 'train':
            dataset = datasets.scDGN('data/{}/'.format(args.dataset), user_id=idx, mode='train',
                                                 transform=transforms.ToTensor())

        elif mode == 'test':
            dataset = datasets.scDGN('data/{}/'.format(args.dataset), user_id=idx, mode='test',
                                                 transform=transforms.ToTensor())
        elif mode == 'val':
            dataset = datasets.scDGN('data/{}/'.format(args.dataset), user_id=idx, mode='val',
                                         transform=transforms.ToTensor())
        else:
            dataset = None
            print('invalid mode!')

        nc, nall, acc, loss = test_img(net_glob, dataset, args)
        nc_arr = np.append(nc_arr, nc)
        nall_arr = np.append(nall_arr, nall)
        acc_arr = np.append(acc_arr, acc)
        loss_arr = np.append(loss_arr, loss)

        # print('user {} end validating {}'.format(idx, mode))
    mean_acc = np.sum(nc_arr) / np.sum(nall_arr) * 100.0
    mean_loss = np.dot(loss_arr, nc_arr) / np.sum(nall_arr)

    return mean_acc, mean_loss