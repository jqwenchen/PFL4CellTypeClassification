import matplotlib.pyplot as plt
import copy
import time
import numpy as np
import json
import os
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import ClassicNN, scDGN, ConvNet1D, CNN
from models.Fed import Weight_Averaged_Aggregation
from models.validate import validate
from utils.vis_util import plot_pca_ct, plot_pca_all, plot_pca, extract_rep
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    best_acc = 0
    N_CELL_TYPES = {'scquery': 39, 'pbmc': 10, 'pancreas_0': 13, 'pancreas_1': 13, 'pancreas_2': 13, 'pancreas_3': 13,
                    'pancreas_4': 13, 'pancreas_5': 13, 'UWB': 2}

    N_GENES = {'scquery': 20499, 'pbmc': 3000, 'pancreas_0': 3000, 'pancreas_1': 3000, 'pancreas_2': 3000, 'pancreas_3': 3000,
               'pancreas_4': 3000, 'pancreas_5': 3000, 'UWB': 55}
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(
            './checkpoint/cifar10/ckpt.pth' + '_' + args.model + '_' + str(args.epoch) + '_' + args.mixup + '_'
            + str(args.seed))
        # net.load_state_dict(checkpoint['net'])
        net_glob = checkpoint['net']
        best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        if args.model == 'ClassicNN':
            net_glob = ClassicNN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
                                 l_dim=N_CELL_TYPES[args.dataset]).to(args.device)

        elif args.model == 'scDGN':
            net_glob = scDGN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
                             dim_label=N_CELL_TYPES[args.dataset], dim_domain=args.dim3).to(args.device)

        elif args.model == 'ConvNet1D':
            net_glob = ConvNet1D(in_channels=1, out_channels=8, n_kernel=100)

        elif args.model == 'CNN':
            net_glob = ClassicNN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
                                 l_dim=N_CELL_TYPES[args.dataset]).to(args.device)

        else:
            net_glob = None
            exit('Error: unrecognized model')


    # if args.model == 'ClassicNN':
    #     net_glob = ClassicNN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
    #                          l_dim=N_CELL_TYPES[args.dataset]).to(args.device)
    #
    # elif args.model == 'scDGN':
    #     net_glob = scDGN(d_dim=N_GENES[args.dataset], dim1=args.dim1, dim2=args.dim2,
    #                      dim_label=N_CELL_TYPES[args.dataset], dim_domain=args.dim3).to(args.device)
    # else:
    #     net_glob = None
    #     exit('Error: unrecognized model')


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
        if iter == 0:
            # Initialise personalised model
            w_per = [w_glob for i in range(args.num_users)]

        # sample clients
        if args.all_clients:
            print("Aggregation over all clients")
            # Initialise local model for global train
            w_locals = [w_glob for i in range(args.num_users)]
            idxs_users = np.arange(args.num_users)
        else:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # Initialise local model for global train
            w_locals = [w_glob for i in range(idxs_users.shape[0])]

        for idx, user_id in enumerate(idxs_users):
            begin = time.time()
            dataset_train = datasets.scDGN('data/{}/'.format(args.dataset), user_id=user_id, mode='train',
                                           transform=transforms.ToTensor())

            local = LocalUpdate(args=args, dataset=dataset_train)
            # local train for updating personalised model
            w_p = local.train_p(net=copy.deepcopy(net_glob).to(args.device), param=w_per[user_id])

            # local train for updating global model
            w = local.train(net=copy.deepcopy(net_glob).to(args.device), param=w_glob)

            end = time.time()
            w_locals[idx] = copy.deepcopy(w)
            w_per[idx] = copy.deepcopy(w_p)
            print('Round {} user {} end training, training time is {}min'.format(iter, user_id, format((end-begin)/60, '.2f')))

        # get num of training samples per user

        if iter == 0:
            filename = os.listdir('data/{}/train/'.format(args.dataset))[0]
            with open('data/{}/train/{}'.format(args.dataset, filename), 'r') as inf:
                cdata = json.load(inf)
            num_samples_per_user = np.array(cdata['num_samples'])

        # aggregates weights
        # Robust Aggregation method
        # krum_Agg()
        w_glob = Weight_Averaged_Aggregation(w_locals, num_samples_per_user[idxs_users])

        # copy weight to net_glob
        #net_glob.load_state_dict(w_glob)

        # validate per round
        train_acc, train_loss = validate(args, net_glob, w_per, mode='train')
        val_acc, val_loss = validate(args, net_glob, w_per, mode='val')

        # test per round
        test_acc, test_loss = validate(args, net_glob, w_per, mode='test')
        if val_acc > best_acc:
            print("Saving...")
            state = {
                # 'net': net.state_dict(),
                'net_glob': net_glob,
                "net_per": w_per,
                'acc': val_acc,
                'epoch': args.rounds,
            }
            if not os.path.isdir('checkpoint/'):
                os.mkdir('checkpoint/')
            torch.save(state,
                       './checkpoint/ckpt.pth' + '_' + args.model + '_' + str(args.rounds) + '_' + str(
                           args.dataset) + '.pth')
            best_acc = val_acc
        if args.rounds - 1:
            state = {
                # 'net': net.state_dict(),
                'net': net_glob,
                "net_per": w_per,
                'acc': val_acc,
                'epoch': args.rounds,
            }
            if not os.path.isdir('checkpoint/'):
                os.mkdir('checkpoint/')
            torch.save(state, './checkpoint/ckpt.pth_' + args.model + '_' + str(args.rounds) + '_' + str(
                args.dataset) + 'last_model.pth')

        train_acc_arr = np.append(train_acc_arr, train_acc)
        train_loss_arr = np.append(train_loss_arr, train_loss)


        val_acc_arr = np.append(val_acc_arr, val_acc)
        val_loss_arr = np.append(val_loss_arr, val_loss)

        print('Round {} -> average train accuracy: {}%, average train loss: {}'.format(iter, format(train_acc, '.2f'),
                                                                                  format(train_loss, '.4f')))

        print('Round {} -> average val accuracy: {}%, average val loss: {}'.format(iter, format(val_acc, '.2f'),
                                                                                  format(val_loss, '.4f')))

        print('Round {} -> average test accuracy: {}%, average test loss: {}'.format(iter, format(test_acc, '.2f'),
                                                                                   format(test_loss, '.4f')))



    # report final results on test set
    test_acc, test_loss = validate(args, net_glob, w_per, mode='test')
    print('Final -> average test accuracy: {}%, average test loss: {}'.format(format(test_acc, '.2f'),
                                                                            format(test_loss, '.4f')))


    
    # plot acc and loss curve
    # plt.figure(figsize=(10, 10))
    # for idx, y_label in enumerate(['Training Accuracy', 'Training Loss', 'Val Accuracy', 'Val Loss']):
    #     ax = plt.subplot(2, 2, idx+1)
    #     X = np.arange(1, train_acc_arr.shape[0] + 1)
    #     if idx == 0:
    #         ax.plot(X, train_acc_arr)
    #     elif idx == 1:
    #         ax.plot(X, train_loss_arr)
    #     elif idx == 2:
    #         ax.plot(X, val_acc_arr)
    #     elif idx == 3:
    #         ax.plot(X, val_loss_arr)
    #
    #     ax.set(ylabel=y_label, xlabel='Global Rounds')
    # plt.savefig('./save/nn_{}_{}.png'.format(args.dataset, args.model))
    #plt.show()

    n_genes = 100
    cell_names = ['cell' + str(i) for i in range(N_CELL_TYPES[args.dataset])]
    gene_names = ['gene' + str(i) for i in range(N_GENES[args.dataset])]
    out_path = 'eval/'

    all_representations = []
    all_labels = []
    for idx in range(len(w_per)):
        net_glob.load_state_dict(w_per[idx])

        NN_representations, NN_labels, NN_domains = extract_rep(net_glob, idx, args)
        all_representations.append(NN_representations)
        all_labels.append(NN_labels)

        # plot pca for each user
        if not os.path.exists('eval/user{}/{}'.format(idx, args.dataset)):
            os.mkdir('eval/user{}/{}'.format(idx, args.dataset))
            os.mkdir('eval/user{}/{}/NN'.format(idx, args.dataset))
        plot_pca(NN_representations, NN_labels, NN_domains, 'NN', idx, expname=args.dataset,
                 nlabels=N_CELL_TYPES[args.dataset])

        # feature importance
        for cate_id in range(N_CELL_TYPES[args.dataset]):
            counts = 0.
            mean_value_ori = np.zeros(N_GENES[args.dataset])
            # analysis NN: backpropogate the gradient of beta and quiescent_stellate 
            mean_value = np.copy(mean_value_ori)
            resulted_values_NN = []
            for i in range(n_genes):
                net_glob.zero_grad()
                mean_value_variable = Variable(torch.Tensor(mean_value).view(1, -1).to(args.device), requires_grad=True)
                act_value = net_glob(mean_value_variable)
                diff_out = act_value[:,cate_id:cate_id+1]
                diff_out.backward()
                mean_value_variable.data.add_(mean_value_variable.grad.data)
                mean_value = mean_value_variable.data.cpu().numpy()
                resulted_values_NN.append(mean_value)


            # save DE gene names 
            cell_type = cell_names[cate_id] #eval/NN_cell_names[cate_id]_100.txt
            if not os.path.exists('./eval/user{}'.format(idx)):
                os.makedirs('./eval/user{}'.format(idx))

            with open(os.path.join(out_path+'user{}/'.format(idx), 'NN_%s_%d.txt'%(cell_type, n_genes)), 'w') as fw:
                diff_NN = (resulted_values_NN[-1]-mean_value_ori)[0]
                diff_NN_ids = np.abs(diff_NN).argsort()[-n_genes:][::-1] # final 100 idx
                for index in diff_NN_ids:
                    fw.write('%s\n'%(gene_names[index]))

    # t-SNE plot
    fig = plt.figure()
    for idx in range(len(all_labels)):
        ts = TSNE(n_components=2, init='pca', random_state=0)
        data = ts.fit_transform(all_representations[idx])
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
        # 遍历所有样本
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(idx), color=plt.cm.Set1(all_labels[idx][i] / N_CELL_TYPES[args.dataset]),
                     fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    plt.title('t-SNE', fontsize=14)
    plt.show()