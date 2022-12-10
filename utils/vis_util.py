import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
CAND_COLORS = np.array(['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f',
                '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554',
                '#61a0a8', '#d48265', '#c23531'])
# pick the domains with enough cells to visualize
def is_valid(domains, cutoff):
    ds = np.unique(domains)
    if ds.shape[0]<2:
        return False
    valid_domain_count = 0 # number of domain with more than cutoff samples
    for d in ds:
        if np.where(domains==d)[0].shape[0] >= cutoff:
            valid_domain_count += 1
    if valid_domain_count < 2:
        return False
    return True


# plot pca visualization for certain domains with cell types colored
def plot_pca_ct(representations, labels, domains, modelname, expname='scquery', cutoff=20):
    if not os.path.exists('eval/%s/%s/pca2/'%(expname, modelname)):
        os.mkdir('eval/%s/%s/pca2/'%(expname, modelname))
    ndomains = np.unique(domains)
    for z_target in ndomains:
        indeces = np.where((domains==z_target))[0]
        # if len(indeces) == 0 or not is_valid(labels[indeces], cutoff):
        #     continue
        pca = PCA(n_components=2)
        pca.fit(representations[indeces])
        X_reduced = pca.transform(representations[indeces])
        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=5, alpha=0.6, c=labels[indeces])
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        # plt.show()
        plt.savefig('eval/%s/%s/pca2/%s_%s.png'%(expname, modelname, modelname, d.accessions_set[z_target]))
        plt.close()


# plot the pca visualization for the whole dataset
def plot_pca_all(representations, labels, domains, modelname, expname='scquery', nlabels=39, cutoff=20):
    indeces = np.arange(representations.shape[0])
    pca = PCA(n_components=2)
    pca.fit(representations[indeces])
    X_reduced = pca.transform(representations[indeces])

    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=5, alpha=0.6, c=CAND_COLORS[-3:][domains[indeces]])
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    plt.savefig('eval/%s/pca_domains_%s.png'%(expname, modelname))
    
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=5, alpha=0.6, c=CAND_COLORS[labels[indeces]])
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    plt.savefig('eval/%s/pca_labels_%s.png'%(expname, modelname))


# plot pca visualization for certain cell types with domains colored
def plot_pca(representations, labels, domains, modelname, id, expname='scquery', nlabels=39, cutoff=20):
    if not os.path.exists('eval/user%d/%s/%s/pca/'%(id, expname, modelname)):
        os.mkdir('eval/user%d/%s/%s/pca/'%(id, expname, modelname))
    fig = plt.figure(figsize=(10, 10))
    for y_target in range(nlabels):
        indeces = np.where((labels==y_target))[0]
        # if len(indeces) == 0 or not is_valid(domains[indeces], cutoff):
        #     continue
        pca = PCA(n_components=2)
        pca.fit(representations[indeces])
        X_reduced = pca.transform(representations[indeces])
        ax = plt.subplot()
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=200, alpha=0.6, c=CAND_COLORS[y_target])
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
    plt.legend(labels=['label'+str(i) for i in range(y_target+1)])
    plt.savefig('eval/user%d/%s/%s/pca/%s.png'%(id, expname, modelname, modelname))


# extract the representations from NN
# def extract_rep(t, d, scDGN=False):
#     representations = None
#     labels = None
#     domains = None
#     n_iter = len(d._train_y)//batch_size
#     t.D.eval()
#     rng_state = np.random.get_state()
#     for i in range(n_iter):
#         x = d._train_X[i*batch_size:(i+1)*batch_size] 
#         y = d._train_y[i*batch_size:(i+1)*batch_size]
#         X = Variable(torch.cuda.FloatTensor(x))
#         if scDGN:
#             z = d._train_acc[i*batch_size:(i+1)*batch_size]
#             f_X = t.D(X, X, mode='eval')
#         else:
#             z = d._train_z[i*batch_size:(i+1)*batch_size]
#             f_X = t.D(X, mode='eval')
#         if representations is None:
#             representations = f_X.cpu().data.numpy()
#             labels = y
#             domains = z
#         else:
#             representations = np.concatenate((representations, f_X.cpu().data.numpy()), 0)
#             labels = np.concatenate((labels, y), 0)
#             domains = np.concatenate((domains, z), 0)

#     # last batch
#     x = d._train_X[(i+1)*batch_size:] 
#     y = d._train_y[(i+1)*batch_size:]
#     X = Variable(torch.cuda.FloatTensor(x))
#     if scDGN:
#         z = d._train_acc[(i+1)*batch_size:]
#         f_X = t.D(X, X, mode='eval')
#     else:
#         z = d._train_z[(i+1)*batch_size:]
#         f_X = t.D(X, mode='eval')
#     representations = np.concatenate((representations, f_X.cpu().data.numpy()), 0)
#     labels = np.concatenate((labels, y), 0)
#     domains = np.concatenate((domains, z), 0)
#     return representations, labels, domains

def extract_rep(net_glob, id, args):
    representations = None
    labels = None
    domains = None
    dataset = datasets.scDGN('data/{}/'.format(args.dataset), user_id=id, mode='test',
                                                 transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=args.bs)
    net_glob.eval()
    for idx, (features, labels) in enumerate(data_loader):
        X, y = features.to(args.device), labels.to(args.device)
        log_probs = net_glob(X)
        z = np.array([id for _ in range(y.size()[0])])
        if representations is None:
            representations = log_probs.cpu().data.numpy()
            labels = y.cpu().data.numpy()
            domains = z
        else:
            representations = np.concatenate((representations, log_probs.cpu().data.numpy()), 0)
            labels = np.concatenate((labels, y.cpu().data.numpy()), 0)
            domains = np.concatenate((domains, z), 0)

    return representations, labels, domains