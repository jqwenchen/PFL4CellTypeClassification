import numpy as np
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
sys.path.append('..')
from utils.options import args_parser

args = args_parser()

# checkpoint = torch.load('/home/jck/Desktop/Ditto/checkpoint/ckpt.pth_ClassicNN_1_pbmc.pth')
checkpoint = torch.load('/home/jck/Desktop/Ditto/checkpoint/ckpt.pth_ClassicNN_50_pancreas_0_last_model.pth')
net_glob = checkpoint['net_glob']


for idx in range(args.num_users):
    dataset = datasets.scDGN('../dg_data/{}/'.format(args.dataset), user_id=idx, mode='test',
                             transform=transforms.ToTensor())

    net_glob.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.bs)
    for idx, (features, labels) in enumerate(data_loader):
        if args.gpu != -1:
            features, labels = features.cuda(),labels.cuda()
        log_probs = net_glob(features)

        test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()

        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))

