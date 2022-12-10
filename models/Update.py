import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, param):
        net.load_state_dict(param)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)

        for iter in trange(self.args.local_ep):
            for features, labels in tqdm(self.ldr_train):
            # for batch_idx, (features, labels) in enumerate(self.ldr_train):
                # revise
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(features)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()


    def train_p(self, net, param):
        '''

        for layer in range(len(grads[1])):
            eff_grad = grads[1][layer] + self.lamda * (self.local_models[idx][layer] - self.global_model[layer])#vk= vk− η(∇Fk(vk) + λ(vk− wt))
            self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad
        '''
        w_glob = copy.deepcopy(net.state_dict()) # copy a w_glob (global model)before update
        net.load_state_dict(param)  # load clients' local model params
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)

        for iter in trange(self.args.local_ep):
            for features, labels in tqdm(self.ldr_train):
            # for batch_idx, (features, labels) in enumerate(self.ldr_train):
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                w_p = copy.deepcopy(net.state_dict()) # copy a local model as w_p, w_p is Vk before update
                net.zero_grad()
                log_probs = net(features) # get the predicted output value
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                """
                loss.backward() and optimizer.step() to compute the normal gradient -> update w_p value 
                """
                # get updated personalised model's grad
                grads = []
                for g_param in net.parameters(): # g_param: params in each layer
                    grads.append(g_param.grad) #append each layer's gradient  -> grads equals to deltaFk(Vk)
                w_glob_value = []
                for w_global_param in w_glob.values():
                    w_glob_value.append(w_global_param) # append w_glob values to w_glob_value list
                for layer, w_param in enumerate(w_p):
                    all_grad = grads[layer] + self.args.lamda * (w_p[w_param] - w_glob_value[layer]) #lambda*(Vk - W^t)
                    """
                    Vk: w_p value before update-> personalized model
                    W^t: w_glob value before update -> a global model
                    if lambda is smaller -> close to personalized model 
                    """
                    w_p[w_param] = w_p[w_param] - self.args.lr * all_grad # get the updated personalized model : Vk
                net.load_state_dict(w_p) # update w_p after a batch , and then load w_p for next iteration

        return net.state_dict()