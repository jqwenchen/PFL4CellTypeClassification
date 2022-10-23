import json
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image

class scDGN(Dataset):

    def __init__(self, root, user_id, mode, transform = None):
        super(scDGN, self).__init__()
        self.mode = mode
        self.transform = transform
        self.user_id = user_id
        if self.mode == 'train' or self.mode == 'val':
            self.frac_train_val = 0.1
            self.file_path = root + 'train/'
            self.filename = os.listdir(self.file_path)[0]

        else:
            self.file_path = root + 'test/'
            self.filename = os.listdir(self.file_path)[0]

        with open(self.file_path + self.filename, 'r') as inf:
            cdata = json.load(inf)
    
        if self.mode == 'train':
            self.num_samples = int(cdata['num_samples'][self.user_id] * (1 - self.frac_train_val))
        elif self.mode == 'val':
            self.num_samples = cdata['num_samples'][self.user_id] - int(cdata['num_samples'][self.user_id] * (1 - self.frac_train_val))
        elif self.mode == 'test':
            self.num_samples = cdata['num_samples'][self.user_id]
            
    def __getitem__(self, index):
        if self.mode == 'train':
            with open(self.file_path + self.filename, 'r') as inf:
                cdata = json.load(inf)
            user_name = cdata['users'][self.user_id]
            user_X = cdata['user_data'][user_name]['X'][0:self.num_samples][index]
            user_y = cdata['user_data'][user_name]['y'][0:self.num_samples][index]
            
        elif self.mode == 'val':
            with open(self.file_path + self.filename, 'r') as inf:
                cdata = json.load(inf)
            user_name = cdata['users'][self.user_id]
            user_X = cdata['user_data'][user_name]['X'][-self.num_samples:][index]
            user_y = cdata['user_data'][user_name]['y'][-self.num_samples:][index]
        
        else:
            with open(self.file_path + self.filename, 'r') as inf:
                cdata = json.load(inf)
            user_name = cdata['users'][self.user_id]
            user_X = cdata['user_data'][user_name]['X'][index]
            user_y = cdata['user_data'][user_name]['y'][index]
        
        user_X = np.array(user_X).astype('float32')
        return user_X, user_y

    def __len__(self):
        return self.num_samples