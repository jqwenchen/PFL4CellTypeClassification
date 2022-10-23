import torch
a = torch.tensor([[1,2,3,4],[1,2,3,4]])
b = 2
print(a*b)
print()
import numpy as np
import os
root = '/home/jck/Desktop/PFL/data/pbmc/'
file_path = root + 'train/'
filename = os.listdir(file_path)[0]
print(filename)

print()
#idxs_users = np.arange(10)
idxs_users = np.random.choice(range(10), 3, replace=False)
print("idxs_users:", idxs_users)
for idx, user_id in enumerate(idxs_users):
    print("idx:", idx)
    print("user_id:", user_id)

print()
import json
with open(file_path + filename, 'r') as inf:
    cdata = json.load(inf)
    user_X =cdata['user_data']['f_0']['X'][0:2][1]
print("user_X:", user_X)