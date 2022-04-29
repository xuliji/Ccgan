import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, shapes_dir, target_dir, names_txt, alpha=0.0):
        self.target_dir = target_dir
        self.shapes_dir = shapes_dir
        self.alpha = float(alpha)
        with open(names_txt, 'r') as f:
            self.names_list = f.read().splitlines()
        temp_list = []
        for name in self.names_list:
            targets = np.load(os.path.join(self.target_dir, name[:-4] + '.npy'), allow_pickle=True)
            if len(np.argwhere(targets[:, 0] == self.alpha)) == 0:
                temp_list.append(name)
        for name in temp_list:
            self.names_list.remove(name)

    def __getitem__(self, index):
        # file path
        shape_path = os.path.join(self.shapes_dir, self.names_list[index])
        target_path = os.path.join(self.target_dir, self.names_list[index][:-4] + '.npy')

        shape = np.array(pd.read_csv(shape_path, sep=' ', header=None), dtype=np.float32)

        data = np.load(target_path, allow_pickle=True)

        index = np.argwhere(data[:, 0] == self.alpha)
        if len(index) != 0:
            target = data[index.item(), :]
            # sample = {'shape': torch.tensor(shape, dtype=torch.float32), 'target': torch.tensor(target, dtype=torch.float32)}

            return torch.tensor(target, dtype=torch.float32), torch.tensor(shape, dtype=torch.float32)

    def __len__(self):
        return len(self.names_list)

if __name__ == '__main__':
    data = MyDataset(shapes_dir='./picked_uiuc', target_dir='./result/07', names_txt='./picked_uiuc_list.txt',
                     alpha=0)

    train_loader = DataLoader(data, batch_size=10, shuffle=False)
    sum = 0
    for i, data in enumerate(train_loader):
        print(data[0].shape, data[1].shape)
        sum += 1
    print(sum)
