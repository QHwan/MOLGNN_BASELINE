from __future__ import print_function, division

import numpy as np
import time
import random
from collections import Counter

import torch
from torch_geometric.data import Data, DataLoader

class MoleculeDataset():
    def __init__(self, args):
        data_file = args.data_file
        data_dict = np.load(data_file, allow_pickle=True)

        self.args = args

        self.data_list = []
        x_list = data_dict['x']
        y_list = data_dict['y']
        edge_index_list = data_dict['edge_index']
        for x, y, edge_index in zip(x_list, y_list, edge_index_list):
            x = torch.from_numpy(np.array(x)).float()
            y = torch.from_numpy(np.array(y)).float()
            edge_index = torch.from_numpy(np.array(edge_index)).long().t().contiguous()
            self.data_list.append(Data(x=x, y=y, edge_index=edge_index))

        self.split_list = data_dict['split']

        self.args.num_node_features = self.data_list[0].num_node_features

        self.preprocess()

    def load_data(self, partition):
        if partition == 'train':
            idx_list = self.train_idx_list
        elif partition == 'val':
            idx_list = self.val_idx_list
        elif partition == 'test':
            idx_list = self.test_idx_list

        data_list = []
        for idx in idx_list:
            data_list.append(self.data_list[idx])

        return data_list

    def preprocess(self):
        n_split = len(self.split_list)
        self.train_idx_list = []
        self.val_idx_list = []
        self.test_idx_list = []

        fold_idx_list = list(range(n_split))
        test_fold_idx = self.args.fold_idx
        val_fold_idx = (test_fold_idx + 1) % n_split

        fold_idx_list.remove(test_fold_idx)
        fold_idx_list.remove(val_fold_idx)
        train_fold_idx = fold_idx_list

        self.test_idx_list.extend(list(self.split_list[test_fold_idx]))
        self.val_idx_list.extend(list(self.split_list[val_fold_idx]))
        for i in train_fold_idx:
            self.train_idx_list.extend(list(self.split_list[i]))

        assert len(set(self.train_idx_list) & set(self.val_idx_list)) == 0
        assert len(set(self.val_idx_list) & set(self.test_idx_list)) == 0
        assert len(set(self.train_idx_list) & set(self.test_idx_list)) == 0
        
