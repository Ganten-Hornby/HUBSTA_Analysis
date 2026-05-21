from sklearn.datasets import load_digits

from torch.utils import data
import numpy as np
import os
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import joblib


class HD_WS_TESTDataset(data.Dataset):
    def __init__(
            self,
            data,
            label,
            index,
            k=10,
            pca_dim=100,
            uselabel=False,
            uniform_param=1,
            train_val="train",
            use_neighbor=True,
            augment_bool=True,
            num_positive_samples=1
    ):
        self.data = data
        self.label = label
        self.ori_index = index

        self.augment_bool = augment_bool

        self.train_val = train_val
        self.uniform_param = uniform_param

        if use_neighbor:
            self.neighbors_index = self.Preprocessing(
                k, pca_dim, uselabel)

    def Preprocessing(self, k, pca_dim, uselabel):
        self.graphwithpca = False

        neighbors_index = self.cal_near_index(
            k=k,
            uselabel=uselabel,
            pca_dim=pca_dim
        )
        return neighbors_index

    def cal_near_index(self, k=10, uselabel=False, pca_dim=100):
        X_rshaped = self.data.reshape((self.data.shape[0], -1))
        if pca_dim < X_rshaped.shape[1]:
            X_rshaped = PCA(n_components=pca_dim).fit_transform(X_rshaped)
        if not uselabel:
            index = NNDescent(X_rshaped, n_jobs=-1)
            neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
            neighbors_index = neighbors_index[:, 1:]
        else:
            dis = pairwise_distances(X_rshaped)
            M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
            dis[(M - M.T) != 0] = dis.max() + 1
            neighbors_index = dis.argsort(axis=1)[:, 1: k + 1]
        return neighbors_index

    def augment(self, data_input_item, index):

        # print('------', index)
        if self.augment_bool:
            neighbor_index_list = self.neighbors_index[index]
            selected_index = np.random.choice(neighbor_index_list)
            alpha = np.random.uniform(0, self.uniform_param)
            try:
                selected_data_input = self.data[selected_index]
                data_input_aug = data_input_item * alpha + selected_data_input * (1 - alpha)
            except:
                data_input_aug = data_input_item
        else:
            # print('no augment', 'self.augment_bool', self.augment_bool, )
            data_input_aug = data_input_item
        return data_input_item, data_input_aug

    def __getitem__(self, index):

        index = index % self.data.shape[0]

        # print('index', index)
        data_input_item = self.data[index]
        label = self.label[index]
        data_input_item, data_input_aug = self.augment(data_input_item, index)

        data_input_item = data_input_item.astype(np.float32)
        data_input_aug = data_input_aug.astype(np.float32)

        index_ori = self.ori_index[index]

        # print('data_input_item', data_input_item.shape, 'data_input_aug', data_input_aug.shape)
        return {
            "data_input_item": data_input_item,
            "data_input_aug": data_input_aug,
            "label": label,
            "index": index,
            'index_ori': index_ori
        }

    def __len__(self):
        return self.data.shape[0]