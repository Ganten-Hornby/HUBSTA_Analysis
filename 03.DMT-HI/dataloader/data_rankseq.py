from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
# from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
from PIL import Image
import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import sklearn
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


from dataloader.data_sourse import DigitsDataset, DigitsRankSeqDataset
from dataloader.data_graph import *
from dataloader.data_insemb import *
from dataloader.data_XU import *
from dataloader.data_face import *


class MnistDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label


class ActivityDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        train_data = pd.read_csv(data_path + "/feature_select/Activity_train.csv")
        test_data = pd.read_csv(data_path + "/feature_select/Activity_test.csv")
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(["subject", "Activity"], axis=1).to_numpy()
        label_str = all_data["Activity"].tolist()
        label_str_set = list(set(label_str))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data - data.min()) / (data.max() - data.min())

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class ProtTPDSCDataset(DigitsRankSeqDataset):
    def load_TPD(self, data_path, train=True):
        list_datapath = [
            data_path + "/Protein_data/TPD_alldata_Histopathology_type_list0704.csv",
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_dis.csv",
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_pro.csv",
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_ret.csv",
        ]

        label_pd = pd.read_csv(
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_lab.csv"
        ).set_index("MS_file_name")

        label_list = []
        for i, path in enumerate(list_datapath):
            if i == 0:
                label_list = pd.read_csv(path, low_memory=False)["Histopathology_type"].tolist()
                data_meta = pd.read_csv(path, low_memory=False).drop(
                    [
                        "Unnamed: 0",
                        "label",
                        "Sets",
                        "Bethesda_Score",
                        "Histopathology_type",
                    ],
                    axis=1,
                )
            else:
                # import pdb; pdb.set_trace()
                file_name = pd.read_csv(path)["MS_file_name"].tolist()
                label_list += label_pd.loc[file_name]["Classificaiton_type"].to_list()
                data_meta_add = pd.read_csv(path).drop(["MS_file_name"], axis=1)
                data_meta = pd.concat([data_meta, data_meta_add], axis=0)

        # data_meta = data_meta.fillna(data_meta.min().min() - 1)
        data_meta.columns = data_meta.columns.str.replace(" ", "")
        data_meta.columns = data_meta.columns.str.replace(";", "")

        # import pdb; pdb.set_trace()
        # transfoer str label to number
        for i, str_i in enumerate(label_list):
            if str_i == "N" or str_i == "M" or str_i == "L" or str_i == "A":
                label_list[i] = "M"
            else:
                label_list[i] = "B"
        label_list_set = list(set(label_list))
        label_list = [label_list_set.index(i) for i in label_list]
        label_list = np.array(label_list)

        # data = np.array(data_meta.to_numpy()).astype(np.float32)

        # na_rate = (data<11).sum(axis=0) / data.shape[0]
        # selectmask = na_rate<0.05
        # data = data[:, selectmask]
        # print('data.shape', data.shape)
        # data = (data-10)/(data.max()-data.min())
        label = np.array(label_list)

        return data_meta, label

    def load_data_cai(self, data_path, train=True):
        label_select = "MetS_F2"

        datapath = data_path + "Protein_data/"
        # data_meta = np.asarray(scipy.io.mmread('data/matrix.mtx').todense())
        label_t1 = pd.read_csv(datapath + "Protein_guo_cai_label.csv", low_memory=False).set_index(
            "Patient_ID"
        )
        label_t1 = label_t1.fillna(-1)
        sam_t1 = pd.read_csv(datapath + "Protein_guo_cai_sam.csv", low_memory=False).set_index(
            "Sample_ID"
        )
        data_t1 = pd.read_csv(datapath + "Protein_guo_cai.csv")
        data_t1 = data_t1.set_index("Sample_ID")
        data_t1 = data_t1.drop(["NL2991baseline", "NL2991F2"])
        data_sam_map_list_t1 = data_t1.index.to_list()
        sam_t1 = sam_t1[sam_t1["Biological replicate"] == 0]
        sample_name_t1 = sam_t1.loc[data_sam_map_list_t1]["Patient_ID"]

        sam_t2 = pd.read_csv(datapath + "Protein_guo_cai_test_sam.csv").set_index(
            "Sample_ID"
        )
        # import pdb; pdb.set_trace()
        sam_t2 = sam_t2[sam_t2["Biological_replicate"] == 0]
        data_t2 = pd.read_csv(datapath + "Protein_guo_cai_test.csv").set_index(
            "Sample_ID"
        )
        data_sam_map_list_t2 = data_t2.index.to_list()
        for i, s in enumerate(data_sam_map_list_t2):
            while "NL0" in data_sam_map_list_t2[i]:
                data_sam_map_list_t2[i] = data_sam_map_list_t2[i].replace("NL0", "NL")
        sample_name_t2 = sam_t2.loc[data_sam_map_list_t2]["Patient_ID"]

        min_value = data_t1.min().min()
        # data_t1 = data_t1.fillna(min_value-1)
        # data_t2 = data_t2.fillna(min_value-1)
        data_all = pd.concat([data_t1, data_t2])
        data_all.columns = data_all.columns.str.split("_").str[0]
        # data_all = data_all.fillna(min_value - 1)

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # all_scaler = scaler.fit_transform(all)

        data_t1 = data_all[: data_t1.shape[0]]
        # data_t2 = all_scaler[data_t1.shape[0]:]
        # # import pdb; pdb.set_trace()
        label = label_t1.loc[sample_name_t1.to_list()][label_select]
        # if train :
        #     label_meta = np.array(label_t1.loc[sample_name_t1.to_list()][label_select])
        #     self.data = tensor(data_t1).float()
        #     self.label = tensor(label_meta).reshape((-1))
        # else:
        #     label_meta = np.array(label_t1.loc[sample_name_t2.to_list()][label_select])
        #     self.data = tensor(data_t2).float()
        #     self.label = tensor(label_meta).reshape((-1))
        return data_t1, label

    def load_data(self, data_path, train=True):
        data, label = self.load_TPD(data_path, train=True)
        data_cai, label_cai = self.load_data_cai(data_path, train=True)
        data_all_allfeature = pd.concat([data, data_cai], axis=0)

        nan_rate = (data_all_allfeature.isna()).sum(axis=0) / data_all_allfeature.shape[0]
        mask = nan_rate < 0.56
        data_all = data_all_allfeature.loc[:, mask]
        
        # import pdb; pdb.set_trace()

        min_value = data_all.min().min()
        data_all = data_all.fillna(min_value - 1)

        label = np.array(label).astype(np.int32)
        label_cai = (1+np.max(label) + np.array(label_cai)).astype(np.int32)

        label_all = np.concatenate([label, label_cai])

        data_all_numpy = np.array(data_all).astype(np.float32)
        # cross_feature = list(set(data.columns) & set(data_cai.columns))

        return data_all_numpy, label_all


class ProtTPDDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        list_datapath = [
            data_path + "/Protein_data/TPD_alldata_Histopathology_type_list0704.csv",
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_dis.csv",
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_pro.csv",
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_ret.csv",
        ]

        label_pd = pd.read_csv(
            data_path + "/Protein_data/TPD_diann_protMatrix_20210701_lab.csv"
        ).set_index("MS_file_name")

        label_list = []
        for i, path in enumerate(list_datapath):
            # print('loading', path, '...')
            if i == 0:
                label_list = pd.read_csv(path, low_memory=False)["Histopathology_type"].tolist()
                data_meta = pd.read_csv(path, low_memory=False).drop(
                    [
                        "Unnamed: 0",
                        "label",
                        "Sets",
                        "Bethesda_Score",
                        "Histopathology_type",
                    ],
                    axis=1,
                )
                # print('data_meta.shape', data_meta.shape)
            else:
                # import pdb; pdb.set_trace()
                file_name = pd.read_csv(path, low_memory=False)["MS_file_name"].tolist()
                label_list += label_pd.loc[file_name]["Classificaiton_type"].to_list()
                data_meta_add = pd.read_csv(path, low_memory=False).drop(["MS_file_name"], axis=1)
                data_meta = pd.concat([data_meta, data_meta_add], axis=0)

        data_meta.fillna(data_meta.min().min() - 1, inplace=True)
        data_meta.columns = data_meta.columns.str.replace(" ", "")
        data_meta.columns = data_meta.columns.str.replace(";", "")

        # import pdb; pdb.set_trace()
        # transfoer str label to number
        for i, str_i in enumerate(label_list):
            if str_i == "N" or str_i == "M" or str_i == "L" or str_i == "A":
                label_list[i] = "M"
            else:
                label_list[i] = "B"
        label_list_set = list(set(label_list))
        label_list = [label_list_set.index(i) for i in label_list]
        label_list = np.array(label_list)

        data = np.array(data_meta.to_numpy()).astype(np.float32)

        na_rate = (data < 11).sum(axis=0) / data.shape[0]
        selectmask = na_rate < 0.05
        data = data[:, selectmask]
        print("data.shape", data.shape)

        # pca to 500
        # pca = PCA(n_components=500)
        # data = pca.fit_transform(data)

        # calculate the variance of each feature
        # data_var = data_meta.var(axis=0)
        # data_var_theath = data_var.sort_values(ascending=False)[1000]
        # mask = data_var > data_var_theath
        # data = data[:, mask]
        # import pdb; pdb.set_trace()

        data = (data - 10) / (data.max() - data.min())

        label = np.array(label_list)
        return data, label


class Must_1_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/V1_Adult_Mouse_Brain/emb.npy")
        label = np.load(data_path + "/V1_Adult_Mouse_Brain/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_200115_08_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/200115_08/emb.npy")
        label = np.load(data_path + "/200115_08/pred_leiden.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_200127_15_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/200127_15/emb.npy")
        label = np.load(data_path + "/200127_15/pred_leiden.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_ME95_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/ME95/emb.npy")
        label = np.load(data_path + "/ME95/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_ME145_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/ME145/emb.npy")
        label = np.load(data_path + "/ME145/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_last_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/V1_Mouse_Brain_Sagittal_Posterior/emb.npy")
        label = np.load(data_path + "/V1_Mouse_Brain_Sagittal_Posterior/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_Ant_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/V1_Mouse_Brain_Sagittal_Anterior/emb_2d.npy")
        label = np.load(data_path + "/V1_Mouse_Brain_Sagittal_Anterior/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_Adult_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/V1_Adult_Mouse_Brain/emb.npy")
        label = np.load(data_path + "/V1_Adult_Mouse_Brain/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_MOB_Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/MOB/emb.npy")
        label = np.load(data_path + "/MOB/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class SAMUSIKDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        train_data = pd.read_csv(data_path + "/feature_select/Activity_train.csv")
        test_data = pd.read_csv(data_path + "/feature_select/Activity_test.csv")
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(["subject", "Activity"], axis=1).to_numpy()
        label_str = all_data["Activity"].tolist()
        label_str_set = list(set(label_str))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data - data.min()) / (data.max() - data.min())

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class KMnistDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        D = datasets.KMNIST(root=data_path, train=True, download=True, transform=None)

        data = (D.data[:60000] / 255).float().reshape((60000, -1))
        label = (D.targets[:60000]).reshape((-1))
        return data, label


class EMnistDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        D = datasets.EMNIST(
            root=data_path, train=True, split="byclass", download=True, transform=None
        )

        data = (D.data[:280000] / 255).float().reshape((280000, -1))
        label = (D.targets[:280000]).reshape((-1))
        return data, label


class Coil20Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        path = data_path + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255
        data = data.astype(np.float32)
        label = np.array(label)
        return data, label


class Coil100Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        path = data_path + "/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 128, 128, 3))
        for i, path_i in enumerate(fig_path):
            # print(i)
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I.resize((128, 128)))
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = (data.astype(np.float32) / 255).reshape(data.shape[0], -1)
        label = np.array(label)
        return data, label


class Cifar10Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        D = datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)

        data = np.array(D.data).astype(np.uint8)
        label = np.array(D.targets).reshape((-1))
        return data, label


class Cifar100Dataset(DigitsRankSeqDataset):
    def cifar100labeltrans(self, label):
        coarse_labels = np.array(
            [
                4,
                1,
                14,
                8,
                0,
                6,
                7,
                7,
                18,
                3,
                3,
                14,
                9,
                18,
                7,
                11,
                3,
                9,
                7,
                11,
                6,
                11,
                5,
                10,
                7,
                6,
                13,
                15,
                3,
                15,
                0,
                11,
                1,
                10,
                12,
                14,
                16,
                9,
                11,
                5,
                5,
                19,
                8,
                8,
                15,
                13,
                14,
                17,
                18,
                10,
                16,
                4,
                17,
                4,
                2,
                0,
                17,
                4,
                18,
                17,
                10,
                3,
                2,
                12,
                12,
                16,
                12,
                1,
                9,
                19,
                2,
                10,
                0,
                1,
                16,
                12,
                9,
                13,
                15,
                13,
                16,
                19,
                2,
                4,
                6,
                19,
                5,
                5,
                8,
                19,
                18,
                1,
                2,
                15,
                6,
                0,
                17,
                8,
                14,
                13,
            ]
        )
        return coarse_labels[label]

    def load_data(self, data_path, train=True):
        D = datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        data = np.array(D.data).astype(np.uint8)
        label = np.array(D.targets).reshape((-1))
        label = self.cifar100labeltrans(label)
        return data, label


class HCL60KDataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape) + 1).astype(np.bool_)
        for l in range(label.max() + 1):
            num_l = (label == l).sum()
            if num_l < 500:
                mask_label[label == l] = False

        data = data[mask_label]
        label = label[mask_label]

        return data, label


class MCAD9119Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/mca_data/mca_data_dim_34947.npy")
        data = (data[:, data.max(axis=0) > 4]).astype(np.float32)
        label = np.load(data_path + "/mca_data/mca_label_dim_34947.npy")

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        # if the number of each category is less than 100, delete this category
        for i in label_count:
            if label_count[i] < 500:
                label[label == i] = -1
        # delete the data with label -1
        data = data[label != -1]
        label = label[label != -1]

        return data, label.astype(np.int32)


class Gast10k1457Dataset(DigitsRankSeqDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)

        label_train_str = list(sadata.obs["celltype"])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str]
        )
        label = np.array(label_train).astype(np.int32)

        return data, label
