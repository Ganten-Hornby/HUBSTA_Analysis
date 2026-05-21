import lightning as pl
import numpy as np
import scanpy as sc
from data_model import dataset_all as dataset
from torch.utils.data import DataLoader


class MyDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path,
            use_rep:str = "X_pca",
            label = None,
            data_name: str = "Digits",
            batch_size: int = 32,
            num_workers: int = 1,
            K: int = 3,
            uselabel: bool = False,
            pca_dim: int = 50,
            n_cluster: int = 25,
            n_f_per_cluster: int = 3,
            l_token: int = 10,
            seed: int = 0,
            rrc_rate: float = 0.8,
            trans_range: int = 6,
            sample_len: int = 500,
            num_positive_samples=1,
            split_ratio=0.8,
            split_ratio_test=0.1,
            max_cell_train_val=640_000,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.use_rep = use_rep
        self.data, self.label, self.index_list = self.read_data(data_path, label)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.sample_len = sample_len
        self.num_positive_samples = num_positive_samples
        self.split_ratio = split_ratio
        self.split_ratio_test = split_ratio_test
        self.max_cell_train_val = max_cell_train_val

    def read_data(self, data_path, label):
        print(f"reading data: {data_path}")
        print("random pseudo-label only")
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        elif data_path.endswith('.h5ad'):
            data = sc.read_h5ad(data_path)
            data = data.obsm[self.use_rep].copy()
        else:
            raise ValueError(f"Unsupported file type: {data_path}. Supported types are .npy and .h5ad.")
        data = data.astype(np.float32)
        label = np.random.randint(0, 3, data.shape[0], dtype=int)  # pseudo-label
        index_list = np.arange(data.shape[0])
        return data, label, index_list

    def setup(self, stage: str):
        (
            train_data, train_label, train_index,
            val_data, val_label, val_index,
            test_data, test_label, test_index
        ) = self.split_data(
            split_ratio=self.split_ratio,
            split_ratio_test=self.split_ratio_test,
            max_cell_train_val=self.max_cell_train_val
        )
        print('train_data.shape', train_data.shape,
              'train_label.shape', train_label.shape,
              'val_data.shape', val_data.shape,
              'val_label.shape', val_label.shape,
              'test_data.shape', test_data.shape,
              'prediction_label.shape', self.label.shape,
              'prediction_data.shape', self.data.shape
              )
        dataset_meta = getattr(dataset, self.data_name + "Dataset")
        self.data_train = dataset_meta(
            data=train_data,
            label=train_label,
            index=train_index,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
        )
        self.data_val = dataset_meta(
            data=val_data,
            label=val_label,
            index=val_index,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
            train_val="val",
        )
        self.data_test = dataset_meta(
            data=test_data,
            label=test_label,
            index=test_index,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
            train_val="test",
        )
        self.data_pred = dataset_meta(
            data=self.data.copy(),
            label=self.label.copy(),
            index=self.index_list.copy(),
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
            train_val="pred",
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        val1 = DataLoader(
            self.data_train,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val2 = DataLoader(
            self.data_val,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_val.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val3 = DataLoader(
            self.data_test,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_test.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return [val1, val2, val3]

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_test.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_pred,
            drop_last=False,
            batch_size=min(self.batch_size, self.data_pred.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def split_data(self, split_ratio=0.8, split_ratio_test=0.1, max_cell_train_val=640000):
        data_len = self.data.shape[0]
        max_cell_train_val = min(max_cell_train_val, data_len)
        train_len = int(max_cell_train_val * split_ratio)
        test_len = int(max_cell_train_val * split_ratio_test)

        rand_index = np.random.permutation(data_len)
        train_index = rand_index[:train_len]
        val_index = rand_index[train_len: train_len + test_len]
        test_index = rand_index[train_len + test_len:max_cell_train_val]

        train_data = self.data[train_index].copy()
        train_label = self.label[train_index].copy()
        val_data = self.data[val_index].copy()
        val_label = self.label[val_index].copy()
        test_data = self.data[test_index].copy()
        test_label = self.label[test_index].copy()

        train_index = self.index_list[train_index]
        valid_index = self.index_list[val_index]
        test_index = self.index_list[test_index]

        return (
            train_data, train_label, train_index,
            val_data, val_label, valid_index,
            test_data, test_label, test_index
        )
