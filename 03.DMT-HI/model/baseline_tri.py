import numpy as np
from lightning import LightningModule
import os
import sys
from PIL import Image
import wandb
import platform
import torch.nn.functional as F
import plotly.graph_objects as go
import lightning as pl
import torch
from torch import nn
import torch.distributed as distributed
from lightning.pytorch.cli import LightningCLI
import torchvision.utils as vutils
from torchvision import models
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy

from lightning import Trainer

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import functools
from call_backs.MaskExp import MaskExpCallBack
from transformers import BertConfig, BertModel
# import pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR as LinearWarmupCosineAnnealingLR

from torch.optim.lr_scheduler import _LRScheduler
from model.resnet import ResNet, BasicBlock

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from datetime import datetime



# from fa import BertModel as fa_BertModel
# from fa import BertConfig as fa_BertConfig
import pandas as pd
import scipy
import uuid

from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dataloader import data_rankseq as data_base

# import lightning_lite
import manifolds

# from kornia.augmentation import Normalize, RandomGaussianBlur, RandomSolarize, RandomGrayscale, RandomResizedCrop, RandomCrop, ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
import math

new_uuid = str(uuid.uuid4())

import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
#import tensorflow as tf
#import umap
import scanpy as sc
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.neighbors import NearestNeighbors
#import pacmap
import numpy as np
#from umap.parametric_umap import ParametricUMAP
from sklearn.preprocessing import MinMaxScaler
#from ivis import Ivis
#import phate
from openTSNE import TSNE
#from hnne import HNNE
import time
np.random.seed(42)
seed=42
#import dmtev
class DMTEVT_model(LightningModule):
    def __init__(
        self,
        method='tsne',
        data_name='mnist',
        p1=1,
        p2=1,
        p3=0,
        n_components=2,
        perplexity=30,
        min_dist=0.1,
        n_neighbors=10,
        early_exaggeration=12,
        num_input_dim=64,
        k=130,
        ntrees=40,
        **kwargs,
    ):
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.min_dist=min_dist
        self.n_neighbors=n_neighbors
        self.k=k
        self.ntrees=ntrees
        self.method=method
        self.p1=p1
        self.p2=p2
        self.p3=p3
        if p3 == 1:
            self.seeds=self.p3*100+self.p1*10+self.p2
            np.random.seed(self.seeds)
            self.p1_index = pd.read_excel('/zangzelin/yuhao/gitlab/dmt_nml/model/p1.xlsx',index_col=0).T
            self.p2_index = pd.read_excel('/zangzelin/yuhao/gitlab/dmt_nml/model/p2.xlsx',index_col=0).T
            self.p1=int(self.p1_index[data_name][method])
            self.p2=int(self.p2_index[data_name][method])
        self.dataidx=int(0)
        self.save_path="save_emb_data/p"+str(p3)+"/"+method+"/"+data_name
        self.data_id=str(p1)+str(p2)
        #elif method == "phate" :
        #    self.tsne = phate.PHATE()
        #elif method == "evnet":
        #    self.tsne = phate.PHATE()
        self.embedding = None
        self.x = None
        self.visited = False
        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.validation_step_outputs_high = None
        self.validation_step_outputs_vis = None
        self.validation_step_lat_vis_exp = None 
        self.validation_origin_input = None
        num_input_dim = self.hparams.num_input_dim
        
        # self.enc = self.InitNetworkMLP(NS=[num_input_dim, 500]+[200]*10+[50])
        
        # self.dec = self.InitNetworkMLP(NS=[vis_dim, 500, 500, num_input_dim], last_relu=False)
        # import pdb; pdb.set_trace()
    def forward(self, x):
        self.tsne.fit(x.cpu().numpy())
    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self.visited = True
        self.dataidx=0
        data_id=self.data_id+"_2"
        index_save_path=self.save_path
        if not os.path.exists(index_save_path):
            os.makedirs(index_save_path, exist_ok=True)
        index_save_name_train = f'{data_id}.npy'
        if not os.path.exists(os.path.join(index_save_path, index_save_name_train)):
            data_input_item = batch['data_input_item']
            data_input_aug = batch['data_input_aug']
            if self.method == "_tsne" :
                from openTSNE import TSNE
                p1_list = [20, 25, 30, 35]
                p2_list = [8, 10, 12, 14, 16]
                if self.p3 == 1:
                    print("p1",self.p1)
                    print("p2",self.p2)
                    self.tsne = TSNE(perplexity=p1_list[self.p1], early_exaggeration=p2_list[self.p2], random_state=self.seeds)
                else :
                    self.tsne = TSNE(perplexity=p1_list[self.p1], early_exaggeration=p2_list[self.p2])
            elif self.method == "ivis" :
                import tensorflow as tf
                from ivis import Ivis
                p1_list = [130, 140, 150, 160]
                p2_list = [40, 45, 50, 55, 60]
                if self.p3 == 1:
                    np.random.seed(self.seeds)
                    tf.random.set_seed(self.seeds)
                    print("p1",self.p1)
                    print("p2",self.p2)
                    self.tsne = Ivis(embedding_dims=2, k=p1_list[self.p1], ntrees=p2_list[self.p2],model='maaten', n_epochs_without_progress=5)
                else :
                    self.tsne = Ivis(embedding_dims=2, k=p1_list[self.p1], ntrees=p2_list[self.p2],model='maaten', n_epochs_without_progress=5)
            elif self.method == "pumap" :
                import tensorflow as tf
                from umap.parametric_umap import ParametricUMAP
                p1_list = [10, 15, 20, 25]
                p2_list = [0.01, 0.05, 0.08, 0.1, 0.15]
                if self.p3 == 1:
                    print("p1",self.p1)
                    print("p2",self.p2)
                    self.tsne = ParametricUMAP(n_components=self.n_components,n_neighbors=p1_list[self.p1], min_dist=p2_list[self.p2], random_state=self.seeds)
                else:
                    # n_components=self.n_components
                    # n_neighbors=p1_list[self.p1]
                    # min_dist=p2_list[self.p2]
                    
                    self.tsne = ParametricUMAP(n_components=n_components,n_neighbors=p1_list[self.p1], min_dist=p2_list[self.p2])
            elif self.method == "_umap" :
                import tensorflow as tf
                import umap
                p1_list = [10, 15, 20, 25]
                p2_list = [0.01, 0.05, 0.08, 0.1, 0.15]
                if self.p3 == 1:
                    print("p1",self.p1)
                    print("p2",self.p2)
                    print("seed",self.seeds)
                    self.tsne = umap.UMAP(n_components=self.n_components,n_neighbors=p1_list[self.p1], min_dist=p2_list[self.p2], random_state=self.seeds)
                else:
                    # print('start umap, time', time.time())
                    n_components=self.n_components
                    n_neighbors=p1_list[self.p1]
                    min_dist=p2_list[self.p2]
                    print("n_components",n_components)
                    print("n_neighbors",n_neighbors)
                    print("min_dist",min_dist)

                    self.tsne = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist)
                    # print('end umap, time', time.time())

            elif self.method == 'pacmap':
                import tensorflow as tf
                import pacmap
                p1_list = [10, 15, 20, 25]
                p2_list = [0.3, 0.4, 0.5, 0.6, 0.7]
                if self.p3 == 1:
                    print("p1",self.p1)
                    print("p2",self.p2)
                    self.tsne = pacmap.PaCMAP(n_components=self.n_components,n_neighbors=p1_list[self.p1], MN_ratio=p2_list[self.p2],save_tree=True, random_state=self.seeds)
                else:
                    self.tsne = pacmap.PaCMAP(n_components=self.n_components,n_neighbors=p1_list[self.p1], MN_ratio=p2_list[self.p2],save_tree=True)
            elif self.method == 'evnet':
                from dmtev.dmtev_ import DMTEV
                p1_list = [1e-3,5e-3,1e-2, 1e-1]
                p2_list = [3, 5,8, 10, 15]
                if self.p3 == 1:
                    print("p1",self.p1)
                    print("p2",self.p2)
                    self.tsne = DMTEV(num_fea_aim=data_input_item.shape[0], device_id=0, epochs=1500,nu=p1_list[self.p1], K=p2_list[self.p2], seed=self.seeds)
                else:
                    self.tsne = DMTEV(num_fea_aim=data_input_item.shape[0], device_id=0, epochs=1500,nu=p1_list[self.p1], K=p2_list[self.p2])
            elif self.method == "hnne" :
                from hnne import HNNE
                p1_list = ['cosine', 'euclidean', 'manhattan', 'l1']
                p2_list = [0.25, 0.35, 0.45, 0.55, 0.65]
                if self.p3 == 1:
                    print("p1",self.p1)
                    print("p2",self.p2)
                    np.random.seed(self.seeds)
                    self.tsne = HNNE(dim=2, metric=p1_list[self.p1], radius=p2_list[self.p2])
                else: 
                    self.tsne = HNNE(dim=2, metric=p1_list[self.p1], radius=p2_list[self.p2])
            if self.method != "nml":
                data_input_item = data_input_item.reshape((data_input_item.shape[0],-1))
            index = batch['index'][:data_input_item.shape[0]]
            if self.method == "ivis" :
                data_input_item = MinMaxScaler().fit_transform(data_input_item.cpu().numpy())
            else:
                data_input_item = data_input_item.cpu().numpy()
            with torch.no_grad():
                if(self.method == "_tsne"):
                    self.embedding=self.tsne.fit(data_input_item)
                else :
                    print(data_input_item.shape)
                    print(data_input_item)
                    start_time = datetime.now()
                    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    self.tsne.fit(data_input_item)
                    end_time = datetime.now()
                    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return None
    def validation_step(self, batch, batch_idx, test=False, dataloader_idx=0):
        # data_input_item, data_input_aug, label, index, = batch
        idx=self.dataidx
        print("dataidx",self.dataidx)
        data_id=self.data_id+"_"+str(idx)
        index_save_path=self.save_path
        print("dataloader_idx",dataloader_idx)
        if dataloader_idx == 0:               
            if self.visited:
                index_save_name_train = f'{data_id}.npy'
                data_input_item = batch['data_input_item']
                if self.method != "nml":
                    data_input_item = data_input_item.reshape((data_input_item.shape[0],-1))
                data_input_aug = batch['data_input_aug']
                index = batch['index']
                self.validation_step_outputs_high = data_input_item
                if not os.path.exists(os.path.join(index_save_path, index_save_name_train)):
                    if self.method == "ivis" :
                        data_input_item = MinMaxScaler().fit_transform(data_input_item.cpu().numpy())
                    else:
                        data_input_item = data_input_item.cpu().numpy()
                    with torch.no_grad():
                        if self.method == "_tsne":
                            self.validation_step_outputs_vis = self.embedding.transform(data_input_item)
                        else:
                            self.validation_step_outputs_vis = self.tsne.transform(data_input_item)
                    
                    np.save(os.path.join(index_save_path, index_save_name_train), self.validation_step_outputs_vis)
                    print(self.validation_step_outputs_high)
                    print(self.validation_step_outputs_vis)
                else:
                    print('load emb data', index_save_name_train)
                    self.validation_step_outputs_vis = np.load(os.path.join(index_save_path, index_save_name_train))
                self.validation_step_outputs_vis = np.array(self.validation_step_outputs_vis)
                self.validation_step_outputs_high = self.validation_step_outputs_high.cpu().numpy()
                self.validation_origin_input = self.validation_step_outputs_high
                self.validation_step_lat_vis_exp = np.array(self.validation_step_outputs_vis)
                self.dataidx=int(self.dataidx)+1
                # self.validation_step_outputs_recons = data_recons