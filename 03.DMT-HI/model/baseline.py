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




# from fa import BertModel as fa_BertModel
# from fa import BertConfig as fa_BertConfig

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import umap
import scanpy as sc
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.neighbors import NearestNeighbors
import pacmap
import numpy as np
from umap.parametric_umap import ParametricUMAP
from sklearn.preprocessing import MinMaxScaler
#from ivis import Ivis
import time
np.random.seed(42)
seed=42

class DMTEVT_model(LightningModule):
    def __init__(
        self,
        method='tsne',
        n_components=2,
        perplexity=30,
        min_dist=0.1,
        n_neighbors=10,
        early_exaggeration=12,
        num_input_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.min_dist=min_dist
        self.n_neighbors=n_neighbors
        if method == 'tsne':
            self.tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, early_exaggeration=self.early_exaggeration)
        elif method == 'umap':
            print(self.n_neighbors)
            print(self.min_dist)
            self.tsne = umap.UMAP(n_components=self.n_components,n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        elif method == 'pacmap':
            self.tsne = pacmap.PaCMAP(n_components=self.n_components,n_neighbors=self.n_neighbors, MN_ratio=self.min_dist)
        self.embedding = None
        self.x = None
        self.visited = False
        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.validation_step_outputs_high = None
        self.validation_step_outputs_vis = None
        num_input_dim = self.hparams.num_input_dim
        
        # self.enc = self.InitNetworkMLP(NS=[num_input_dim, 500]+[200]*10+[50])
        
        # self.dec = self.InitNetworkMLP(NS=[vis_dim, 500, 500, num_input_dim], last_relu=False)
        # import pdb; pdb.set_trace()
    def forward(self, x):
        self.embedding = torch.tensor(self.tsne.fit_transform(x.cpu().numpy())).reshape(-1,2)
        return self.embedding

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self.visited = True
        return None
    def validation_step(self, batch, batch_idx, test=False, dataloader_idx=0):
        # data_input_item, data_input_aug, label, index, = batch
        if dataloader_idx == 0:
            data_input_item = batch['data_input_item']
            data_input_aug = batch['data_input_aug']
            index = batch['index']
            if self.visited:
                self.validation_step_outputs_high = torch.tensor(data_input_item)
                with torch.no_grad():
                    self.validation_step_outputs_vis = self(data_input_item)
                print(self.validation_step_outputs_high)
                print(self.validation_step_outputs_vis)
                # self.validation_step_outputs_recons = data_recons