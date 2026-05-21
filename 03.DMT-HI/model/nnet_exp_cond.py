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

def gumbel_softmax_topN(logits, tau= 1, hard= False, eps= 1e-10, dim = -1, top_N=10):
    '''
        Find the top N figure to mask
    '''
    # if has_torch_function_unary(logits):
    #     return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k=top_N, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True, use_BN=True, use_DO=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(nn.Linear(in_dim, out_dim))
        if use_DO:
            m_l.append(nn.Dropout(p=0.02))
        if use_BN:
            m_l.append(nn.BatchNorm1d(out_dim))
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        if self.block[0].weight.shape[0] == self.block[0].weight.shape[1]:
            return self.block(x) + x
        else:
            return self.block(x)


class DMTEVT_model(LightningModule):
    def __init__(
        self,
        lr=0.005,
        sigma=0.05,
        sample_rate_feature=0.6,
        num_input_dim=64,
        num_train_data=60000,
        weight_decay=0.0001,
        top_k_1=3,
        top_k_2=1000,
        weight_mse=2,
        weight_nepo=1,
        tau=20,
        **kwargs,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        num_input_dim = self.hparams.num_input_dim
        self.enc = self.InitNetworkMLP(NS=[num_input_dim, 500]+[200]*10+[50])
        self.dec = self.InitNetworkMLP(NS=[50, 500, 500, num_input_dim], last_relu=False)
        self.vis = self.InitNetworkMLP(NS=[50, 500, 2], last_relu=False)
        self.exp = self.InitNetworkMLP(NS=[4, 500, 500, num_input_dim])
        
        self.vis_dictionary = torch.randn((num_train_data, 2)).to(self.device)

        # import pdb; pdb.set_trace()

    def InitNetworkMLP(self, NS, last_relu=True, use_DO=True, use_BN=True, use_RL=True):

        m_l = []
        for i in range(len(NS) - 1):
            if i == len(NS) - 2 and not last_relu:
                m_l.append(NN_FCBNRL_MM(NS[i], NS[i + 1], use_RL=False, use_DO=use_DO, use_BN=use_BN))
            else:
                m_l.append(NN_FCBNRL_MM(NS[i], NS[i + 1], use_RL=True, use_DO=use_DO, use_BN=use_BN))
            
        model_pat = nn.Sequential(*m_l)

        return model_pat

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        '''
        Calculates the pairwise Euclidean distance
        '''
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0]) == 1] = 1e-12
        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _Similarity(self, dist, sigma=0.3):
        '''
        Compute the similarity matrix based on distances
        sigma parameter controlling the width of the kernel
        '''
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        Pij = torch.exp(-dist / (2 * sigma ** 2))
        return Pij

    def LossManifold(
        self,
        input_data,
        latent_data,
        sigma=0.03,
        top_k_2=500,
        top_k_1=3,
        high_tok_rank=1,
    ):
        '''
            Manifold-Contrastive Loss
            
        '''
        data_1 = input_data[: input_data.shape[0] // 2]
        
        dis_P = self._DistanceSquared(data_1)
        latent_data_1 = latent_data[: input_data.shape[0] // 2]
        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity(
            dist=dis_Q_2,
            sigma=sigma,
        )
        
        _, indices = torch.topk(dis_P, high_tok_rank, dim=-1, largest=False)
        dis_P_topk_mask = torch.zeros_like(Q_2, dtype=torch.bool)
        dis_P_topk_mask.scatter_(dim=-1, index=indices, value=True)
        
        _, indices2 = torch.topk(dis_P, top_k_2, dim=-1, largest=False)
        dis_P_topk_mask2 = torch.zeros_like(Q_2, dtype=torch.bool)
        dis_P_topk_mask2.scatter_(dim=-1, index=indices2[:, top_k_1:], value=True)
        
        po = Q_2[dis_P_topk_mask.detach()]
        ne = Q_2[dis_P_topk_mask2.detach()]
        return ne.mean(), 1 - po.mean()

    def get_weight(self, cond):
        
        rand = torch.randn(cond.shape[0], 2).to(cond.device)
        input_data = torch.cat([cond, rand], dim=1)
        w = self.exp(input_data)
        weight = F.tanh(w)*10
        return weight

    def forward(self, x, tau=100.0, cond=None):

        num_select = int(x.shape[1]*self.hparams.sample_rate_feature)
        weight = self.get_weight(cond=cond)
        if tau is not None:
            self.mask = gumbel_softmax_topN(weight, tau=tau, hard=True, top_N=num_select)
            x = x * self.mask
        else:
            top_k_index = weight.topk(num_select, dim=1)[1]
            self.mask = torch.zeros_like(weight)
            self.mask = self.mask.scatter_(1, top_k_index, 1) > 0
            x = x * self.mask
                    
        lat_high_dim = self.enc(x)
        lat_vis = self.vis(lat_high_dim)
        return lat_high_dim, lat_vis

    def get_tau(self, epoch, total_epochs=900, tau_start=100, tau_end=1.001):
        """
        计算指定 epoch 的 tau 值。
        
        :param epoch: 当前的 epoch 数 (从0开始)
        :param total_epochs: 总的 epoch 数
        :param tau_start: 开始的 tau 值
        :param tau_end: 结束的 tau 值
        :return: 计算得到的 tau 值
        """
        if epoch >= total_epochs:
            return tau_end
        else:    
            return tau_start * (tau_end / tau_start) ** (epoch / (total_epochs-1))

    def training_step(self, batch, batch_idx):

        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        index = batch['index']

        cond = self.vis_dictionary[index.to(self.vis_dictionary.device)]
        # self.tau = self.get_tau(self.current_epoch)
        self.tau = self.hparams.tau
        data_input_item = torch.cat([data_input_item, data_input_aug])
        lat_high_dim, lat_vis = self(
            data_input_item, 
            tau=self.tau if self.tau>=1 else None,
            cond = torch.cat([cond.detach().to(self.device), cond.detach().to(self.device)])
            )
        
        # save vis to vis_dictionary
        # import pdb; pdb.set_trace()
        
        loss_topo_ne, loss_topo_po = self.LossManifold(
            input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            sigma=self.hparams.sigma,
            top_k_1=self.hparams.top_k_1,
            top_k_2=self.hparams.top_k_2
        )
        
        loss_topo = (loss_topo_ne*self.hparams.weight_nepo + loss_topo_po)/(1+self.hparams.weight_nepo) # + loss_crossentropy
        loss_mse = ((self.dec(lat_high_dim)-data_input_item)**2).mean()
        loss_all = loss_topo + loss_mse*self.hparams.weight_mse #+ kloss# + mlm_loss / 10

        self.log('loss_all', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('loss_mse', loss_mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_topo_ne', loss_topo_ne, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_topo_po', loss_topo_po, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('lr', float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_all

    def validation_step(self, batch, batch_idx, test=False):
        # data_input_item, data_input_aug, label, index, = batch
        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        index = batch['index']

        if self.vis_dictionary.device != self.device:
            self.vis_dictionary = self.vis_dictionary.to(self.device)
            # match the dtype of vis_dictionary

        # data_aug = self.transform(data_aug.permute(0,3,1,2)/255)
        cond = self.vis_dictionary[index.to(self.vis_dictionary.device)]
        lat_high_dim, lat_vis = self(
            data_input_item,
            cond = cond.detach().to(self.device)
            )
        data_recons = self.dec(lat_high_dim)
        
        if self.vis_dictionary.dtype != lat_vis.dtype:
            self.vis_dictionary = self.vis_dictionary.to(lat_vis.dtype)
        
        self.vis_dictionary[index] = lat_vis[:data_input_aug.shape[0]]
        
        self.validation_step_outputs_high = lat_high_dim
        self.validation_step_outputs_vis = lat_vis
        self.validation_step_outputs_recons = data_recons

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.hparams.weight_decay, lr=self.hparams.lr)
        # self.scheduler = StepLR(optimizer, step_size=self.hparams.max_epochs // 10, gamma=0.8)
        # return [optimizer], [self.scheduler]
        return optimizer

