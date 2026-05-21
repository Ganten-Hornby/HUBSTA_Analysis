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

def muti_gumbel(logits, tau= 1, hard= False, eps= 1e-10, dim = -1, top_N=10, num_muti_mask=10):
    
    mask_list = []
    for i in range(num_muti_mask):
        mask = gumbel_softmax_topN(logits[:,i,:], tau=tau, hard=hard, eps=eps, dim=dim, top_N=top_N)
        mask_list.append(mask)
    return torch.stack(mask_list, dim=1)

def gumbel_softmax_topN(logits, tau= 1, hard= False, eps= 1e-10, dim = -1, top_N=10):

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

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel(config)
        # 为了直接访问BertEmbeddings和BertEncoder，我们在这里不直接使用self.bert
        self.embeddings = self.bert.embeddings
        self.encoder = self.bert.encoder
        # self.pooler = self.bert.pooler
        self.fc = nn.Sequential(
            NN_FCBNRL_MM(config.hidden_size, 1000),
            NN_FCBNRL_MM(1000, 1000),
            NN_FCBNRL_MM(1000, config.hidden_size),
        )
        self.pooler = self.bert.pooler

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, mask=None):
        # Step 1: 使用Bert的嵌入层
        # if inputs_embeds is None:
        #     inputs_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        inputs_embeds = input_ids
        # if mask is not None:
        # import pdb; pdb.set_trace()
        # inputs_embeds = inputs_embeds * mask.reshape(-1, 1, inputs_embeds.shape[-1])

        # inputs_embeds_fc = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        # fc_out = self.fc(inputs_embeds_fc).reshape(inputs_embeds.shape[0], inputs_embeds.shape[1], -1)

        encoder_outputs = self.encoder(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        # 取编码器的输出
        sequence_output = encoder_outputs['last_hidden_state'] # + fc_out
        # sequence_output = fc_out
        
        pooled_output = self.pooler(sequence_output)
        
        # 返回编码器的输出和pooled输出，根据需要返回其他输出
        return pooled_output


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers=2, 
        num_attention_heads=6, 
        hidden_size=240, 
        intermediate_size=300, 
        max_position_embeddings=784, 
        num_input_dim=784,
        out_dim=50,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_muti_mask=10,
        ):
        super(TransformerEncoder, self).__init__()
        self.num_muti_mask = num_muti_mask
        config = BertConfig(
            vocab_size=num_input_dim+2,  # 词汇表大小
            # embedding_size=trans_embedding_size,  # 嵌入层大小
            hidden_size=hidden_size,  # 隐藏层大小
            num_hidden_layers=num_layers,  # 隐藏层的数量
            num_attention_heads=num_attention_heads,  # 注意力头的数量
            intermediate_size=intermediate_size,  # 前馈网络的大小
            max_position_embeddings=max_position_embeddings,  # 最大序列长度
            hidden_dropout_prob=hidden_dropout_prob,  # 隐藏层的Dropout概率
            attention_probs_dropout_prob=attention_probs_dropout_prob,  # 注意力层的Dropout概率
            # num_hidden_layers=num_layers_Transformer,
        )
        self.enc = CustomBertModel(config)
        self.fc = nn.Sequential(
            NN_FCBNRL_MM(hidden_size, hidden_size),
            NN_FCBNRL_MM(hidden_size, hidden_size),
            NN_FCBNRL_MM(hidden_size, out_dim, use_RL=False),
        )
        
    def forward(self, x, mask=None):
        x = x.reshape(x.shape[0], 1, x.shape[1]).expand(
            x.shape[0], self.num_muti_mask, x.shape[1])
        x = x * mask
        emb = self.enc(x)
        return self.fc(emb)


class DMTEVT_model(LightningModule):
    def __init__(
        self,
        lr=0.005,
        sigma=0.05,
        sample_rate_feature=0.6,
        num_input_dim=64,
        num_train_data=60000,
        weight_decay=0.0001,
        w_nb=1,
        w_fp=1,
        weight_mse=2,
        weight_nepo=1,
        tau=20,
        T_num_layers=2,
        T_num_attention_heads=6,
        T_hidden_size=240,
        T_intermediate_size=300,
        T_hidden_dropout_prob=0.1,
        T_attention_probs_dropout_prob=0.1,
        ckpt_path=None,
        num_muti_mask=10,
        out_dim=500,
        **kwargs,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        num_input_dim = self.hparams.num_input_dim
        
        # self.enc = self.InitNetworkMLP(NS=[num_input_dim, 500]+[200]*10+[50])
        self.enc = TransformerEncoder(
            num_layers=T_num_layers, 
            num_attention_heads=T_num_attention_heads, 
            hidden_size=T_hidden_size,
            intermediate_size=T_intermediate_size,
            max_position_embeddings=20,
            num_input_dim=num_input_dim,
            hidden_dropout_prob=T_hidden_dropout_prob,
            attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            out_dim=out_dim,
            num_muti_mask=num_muti_mask,
        )
        self.dec = self.InitNetworkMLP(NS=[out_dim, 500, 500, num_input_dim], last_relu=False)
        self.vis = self.InitNetworkMLP(NS=[out_dim, 500, 2], last_relu=False)
        self.exp = self.InitNetworkMLP(NS=[4, 500, 500, num_input_dim*num_muti_mask])
        
        self.vis_dictionary = torch.randn((num_train_data, 2)).to(self.device)

        if ckpt_path is not None:
            print('load ckpt from:', ckpt_path)
        #     # if not os.path.exists(ckpt_path+'/all.ckpt'):
        #     #     convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, ckpt_path+'/all.ckpt')
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])

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
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        Pij = torch.exp(-dist / (2 * sigma ** 2))
        return Pij

    def LossManifold(
        self,
        input_data,
        latent_data,
        sigma=0.03,
        w_nb=500,
        w_fp=3,
        high_tok_rank=1,
    ):

        # data_1 = input_data[: input_data.shape[0] // 2]
        
        # dis_P = self._DistanceSquared(data_1)
        latent_data_1 = latent_data[: input_data.shape[0] // 2]
        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        # dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2) + 1 
        NB = torch.pairwise_distance(latent_data_1, latent_data_2, p=2, keepdim=False) + 1
        loss_NB = (NB/(10+NB)).mean()
        
        n_select_each_sample = 4
        loss_fp_list = []
        for i in range(n_select_each_sample):
            latent_data_2_rand = latent_data_2[torch.randperm(latent_data_2.shape[0])]
            FP = torch.pairwise_distance(latent_data_1, latent_data_2_rand, p=2, keepdim=False) + 1
            loss_fp_list.append((1/(1+FP)).mean())

        loss_fp = torch.stack(loss_fp_list).mean()

        return loss_fp*w_fp, loss_NB*w_nb

    def get_weight(self, cond):
        
        rand = torch.randn(cond.shape[0], 2).to(cond.device)
        input_data = torch.cat([cond, rand], dim=1)
        w = self.exp(input_data).reshape((cond.shape[0], self.hparams.num_muti_mask, -1))
        weight = F.tanh(w)*10
        return weight


    def batch_orthogonal_loss(self, batch_u, batch_v):
        # 计算批量向量的点积，假设batch_u和batch_v的形状为[N, D]，其中N是批量大小，D是向量维度
        dot_products = torch.sum(batch_u * batch_v, dim=1)
        # 返回点积平方的平均值作为损失
        loss = torch.mean(dot_products ** 2)
        return loss


    def forward(self, x, tau=100.0, cond=None):

        num_select = int(x.shape[1]*self.hparams.sample_rate_feature)
        weight = self.get_weight(cond=cond)
        # import pdb; pdb.set_trace()
        if tau is not None:
            self.mask = muti_gumbel(weight, tau=tau, hard=True, top_N=num_select)
            # x = x * self.mask[:,0,:]
        else:
            top_k_index = weight.topk(num_select, dim=1)[1]
            # import pdb; pdb.set_trace()
            self.mask = torch.zeros_like(weight)
            self.mask = self.mask.scatter_(1, top_k_index, 1) > 0
            # x = x * self.mask
        lat_high_dim = self.enc(x, mask=self.mask)
        lat_vis = self.vis(lat_high_dim)
                
        return lat_high_dim, lat_vis, weight

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
        lat_high_dim, lat_vis, weight = self(
            data_input_item, 
            tau=self.tau if self.tau>1 else None,
            cond = torch.cat([
                cond.detach().to(self.device), 
                cond.detach().to(self.device)])
            )
        lat_vis = (lat_vis - lat_vis.mean(0).detach()) / lat_vis.std(0).detach()
        
        # import pdb; pdb.set_trace()
        # self.batch_orthogonal_loss()
        
        loss_topo_ne, loss_topo_po = self.LossManifold(
            input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            sigma=self.hparams.sigma,
            w_nb=self.hparams.w_nb,
            w_fp=self.hparams.w_fp
        )
        
        # self.vis_dictionary[index] = lat_vis[:data_input_aug.shape[0]]
        
        loss_topo = (loss_topo_ne*self.hparams.weight_nepo + loss_topo_po)/(1+self.hparams.weight_nepo) # + loss_crossentropy
        loss_mse = ((self.dec(lat_high_dim)-data_input_item)**2).mean()
        loss_all = loss_topo + loss_mse*self.hparams.weight_mse #+ kloss# + mlm_loss / 10

        self.log('loss_all', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('loss_mse', loss_mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_topo_ne', loss_topo_ne, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_topo_po', loss_topo_po, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('lr', float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_all

    def validation_step(self, batch, batch_idx, test=False, dataloader_idx=0):
        # data_input_item, data_input_aug, label, index, = batch
        
        if dataloader_idx == 0:
            data_input_item = batch['data_input_item']
            data_input_aug = batch['data_input_aug']
            index = batch['index']

            if self.vis_dictionary.device != self.device:
                self.vis_dictionary = self.vis_dictionary.to(self.device)
                # match the dtype of vis_dictionary

            cond = self.vis_dictionary[index.to(self.vis_dictionary.device)]
            lat_high_dim, lat_vis, weight = self(
                data_input_item,
                cond = cond.detach().to(self.device)
                )
            data_recons = self.dec(lat_high_dim)
            
            if self.vis_dictionary.dtype != lat_vis.dtype:
                self.vis_dictionary = self.vis_dictionary.to(lat_vis.dtype)
            
            self.val_validation_step_outputs_high = lat_high_dim
            self.val_validation_step_outputs_vis = lat_vis
            self.val_validation_step_outputs_recons = data_recons
        if dataloader_idx == 1:
            data_input_item = batch['data_input_item']
            data_input_aug = batch['data_input_aug']
            index = batch['index']

            if self.vis_dictionary.device != self.device:
                self.vis_dictionary = self.vis_dictionary.to(self.device)
                # match the dtype of vis_dictionary

            cond = self.vis_dictionary[index.to(self.vis_dictionary.device)]
            lat_high_dim, lat_vis, weight = self(
                data_input_item,
                cond = cond.detach().to(self.device)
                )
            data_recons = self.dec(lat_high_dim)
            
            if self.vis_dictionary.dtype != lat_vis.dtype:
                self.vis_dictionary = self.vis_dictionary.to(lat_vis.dtype)
            
            self.train_validation_step_outputs_high = lat_high_dim
            self.train_validation_step_outputs_vis = lat_vis
            self.train_validation_step_outputs_recons = data_recons
    
    def test_step(self, batch, batch_idx):
        
        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        label = batch['label']

        if self.vis_dictionary.device != self.device:
            self.vis_dictionary = self.vis_dictionary.to(self.device)
            # match the dtype of vis_dictionary

        cond_rand = torch.randn((data_input_item.shape[0], 2)).to(self.vis_dictionary.device)
        lat_high_dim, cond, weight = self(
            data_input_item,
            cond = cond_rand.detach().to(self.device)
            )
        
        # cond = lat_vis
        lat_high_dim, cond, weight = self(
            data_input_item,
            cond = cond.detach().to(self.device)
            )
        
        lat_high_dim, lat_vis, weight = self(
            data_input_item,
            cond = cond.detach().to(self.device)
            )

        self.test_step_outputs_high = lat_high_dim
        self.test_step_outputs_vis = lat_vis
        self.test_step_outputs_label = label

        # import pdb; pdb.set_trace()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            weight_decay=self.hparams.weight_decay, 
            lr=self.hparams.lr
            )
        return optimizer

