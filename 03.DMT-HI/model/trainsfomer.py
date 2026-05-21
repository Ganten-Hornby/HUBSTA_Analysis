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

from util import (
    LineBackground,
    ScatterVis,
    ScatterCenter,
    Layout,
    OuterRing,
    OuterRing_S,
)
from model.fa import BertModel as fa_BertModel
from model.fa import BertConfig as fa_BertConfig

import scipy
import uuid
import matplotlib.pyplot as plt

import eval.eval_core as ec
import eval.eval_core_base as ecb

from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dataloader import data_rankseq as data_base

# import lightning_lite
import manifolds

# from kornia.augmentation import Normalize, RandomGaussianBlur, RandomSolarize, RandomGrayscale, RandomResizedCrop, RandomCrop, ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
import math

new_uuid = str(uuid.uuid4())

class ToPoincareModel(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """

    def __init__(self, c, manifold="PoincareBall"):
        super(ToPoincareModel, self).__init__()
        self.c = c
        self.manifold = manifold  # 'PoincareBall'
        self.manifold = getattr(manifolds, self.manifold)()

    def forward(self, x):
        z = (
            self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c),
                c=self.c,
            )
            / 1.414
        )
        return z


def gpu2np(a):
    return a.cpu().detach().numpy()


def gpu2cpu(a):
    return a.detach()


class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        m_l.append(nn.BatchNorm1d(out_dim))

        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)


class DMTEVT_Encoder(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
        sample_len,
        trans_embedding_size,
        num_attention_heads=6,
        intermediate_size=128,
    ):
        super(DMTEVT_Encoder, self).__init__()
        self.data_name = data_name
        self.l_token = l_token
        self.l_token2 = l_token2
        # self.pos_encoder = PositionalEncoding(d_model=l_token2, max_len=sample_len)

        self.model_transformer1 = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
            trans_embedding_size=trans_embedding_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(sample_len),
            nn.Linear(sample_len, 500),
            nn.ReLU(),
            nn.Linear(500, l_token2),
            nn.BatchNorm1d(l_token2),
        )
        self.linear = nn.Linear(768, num_input_dim)

    def forward(self, x, masked_tokens_mask):

        batch_size = x.shape[0]
        if "Cifar" in self.data_name:
            lat_high_dim = self.model_transformer2(x).reshape((batch_size, -1))
        else:
            # if len(x.shape)==2:
            #     x = self.tokenlization(x, self.n_token_feature, batch_size)
            # word_len = x.shape[1]
            # x = x.reshape((batch_size * word_len, self.l_token))
            # lat = self.model_token(x)
            # lat = self.pos_encoder(lat)
            # data_pre = lat.reshape(batch_size, word_len, self.l_token2)

            # train_all = self.model_transformer1(input_ids=x)
            train_all = self.model_transformer1(
                input_ids=x,
                ).last_hidden_state
            lat_high_dim_1 = train_all.mean(dim=2)
            predict_vac = self.linear(train_all)

            # lat_high_dim_1 = self.model_transformer1(x.float())

            # lat_high_dim_2 = self.model_transformer2(lat.reshape(batch_size, -1))
            # print('lat_high_dim_1', lat_high_dim_1.shape)
            lat_high_dim = self.batchnorm(lat_high_dim_1)

        return lat_high_dim, predict_vac

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
        trans_embedding_size=64,
        num_attention_heads=6,
        intermediate_size=128,
    ):
        # model_token = nn.Embedding(num_input_dim, l_token_2)

        if "Cifar" not in data_name:

            config = fa_BertConfig(
                vocab_size=num_input_dim+2,  # 词汇表大小
                # embedding_size=trans_embedding_size,  # 嵌入层大小
                hidden_size=768,  # 隐藏层大小
                num_hidden_layers=num_layers_Transformer,  # 隐藏层的数量
                num_attention_heads=num_attention_heads,  # 注意力头的数量
                intermediate_size=intermediate_size,  # 前馈网络的大小
                max_position_embeddings=num_input_dim + 1,  # 最大序列长度
                # num_hidden_layers=num_layers_Transformer,
            )
            model_transformer1 = fa_BertModel(config)

        return model_transformer1


class DMTEVT_Vis(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Vis, self).__init__()
        self.model_down = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, lat_high_dim):
        lat_vis = self.model_down(lat_high_dim)
        return lat_vis

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        if "Cifar" not in data_name:
            m_b = []
            m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
            m_b.append(nn.BatchNorm1d(2))
            model_down = nn.Sequential(*m_b)
        else:
            m_b = []
            m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
            model_down = nn.Sequential(*m_b)

        return model_down


class DMTEVT_Decoder(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Decoder, self).__init__()
        self.model_decoder = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, x):
        return self.model_decoder(x)

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        m_decoder = []
        m_decoder.append(NN_FCBNRL_MM(l_token_2, 100))
        m_decoder.append(NN_FCBNRL_MM(100, 100))
        m_decoder.append(NN_FCBNRL_MM(100, num_input_dim, use_RL=False))
        # m_decoder.append(nn.Sigmoid())
        model_decoder = nn.Sequential(*m_decoder)

        return model_decoder


class DMTEVT_model(LightningModule):
    def __init__(
        self,
        lr=0.001,
        nu=0.01,
        data_name="mnist",
        batch_size=900,
        max_epochs=1000,
        class_num=7,
        steps=20000,
        num_pat=8,
        num_fea_aim=500,
        l_token=50,
        l_token_2=50,
        num_layers_Transformer=1,
        num_latent_dim=2,
        num_input_dim=64,
        laten_down=500,
        weight_decay=0.0001,
        kmean_scale=0.01,
        loss_rec_weight=1.0,
        preprocess_epoch=100,
        transformer2_indim=3750,
        marker_size=2,
        sample_len=500,
        trans_embedding_size=64,
        num_attention_heads=6,
        w=1,
        intermediate_size=128,
        uselabel=False,
        **kwargs,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.dictinputdict = {}
        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.wandb_logs = {}
        self.mse = torch.nn.MSELoss()  #
        self.ce = torch.nn.CrossEntropyLoss()
        # self.setup()
        # self.step_per_epoch = n_train_sample // self.hparams.batch_size
        # self.epochs = self.hparams.steps // self.step_per_epoch
        self.center_change = False
        self.train_state = "train"
        
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        self.validation_step_outputs_data = []
        self.validation_step_reconstruct = []
        self.validation_step_outputs_lat = []
        self.validation_step_outputs_label = []
        # self.validation_step_outputs_index = []

        self.enc = DMTEVT_Encoder(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
            sample_len=self.hparams.sample_len,
            trans_embedding_size=self.hparams.trans_embedding_size,
            num_attention_heads=self.hparams.num_attention_heads,
            intermediate_size=self.hparams.intermediate_size,
        )
        self.dec = DMTEVT_Decoder(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )
        self.vis = DMTEVT_Vis(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, class_num),
        )

        if "Cifar" in data_name:
            self.transform_base = torch.nn.Sequential(
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            )
            self.transform = torch.nn.Sequential(
                RandomResizedCrop((32, 32)),  # 随机裁剪和调整大小
                RandomHorizontalFlip(),  # 随机水平翻转
                ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2, p=0.8
                ),
                RandomGrayscale(p=0.2),
                RandomSolarize(0.2),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            )

        if self.hparams.num_fea_aim < 1:
            self.hparams.num_fea_aim = int(
                self.hparams.num_input_dim * self.hparams.num_fea_aim
            )
        else:
            self.hparams.num_fea_aim = int(self.hparams.num_fea_aim)
        self.hparams.num_fea_aim = min(
            self.hparams.num_fea_aim, self.hparams.num_input_dim
        )

        self.Kcluster_layer = nn.Linear(
            self.hparams.l_token_2, self.hparams.class_num, bias=False
        )
        self.kmean_normalize_cluster_center()
        self.ToPoincare = ToPoincareModel(
            c=0.5,
        )

    def _DistanceSquared(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-6)

        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()

    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
            gamma
            * torch.tensor(2 * 3.14)
            * gamma
            * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij

    def LossManifold(
        self,
        v_input,
        input_data,
        latent_data,
        v_latent,
        w=0.1,
        metric="euclidean",
        label=None,
    ):
        # normalize the input and latent data
        # import pdb; pdb.set_trace()
        # input_data = input_data / torch.std(input_data, dim=0).detach()
        # latent_data = latent_data / torch.std(latent_data, dim=0).detach()

        data_1 = input_data[: input_data.shape[0] // 2]
        dis_P = self._DistanceSquared(data_1, data_1)
        latent_data_1 = latent_data[: input_data.shape[0] // 2]
        dis_P_2 = dis_P  # + nndistance.reshape(1, -1)
        P_2 = self._Similarity(dist=dis_P_2, gamma=self._CalGamma(v_input), v=v_input)
        latent_data_2 = latent_data[(input_data.shape[0] // 2) :]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        eye_mask = torch.eye(P_2.shape[0]).to(input_data.device)
        loss_ce_posi = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 1], Q_=Q_2[eye_mask == 1]
        )
        loss_ce_nega = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 0], Q_=Q_2[eye_mask == 0]
        )
        w1, w2 = 1 / (1 + w), w / (1 + w)
        return w2 * loss_ce_nega, w1 * loss_ce_posi / self.hparams.batch_size

    def kmean_normalize_cluster_center(self):
        self.Kcluster_layer.weight.data = (
            F.normalize(self.Kcluster_layer.weight.data, dim=1) * 2.0
        )

    def kcompute_cluster_center(self):
        center_high_dim = self.Kcluster_layer.weight
        return self.vis(center_high_dim).detach().cpu().numpy()

    def kmeans_hand_center(self, latenvec):
        index = torch.randint(
            low=0,
            high=len(latenvec) // 2,
            size=(self.hparams.class_num,),
        )
        center = latenvec[index]
        return center

    def kmeans_loss(self, mid):
        mid = mid.detach()
        dis_cluster = self._DistanceSquared(mid, self.Kcluster_layer.weight) / 100
        sim = self._Similarity(dist=dis_cluster, gamma=self._CalGamma(1), v=1)
        soft_label = F.softmax(sim.detach(), dim=1)
        hard_label = torch.argmax(soft_label, dim=1)
        delta = torch.zeros(
            (mid.shape[0], self.hparams.class_num), requires_grad=False
        ).cuda()
        label_range = torch.arange(0, delta.size(0)).long()
        delta[label_range, hard_label] = 1
        loss_clu_batch = 1 - torch.mul(delta, sim)
        loss_clu_batch = self.hparams.kmean_scale * loss_clu_batch.mean()
        return loss_clu_batch

    def forward(self, x):
        lat_high_dim, predict_vac = self.enc(x)
        lat_vis = self.vis(lat_high_dim)
        return lat_high_dim, lat_vis, predict_vac

    # def forward(self, x, masked_tokens_mask, batch_idx):
    #     return self.forward_fea(x, masked_tokens_mask, batch_idx)

    def encoder(self, x):
        return (self.forward_fea(torch.tensor(x), 0)[0]).detach().cpu().numpy()

    def decoder(self, x):
        return self.model_decoder(torch.tensor(x)).detach().cpu().numpy()

    def _Distance_squared_CPU(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        d = dist.clamp(min=1e-36)
        return d.detach().cpu().numpy()

    def pw_cosine_distance(self, input_a, input_b):
        normalized_input_a = torch.nn.functional.normalize(input_a)
        normalized_input_b = torch.nn.functional.normalize(input_b)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        res *= -1  # 1-res without copy
        res += 1
        return res

    def forward_simi(self, x, features, nu=1):
        # check x is tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        lat_high_dim, lat_vis = self.forward_fea(x, features, batch_idx=0)

        self.aim_cluster = 0
        center = self.Kcluster_layer.weight.data.detach()
        center_vis = self.vis(center)
        # dis = torch.norm(lat_high_dim - center, dim=1)
        dis = self._DistanceSquared(lat_vis, center_vis) / 1000
        # print(dis)
        # out = self.pw_cosine_distance(lat_high_dim, center)
        out_dis = self._Similarity(dist=dis, gamma=self._CalGamma(nu), v=nu)
        # out = 1/dis
        out = out_dis / out_dis.max(dim=1, keepdim=True)[0]
        # out = torch.softmax(out_dis, dim=1)
        # import pdb; pdb.set_trace()
        # print(out)
        return out

    def AE_predict(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()

            x = self.tokenlization(x, self.n_token_feature, x.shape[0])

            lat_high_dim, lat_vis = self.forward_fea(x, batch_idx=0)
            decode_data = self.dec(lat_high_dim)
            return gpu2np(decode_data)

    def training_step(self, batch, batch_idx):
        
        data_input_item, data_input_aug, label, index, = batch

        data_input_item = torch.cat([data_input_item, data_input_aug])
        lat_high_dim, lat_vis, predict_vac = self(data_input_item)
        
        loss_topo_ne, loss_topo_po = self.LossManifold(
            v_input=100,
            input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            v_latent=self.hparams.nu,
            w=self.hparams.w,
            label=label if self.hparams.uselabel else None,
        )
        
        loss_topo = loss_topo_ne + loss_topo_po # + loss_crossentropy

        self.wandb_logs = {
            # "loss_mse": loss_mse,
            # "loss_crossentropy": loss_crossentropy.item(),
            # "mlm_loss": mlm_loss.item(),
            "loss_topo_ne": loss_topo_ne.item(),
            "loss_topo_po": loss_topo_po.item(),
            # "loss_rec": loss_rec.item(),
            # "loss_kmeans": kloss.item(),
            "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
            # "epoch": float(self.current_epoch),
            # "T": self.t_list[self.current_epoch],
        }

        # if self.current_epoch >= self.hparams.preprocess_epoch:
        # loss_all = loss_topo + loss_rec + kloss + mlm_loss / 100
        loss_all = loss_topo # + loss_rec + kloss# + mlm_loss / 10

        return loss_all

    def validation_step(self, batch, batch_idx, test=False):
        # augmentation
        # if (self.current_epoch + 1) % self.hparams.log_interval == 0:
        # if test:
        # rank_input, data_input_item, label, index, n_token_feature = batch
        # else:
        data_input_item, data_input_aug, label, index, = batch

        batch_size = index.shape[0]
        # self.n_token_feature = n_token_feature
        # if 'Cifar' in self.hparams.data_name:
        #     data = self.transform_base(data_after_tokened.permute(0,3,1,2)/255)
        # else:

        data = rank_input
        # data_aug = self.transform(data_aug.permute(0,3,1,2)/255)
        masked_tokens_mask = label
        lat_high_dim, lat_vis, predict_vac = self(rank_input, masked_tokens_mask, batch_idx)
        data_recons = self.dec(lat_high_dim)

        self.validation_step_outputs_data.append(gpu2cpu(data)[:batch_size])
        self.validation_step_outputs_lat.append(gpu2cpu(lat_vis)[:batch_size])
        self.validation_step_reconstruct.append(gpu2cpu(data_recons)[:batch_size])
        self.validation_step_outputs_label.append(gpu2cpu(label))
        # return data.mean()

    def on_validation_epoch_end(self):
        data = torch.cat(self.validation_step_outputs_data)
        emb_vis = torch.cat(self.validation_step_outputs_lat)
        label = torch.cat(self.validation_step_outputs_label)
        reconstruction = torch.cat(self.validation_step_reconstruct)
        gathered_data = gpu2np(data)
        gathered_emb_vis = gpu2np(emb_vis)
        gathered_label = gpu2np(label)

        # import pdb; pdb.set_trace()

        if self.trainer.is_global_zero and self.current_epoch>0:
            ecb_e_train = ecb.Eval(
                input=gathered_data, latent=gathered_emb_vis, label=gathered_label, k=10
            )
            cluster_center = None
            # cluster_center = self.kcompute_cluster_center()
            self.wandb_logs.update(
                {
                    "SVC_{}".format(
                        self.train_state
                    ): ecb_e_train.E_Classifacation_SVC(),  # SVC= ecb_e.E_Classifacation_SVC(),
                    "KNN_{}".format(
                        self.train_state
                    ): ecb_e_train.E_Clasting_Kmeans(),  # SVC= ecb_e.E_Classifacation_SVC(),
                }
            )
            print(self.wandb_logs)
            self.log_dict(self.wandb_logs)
            self.wandb_logs.clear()
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            dict_imgs = {
                "fig_st_{}".format(self.train_state): self.px_plot_embedding(
                    gathered_emb_vis, gathered_label, platform="10x", save_path=None
                ),
                "fig_{}".format(self.train_state): self.plot_scatter(
                    gathered_emb_vis, gathered_label, cluster_center, P=False
                ),
                "fig_H_{}".format(self.train_state): self.plot_scatter(
                    gathered_emb_vis, gathered_label, cluster_center, P=True
                ),
                "fig_H_{}_clearn".format(self.train_state): self.plot_scatter(
                    gathered_emb_vis, gathered_label, cluster_center, P=True, clean=True
                ),
            }
            if reconstruction.shape[1] == 784:
                rec_tensor = torch.tensor(reconstruction[:64]).reshape(-1, 1, 28, 28)
                grid_image_recons = (
                    vutils.make_grid(rec_tensor, normalize=False, nrow=8)
                    .detach()
                    .cpu()
                    .numpy()
                )
                grid_image_recons_image = wandb.Image(
                    grid_image_recons.transpose(1, 2, 0)
                )
                dict_imgs.update(
                    {
                        "reconstruction_{}".format(
                            self.train_state
                        ): grid_image_recons_image,
                    }
                )

            self.logger.experiment.log(dict_imgs)
        self.validation_step_outputs_data.clear()
        self.validation_step_outputs_lat.clear()
        self.validation_step_outputs_label.clear()
        self.validation_step_reconstruct.clear()

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        self.train_state = "test"
        self.validation_step(batch, batch_idx, test=True)

    def on_test_epoch_end(self):
        # Here we just reuse the validation_epoch_end for testing
        print("------------")
        print("----test----")
        print("------------")
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        # parameters = [
        #     {'params': self.enc.parameters(), 'lr': self.hparams.lr},
        #     {'params': self.dec.parameters(), 'lr': self.hparams.lr},
        #     {'params': self.vis.parameters(), 'lr': self.hparams.lr},
        #     {'params': self.discriminator.parameters(), 'lr': self.hparams.lr},
        #     {'params': self.Kcluster_layer.parameters(), 'lr': self.hparams.lr*100}  # 设置 conv 层的学习率为 0.01
        # ]

        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=self.hparams.weight_decay
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.hparams.max_epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def px_plot_embedding(
        self,
        latent,
        label,
        platform="10x",
        save_path="data_{}.png".format(new_uuid),
        title=None,
    ):
        title = f"{len(np.unique(label))} classes" if title is None else title

        embedding = latent

        # Save Figure
        if save_path is not None:
            figsize = (7, 7) if platform == "10x" else (10, 10)

            fig_plt, ax = plt.subplots(figsize=figsize, dpi=400)
            ax.scatter(
                x=embedding[:, 0], y=embedding[:, 1], c=self._map_color(label), s=1
            )
            plt.axis("off")
            fig_plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            return wandb.Image(save_path)

    def _map_color(self, label: np.ndarray, style="p_d24") -> list:
        if style == "bright":
            color_scheme = [
                "#FD3216",
                "#00FE35",
                "#6A76FC",
                "#FED4C4",
                "#FE00CE",
                "#0DF9FF",
                "#F6F926",
                "#FF9616",
                "#479B55",
                "#EEA6FB",
                "#DC587D",
                "#D626FF",
                "#6E899C",
                "#00B5F7",
                "#B68E00",
                "#C9FBE5",
                "#FF0092",
                "#22FFA7",
                "#E3EE9E",
                "#86CE00",
                "#BC7196",
                "#7E7DCD",
                "#FC6955",
                "#E48F72",
            ]
        elif style == "dark":
            color_scheme = [
                "#4e79a7",
                "#f28e2c",
                "#e15759",
                "#76b7b2",
                "#59a14f",
                "#edc949",
                "#af7aa1",
                "#ff9da7",
                "#9c755f",
                "#bab0ab",
                "#1b9e77",
                "#d95f02",
                "#7570b3",
                "#e7298a",
                "#66a61e",
                "#e6ab02",
                "#a6761d",
                "#666666",
            ]
        elif style == "p_d24":
            color_scheme = [
                "#2E91E5",
                "#E15F99",
                "#1CA71C",
                "#FB0D0D",
                "#DA16FF",
                "#222A2A",
                "#B68100",
                "#750D86",
                "#EB663B",
                "#511CFB",
                "#00A08B",
                "#FB00D1",
                "#FC0080",
                "#B2828D",
                "#6C7C32",
                "#778AAE",
                "#862A16",
                "#A777F1",
                "#620042",
                "#1616A7",
                "#DA60CA",
                "#6C4516",
                "#0D2A63",
                "#AF0038",
                "#D3D3D3",
            ]

        return [color_scheme[i % len(color_scheme)] for i in label]

    def plot_scatter(self, emb_vis, labels, cluster_center, P=True, clean=False):
        if cluster_center is None:
            cluster_center = np.random.randn(30, 2)

        if cluster_center.shape[1] != 2:
            cluster_center = cluster_center[:, :2]
            emb_vis = emb_vis[:, :2]

        fig = go.Figure()
        if P:
            mean_values_ins = np.mean(emb_vis, axis=0)
            var_values_ins = np.std(emb_vis, axis=0) * 0.5
            emb_c_vis_center = np.concatenate([cluster_center, emb_vis])
            emb_c_vis_center = gpu2np(
                self.ToPoincare(
                    torch.Tensor((emb_c_vis_center - mean_values_ins) / var_values_ins)
                )
            )
            emb_vis = emb_c_vis_center[cluster_center.shape[0] :]
            cluster_center = emb_c_vis_center[: cluster_center.shape[0]]
            if not clean:
                fig = LineBackground(fig, self)
                fig = OuterRing(fig, cluster_center, labels)
            else:
                fig = LineBackground(fig, self)
                # fig = OuterRing_S(fig, cluster_center, labels)

        fig = ScatterVis(fig, emb_vis, labels, size=self.hparams.marker_size)
        fig = ScatterCenter(fig, cluster_center, labels)
        fig = Layout(fig)

        return fig

    def pdist2(self, x: torch.Tensor, y: torch.Tensor):
        # calculate the pairwise distance
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)
        return dist

