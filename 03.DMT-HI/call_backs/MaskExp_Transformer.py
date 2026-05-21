from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np

import matplotlib.pyplot as plt

class MaskExpCallBack(Callback):
    def __init__(self, inter=10, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.plot_sample_train = 0
        self.plot_sample_test = 0
        self.val_mask_list = []
        
        self.val_ori_single_image_list = []


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        data_input_item = batch["data_input_item"]
        
        mask_this_batch = pl_module.mask
        
        self.val_mask_list.append(mask_this_batch)
        self.val_ori_single_image_list.append(data_input_item)

    def plot_statistic(self, gathered_mask, ori_img, trainer):        
        # plot mean with heatmap
        if gathered_mask.shape[1] == 784:
            mask_mean = np.mean(gathered_mask, axis=0).reshape(28, 28)
        else:
            # rise an error
            raise ValueError("Not a 28x28 mask")
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_mean, cmap='viridis')
        plt.colorbar()
        trainer.logger.experiment.log({"mask_mean": [wandb.Image(plt)]})
        plt.close()
    
    def plot_single_image_and_mask(self, gathered_mask, gathered_ori_img, trainer):

        plt.figure(figsize=(10, 30))
        for i in range(5):
            plt.subplot(5, 2, i*2+1)
            plt.imshow(gathered_ori_img[i].reshape(28, 28), cmap='gray')
            plt.colorbar()
            plt.subplot(5, 2, i*2+2)
            plt.imshow(gathered_mask[i].reshape(28, 28), cmap='viridis')
            plt.colorbar()
        trainer.logger.experiment.log({"mask_single": [wandb.Image(plt)]})
        plt.close()



    def on_validation_epoch_end(self, trainer, pl_module):
        
        # if trainer.current_epoch % self.inter == 0:
        import pdb; pdb.set_trace()
        mask = torch.cat(self.val_mask_list).to(pl_module.device)
        ori_img = torch.cat(self.val_ori_single_image_list).to(pl_module.device)
        
        if trainer.is_global_zero:  # 检查是否为 rank 0 的进程
            
            gathered_mask= trainer.strategy.all_gather(mask).cpu().detach().numpy()
            gathered_ori_img = trainer.strategy.all_gather(ori_img).cpu().detach().numpy()
            
            self.plot_statistic(gathered_mask, ori_img, trainer)
            self.plot_single_image_and_mask(gathered_mask, gathered_ori_img, trainer)
            # shape (N, num_features)

            
        self.val_ori_single_image_list = []
        self.val_mask_list = []