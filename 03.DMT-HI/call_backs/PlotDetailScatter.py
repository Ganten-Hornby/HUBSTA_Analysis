

from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def euclidean_to_hyperbolic_matrix(u, c=0.5, min_norm = 1e-15):
    u = torch.tensor(u).float()
    
    u = 1.5 * ( u-u.mean(dim=0) )/u.std(dim=0)
    
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min_norm)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1.detach().numpy()



class PlotDetailScatterBack(Callback):
    def __init__(self, inter=10, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.plot_sample_train = 0
        self.plot_sample_test = 0
        # self.train_len_list = []
        # self.train_acc_list = []
        self.val_input= {'val1': [], 'val2': [], 'val3': []}
        self.val_high = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis_exp = {'val1': [], 'val2': [], 'val3': []}
        self.val_label = {'val1': [], 'val2': [], 'val3': []}


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]
        key_value = f"val{dataloader_idx+1}"
        
        self.val_input[key_value].append(pl_module.validation_origin_input)
        self.val_high[key_value].append(pl_module.validation_step_outputs_high)
        self.val_vis[key_value].append(pl_module.validation_step_outputs_vis)
        self.val_vis_exp[key_value].append(pl_module.validation_step_lat_vis_exp)
        
        # self.validation_weight = pl_module.validation_weight
        # self.val_recon.append(pl_module.validation_step_outputs_recons)
        self.val_label[key_value].append(label)

    def plot_scatter(self, scatter, img, label, trainer, hyper=False, text=''):
        
        if hyper:
            scatter = euclidean_to_hyperbolic_matrix(scatter)
        else:
            scatter = (scatter - np.mean(scatter, axis=0)) / np.std(scatter, axis=0)
        
        d3_color_list = 10 * ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        # 根据label选择颜色
        colors = [d3_color_list[class_number] for class_number in label]
        
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # 绘制散点图
        ax.scatter(scatter[:, 0], scatter[:, 1], c=colors, s=5, alpha=0.7)
        
        if hyper:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            # 移除坐标轴
            ax.axis('off')
        else:
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        
        fig.tight_layout()
        plt.savefig(f"plots_results/scatter_{text}_{trainer.current_epoch}.pdf", dpi=600)
        plt.close()


    def plot_scatter_with_img(self, scatter, img, label, trainer, hyper = False, text=''):
        
        if hyper:
            scatter = euclidean_to_hyperbolic_matrix(scatter)
        else:
            scatter = (scatter - np.mean(scatter, axis=0)) / np.std(scatter, axis=0)
        
        d3_color_list = 10*["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        cmap_list = []
        for i in range(len(d3_color_list)):
            # 修改颜色，将白色部分设为透明
            colors = [(1, 1, 1, 0), d3_color_list[i]]  # RGBA 格式：白色为透明
            custom_cmap = LinearSegmentedColormap.from_list(f"custom_cmap{i}", colors)
            cmap_list.append(custom_cmap)

        fig, ax = plt.subplots(figsize=(20, 16))
        
        if img[i].shape[0] == 784:
            for i in range(scatter.shape[0]):
                class_number = label[i]
                fig = plt.figure(figsize=(2, 2))
                # plt.hist(data_item, bins=20, color=d3_color_list[label[i]], histtype='stepfilled')
                plt.imshow(img[i].reshape(28, 28), cmap=cmap_list[class_number])
                plt.axis('off')
                # plt.tight_layout(h_pad=-0.5, w_pad=-0.5)
                plt.subplots_adjust(left=0.00, right=0.99, top=0.99, bottom=0.01) 
                plt.savefig('save.png', format='png')
                plt.close(fig)
                
                bar_img = Image.open('save.png').convert('RGBA')
                imagebox = OffsetImage(bar_img, zoom=0.1)
                # imagebox = OffsetImage(bar_img, zoom=0.2)
                # ab = AnnotationBbox(imagebox, scatter[i], frameon=True, edgecolor=)
                ab = AnnotationBbox(
                    imagebox, 
                    scatter[i], 
                    frameon=True,  # 启用方框
                    bboxprops=dict(edgecolor=d3_color_list[label[i]], linewidth=0.5, linestyle='--')  # 设置边框样式
                )
                ax.add_artist(ab)
        else:
            # plot a bar plot of feature distribution in the scatter position
            
            # import pdb; pdb.set_trace()
            img_feature = img.T
            num_cluster = 8
            # cluster the feature
            kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(img_feature)
            kmeans_label = kmeans.labels_
            
            ave = []
            for f in range(num_cluster):
                ave.append(img[:, kmeans_label==f].mean())
                
            ave_sort_list = np.argsort(ave).tolist()
            
            data_new_list = []
            for i in  range(scatter.shape[0]):
                data_new = []
                for f in ave_sort_list:
                    # data_new.append(img[i][kmeans_label==f].mean()/ave[f] - 0.5 )
                    data_new.append(img[i][kmeans_label==f].mean()-ave[f])
                data_new_list.append(data_new)
            
            data_new_list = np.array(data_new_list)     
            
            # data_new_list = data_new_list-data_new_list.min()
            
            for i in  range(scatter.shape[0]):
                print(i, scatter.shape)                
                data_item = data_new_list[i]
                # data_item = data_item[data_item>0.25]
                # import pdb; pdb.set_trace()
                # Generate a bar first and convert it to an image
                fig = plt.figure(figsize=(2, 2))
                # plt.hist(data_item, bins=20, color=d3_color_list[label[i]], histtype='stepfilled')
                plt.bar(range(len(data_item)), data_item, color=d3_color_list[label[i]])
                plt.ylim(-0.3, 0.3)
                plt.axis('off')
                # plt.tight_layout(h_pad=-0.5, w_pad=-0.5)
                plt.subplots_adjust(left=0.00, right=0.99, top=0.99, bottom=0.01) 
                plt.savefig('save.png', format='png')
                plt.close(fig)
                # buf.seek(0)
                bar_img = Image.open('save.png').convert('RGBA')
                
                # Add the bar image to the scatter plot
                imagebox = OffsetImage(bar_img, zoom=0.04)
                # ab = AnnotationBbox(imagebox, scatter[i], frameon=True, edgecolor=)
                ab = AnnotationBbox(
                    imagebox, 
                    scatter[i], 
                    frameon=True,  # 启用方框
                    bboxprops=dict(edgecolor=d3_color_list[label[i]], linewidth=0.5, linestyle='--')  # 设置边框样式
                )
                ax.add_artist(ab)
                
        
        if hyper:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            # remove axis
            ax.axis('off')
        else:
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        fig.tight_layout()
        plt.savefig(f"plots_results/scatter_detail_{text}_{trainer.current_epoch}.pdf", dpi=900)
        plt.close()
        
        # trainer.logger.log_metrics({
        #     # "SVC_val": acc_mean, 
        #     "epoch": trainer.current_epoch,
        #     "plots_results/scatter_detail": wandb.Image(fig)
        #     })

    def on_validation_epoch_end(self, trainer, pl_module):
        # if trainer.current_epoch % self.inter == 0:
        # import pdb; pdb.set_trace()
        
        num_val = len(self.val_vis)
        
        for val_index in range(num_val):
            
            val_name = f"val{val_index+1}"
            
            val_input_current = self.val_input[val_name]
            val_vis_current = self.val_vis[val_name]
            val_vis_exp_current = self.val_vis_exp[val_name]
            hight_vis_current = self.val_high[val_name]
            val_label_current = self.val_label[val_name]
            
            print(val_name, len(val_vis_current), len(val_label_current))
            
            # if trainer.max_epochs == 1 and pl_module.visited or len(val_vis_current) > 0 and trainer.current_epoch > 1 and (trainer.current_epoch+1) % self.inter == 0:
            if True:
                
                if not isinstance(val_vis_current[0], np.ndarray):

                    val_input = torch.cat(val_input_current).cuda()
                    val_vis = torch.cat(val_vis_current).cuda()
                    val_vis_exp = torch.cat(val_vis_exp_current).cuda()
                    hight_vis = torch.cat(hight_vis_current).cuda()
                    val_label = torch.cat(val_label_current).cuda()

                    gathered_val_input = trainer.strategy.all_gather(val_input).cpu().detach().numpy()
                    gathered_val_vis = trainer.strategy.all_gather(val_vis).cpu().detach().numpy()
                    # gathered_val_vis_exp = trainer.strategy.all_gather(val_vis_exp).cpu().detach().numpy()
                    gathered_hight_vis = trainer.strategy.all_gather(hight_vis).cpu().detach().numpy()
                    gathered_val_label = trainer.strategy.all_gather(val_label).cpu().detach().numpy()
                else:

                    gathered_val_input = np.concatenate(val_input_current)
                    gathered_val_vis = np.concatenate(val_vis_current)
                    gathered_val_vis_exp = np.concatenate(val_vis_exp_current)
                    gathered_hight_vis = np.concatenate(hight_vis_current)
                    val_label = torch.cat(val_label_current).cuda()
                    gathered_val_label = trainer.strategy.all_gather(val_label).cpu().detach().numpy()
                
                if len(gathered_val_vis.shape) == 3:
                    gathered_val_vis = gathered_val_vis.reshape(-1, *gathered_val_vis.shape[2:])
                    gathered_hight_vis = gathered_hight_vis.reshape(-1, *gathered_hight_vis.shape[2:])
                    gathered_val_label = gathered_val_label.reshape(-1)
                
                if trainer.is_global_zero:          # 检查是否为 rank 0 的进程
                    # print('gathered_val_vis.shape', gathered_val_vis.shape)

                    # fig_scatter = self.plot_scatter(gathered_val_vis, gathered_val_label, trainer)
                    # fig_scatter_hyper = self.plot_scatter_hyper(gathered_val_vis, gathered_val_label, trainer)
                    

                    # dict_numm_labels = {}
                    # for l in gathered_val_label:
                    #     if l in dict_numm_labels:
                    #         dict_numm_labels[l] += 1
                    #     else:
                    #         dict_numm_labels[l] = 1
                    # dict_numm_labels = sorted(dict_numm_labels.items(), key=lambda x: x[1], reverse=True)
                    
                    # dict_numm_labels = dict_numm_labels[:10]
                    # label_list = [x[0] for x in dict_numm_labels]
                    import pdb; pdb.set_trace()
                    
                    # # for i in range(gathered_val_vis.shape[0]):
                    # mask = np.isin(gathered_val_label, label_list)
                    
                    # gathered_val_input = gathered_val_input[mask]
                    # gathered_val_vis = gathered_val_vis[mask]
                    # gathered_hight_vis = gathered_hight_vis[mask]
                    # gathered_val_label = gathered_val_label[mask]
                    
                    self.plot_scatter(
                        gathered_val_vis, 
                        gathered_val_input, 
                        gathered_val_label, 
                        trainer,
                        text=f'val_{val_index}',
                        hyper=True,
                        )

                    
                    if gathered_val_vis.shape[0] > 5000:
                        random_index = np.random.choice(gathered_val_vis.shape[0], 5000, replace=False)

                        gathered_val_input = gathered_val_input[random_index]
                        gathered_val_vis = gathered_val_vis[random_index]
                        gathered_hight_vis = gathered_hight_vis[random_index]
                        gathered_val_label = gathered_val_label[random_index]


                    
                
                    self.plot_scatter_with_img(
                        gathered_val_vis, 
                        gathered_val_input, 
                        gathered_val_label, 
                        trainer,
                        text=f'val_{val_index}',
                        hyper=True
                        )
            
        self.val_input = {'val1': [], 'val2': [], 'val3': []}    
        self.val_high = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis = {'val1': [], 'val2': [], 'val3': []}
        self.val_label = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis_exp = {'val1': [], 'val2': [], 'val3': []}
        