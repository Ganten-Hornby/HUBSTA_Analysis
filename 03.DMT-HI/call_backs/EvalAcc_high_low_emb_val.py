from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go



def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def euclidean_to_hyperbolic_matrix(u, c=0.5, R=1.5, min_norm=1e-15, norm=True):
    u = torch.tensor(u).float()

    if norm:
        u_mean = u.mean(dim=0)
        u_std = u.std(dim=0)
        u = R * (u - u_mean) / u_std

    sqrt_c = c**0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min_norm)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1.detach().numpy()


class EvalCallBack(Callback):
    def __init__(
        self, inter=10, dirpath="", dataset="", only_val=False, *args, **kwargs
    ):
        super().__init__()
        self.inter = inter
        self.plot_sample_train = 0
        self.plot_sample_test = 0
        self.only_val = only_val
        self.val_high = {"val1": [], "val2": [], "val3": []}
        self.val_vis = {"val1": [], "val2": [], "val3": []}
        self.val_label = {"val1": [], "val2": [], "val3": []}
        self.noist_test_result_dict = {"val1": [], "val2": [], "val3": []}

        self.test_high = []
        self.test_vis = []
        self.test_recon = []
        self.test_label = []

        self.dirpath = dirpath
        self.dataset = dataset
        self.best_acc = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]

        key_value = f"val{dataloader_idx+1}"

        self.val_high[key_value].append(pl_module.validation_step_outputs_high)
        self.val_vis[key_value].append(pl_module.validation_step_outputs_vis)
        self.val_label[key_value].append(label)
        self.noist_test_result_dict[key_value].append(pl_module.noist_test_result_dict)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]

        # self.test_high.append(pl_module.test_step_outputs_high)
        self.test_vis.append(pl_module.test_step_outputs_vis)
        # self.test_recon.append(pl_module.test_step_outputs_recons)
        self.test_label.append(label)

    def on_test_epoch_end(self, trainer, pl_module):

        test_vis = torch.cat(self.test_vis).cuda()
        test_label = torch.cat(self.test_label).cuda()
        gathered_val_vis = trainer.strategy.all_gather(test_vis).cpu().detach().numpy()
        gathered_val_label = (
            trainer.strategy.all_gather(test_label).cpu().detach().numpy()
        )

        gathered_val_vis = gathered_val_vis.reshape(-1, 2)
        gathered_val_label = gathered_val_label.reshape(-1)

        acc_mean = self.get_svc_acc(gathered_val_vis, gathered_val_label, trainer)
        print("gathered_val_vis", gathered_val_vis.shape)
        print("gathered_val_label", gathered_val_label.shape)
        print("acc_mean", acc_mean)

    def plot_scatter(self, gathered_val_vis, gathered_val_label, trainer):
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            gathered_val_vis[:, 0],
            gathered_val_vis[:, 1],
            c=gathered_val_label,
            cmap="rainbow",
            s=10,
        )
        return fig

    def plot_line_single_line(self, start, end, step, ax, color="grey", map=None):
        """
        Plots a single line from start to end with a given number of steps.

        Parameters:
        start (tuple): The starting point (x, y).
        end (tuple): The ending point (x, y).
        step (int): The number of steps (points) in the line.
        ax (matplotlib.axes.Axes): The axes on which to plot the line.
        color (str): The color of the line. Default is 'blue'.
        map (callable): An optional mapping function to transform the points.

        Returns:
        None
        """
        A = np.array(start)
        B = np.array(end)
        t = np.linspace(0, 1, step)
        x = A[0] + t * (B[0] - A[0])
        y = A[1] + t * (B[1] - A[1])

        if map is not None:
            data = map(np.array([x, y]).T, norm=False)
            x, y = data[:, 0], data[:, 1]
        # import pdb; pdb.set_trace()
        # print('x', x)
        # print('y', y)

        ax.plot(x, y, color=color, linewidth=1, alpha=0.2)

    def plot_back_ground(self, ax, step, map=None):

        self.plot_line_single_line(
            start=(-10, -10),
            end=(10, 10),
            step=500,
            ax=ax,
            map=euclidean_to_hyperbolic_matrix,
        )

        self.plot_line_single_line(
            start=(0, -10),
            end=(0, 10),
            step=500,
            ax=ax,
            map=euclidean_to_hyperbolic_matrix,
        )

        self.plot_line_single_line(
            start=(10, 10),
            end=(-10, -10),
            step=500,
            ax=ax,
            map=euclidean_to_hyperbolic_matrix,
        )

        self.plot_line_single_line(
            start=(-10, 10),
            end=(10, -10),
            step=500,
            ax=ax,
            map=euclidean_to_hyperbolic_matrix,
        )

        self.plot_line_single_line(
            start=(-10, 0),
            end=(10, 0),
            step=500,
            ax=ax,
            map=euclidean_to_hyperbolic_matrix,
        )

        for i in range(9):
            self.plot_line_single_line(
                start=(-2, i * 0.5 - 2),
                end=(2, i * 0.5 - 2),
                step=500,
                ax=ax,
                map=euclidean_to_hyperbolic_matrix,
            )

        for i in range(9):
            self.plot_line_single_line(
                start=(i * 0.5 - 2, -2),
                end=(i * 0.5 - 2, 2),
                step=500,
                ax=ax,
                map=euclidean_to_hyperbolic_matrix,
            )

    def plot_scatter_hyper(
        self, gathered_val_vis, gathered_val_label, trainer, R=1.5, exp_emb=None
    ):
        from matplotlib.patches import Circle

        u_mean = gathered_val_vis.mean(axis=0)
        u_std = gathered_val_vis.std(axis=0)
        # u = R * (u - u_mean) / u_std

        gathered_val_vis_hy = euclidean_to_hyperbolic_matrix(gathered_val_vis, R=R)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        self.plot_back_ground(ax, 500, map=euclidean_to_hyperbolic_matrix)

        plt.scatter(
            gathered_val_vis_hy[:, 0],
            gathered_val_vis_hy[:, 1],
            c=gathered_val_label,
            cmap="rainbow",
            s=10,
        )

        if exp_emb is not None:
            # exp_emb
            R_emb = 1.8
            circle = Circle((0, 0), R_emb, edgecolor="black", facecolor="none")
            ax.add_patch(circle)
            exp_emb_hy = (exp_emb - u_mean)/u_std
            exp_emb_hy = R_emb * exp_emb_hy / np.linalg.norm(exp_emb_hy, axis=1, keepdims=True)

            plt.scatter(exp_emb_hy[:, 0], exp_emb_hy[:, 1], c="black", s=100)

        circle = Circle((0, 0), R, edgecolor="black", facecolor="none")
        ax.add_patch(circle)
        # Remove the border
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def plot_scatter_hyper_plotly(
        self, gathered_val_vis, gathered_val_label, trainer, R=1.5, exp_emb=None
    ):


        u_mean = gathered_val_vis.mean(axis=0)
        u_std = gathered_val_vis.std(axis=0)

        gathered_val_vis_hy = euclidean_to_hyperbolic_matrix(gathered_val_vis, R=R)

        fig = go.Figure()

        scatter = go.Scatter(
            x=gathered_val_vis_hy[:, 0],
            y=gathered_val_vis_hy[:, 1],
            mode='markers',
            marker=dict(
                color=gathered_val_label,
                colorscale='Rainbow',
                size=2,
            ),
            text=[f"Label: {label}" for label in gathered_val_label],
            hoverinfo='text',
        )
        fig.add_trace(scatter)

        if exp_emb is not None:
            R_emb = 1.8
            exp_emb_hy = (exp_emb - u_mean) / u_std
            exp_emb_hy = R_emb * exp_emb_hy / np.linalg.norm(exp_emb_hy, axis=1, keepdims=True)

            exp_scatter = go.Scatter(
                x=exp_emb_hy[:, 0],
                y=exp_emb_hy[:, 1],
                mode='markers',
                marker=dict(
                    color='black',
                    size=3
                )
            )
            circle_l2 = go.Scatter(
                x=[R_emb * np.cos(theta) for theta in np.linspace(0, 2*np.pi, 100)],
                y=[R_emb * np.sin(theta) for theta in np.linspace(0, 2*np.pi, 100)],
                mode='lines',
                # line=dict(color='red'),
                line=dict(color='black', width=1),  # Adjust line width here
            )
            fig.add_trace(circle_l2)
            fig.add_trace(exp_scatter)

        circle_l1 = go.Scatter(
            x=[R * np.cos(theta) for theta in np.linspace(0, 2*np.pi, 100)],
            y=[R * np.sin(theta) for theta in np.linspace(0, 2*np.pi, 100)],
            mode='lines',
            # line=dict(color='black'),
            line=dict(color='black', width=1)  # Adjust line width here
        )
        fig.add_trace(circle_l1)

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
            width=800,
            height=800,
            paper_bgcolor='rgba(255,255,255,1)',  # Background color of the paper
            plot_bgcolor='rgba(255,255,255,1)',  # Background color of the plot
        )

        return fig



    def get_svc_acc(self, gathered_val_vis, gathered_val_label, trainer):

        method = SVC(max_iter=900000, kernel='linear')
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        n_scores = cross_val_score(
            method,
            StandardScaler().fit_transform(gathered_val_vis),
            gathered_val_label,
            scoring="accuracy",
            cv=cv,
            n_jobs=1,
        )
        acc_mean = np.mean(n_scores)
        return acc_mean

    def ins_exp_relation_with_shap(self, noist_test_result_dict):
        # print('000')

        num_ins = noist_test_result_dict.shape[0]
        num_exp = noist_test_result_dict.shape[2]

        mean_dis = noist_test_result_dict.mean(dim=1)
        mean_dis_topk = mean_dis.topk(3, dim=1).indices

        mask_exp_ins = torch.zeros((num_ins, num_exp))
        batch_indices = torch.arange(num_ins).unsqueeze(1).expand(-1, 3)
        mask_exp_ins[batch_indices, mean_dis_topk] = 1

        return mask_exp_ins

    def on_validation_epoch_end(self, trainer, pl_module):

        num_val = len(self.val_vis)

        for val_index in range(num_val):

            val_name = f"val{val_index+1}"

            val_vis_current = self.val_vis[val_name]
            hight_vis_current = self.val_high[val_name]
            val_label_current = self.val_label[val_name]
            noist_test_result_dict_current = self.noist_test_result_dict[val_name]

            print(val_name, len(val_vis_current), len(val_label_current))

            if len(val_vis_current) > 0:
                val_vis = torch.cat(val_vis_current).cuda()
                hight_vis = torch.cat(hight_vis_current).cuda()
                val_label = torch.cat(val_label_current).cuda()
                gathered_val_vis = (
                    trainer.strategy.all_gather(val_vis).cpu().detach().numpy()
                )
                gathered_hight_vis = (
                    trainer.strategy.all_gather(hight_vis).cpu().detach().numpy()
                )
                gathered_val_label = (
                    trainer.strategy.all_gather(val_label).cpu().detach().numpy()
                )

                gathered_val_vis = gathered_val_vis.reshape(
                    -1, gathered_val_vis.shape[-1]
                )
                gathered_hight_vis = gathered_hight_vis.reshape(
                    -1, gathered_hight_vis.shape[-1]
                )
                gathered_val_label = gathered_val_label.reshape(-1)

                if trainer.is_global_zero:  # 检查是否为 rank 0 的进程
                    if gathered_val_vis.shape[0] > 10000:
                        random_index = np.random.choice(
                            gathered_val_vis.shape[0], 10000, replace=False
                        )
                        gathered_val_vis = gathered_val_vis[random_index]
                        gathered_hight_vis = gathered_hight_vis[random_index]
                        gathered_val_label = gathered_val_label[random_index]

                        noist_test_result_dict_current = torch.stack(
                            noist_test_result_dict_current
                        )
                        noist_test_result_dict_current = (
                            noist_test_result_dict_current.permute(0, 2, 1, 3)
                        )
                        noist_test_result_dict_current = (
                            noist_test_result_dict_current.reshape(
                                -1, *noist_test_result_dict_current.shape[-2:]
                            )
                        )
                        noist_test_result_dict_current = noist_test_result_dict_current[
                            random_index
                        ]

                    if gathered_val_vis.shape[1] > 2:
                        tsne = TSNE(n_components=2, random_state=0)
                        gathered_val_vis = tsne.fit_transform(gathered_val_vis)

                    mask_ins_exp = self.ins_exp_relation_with_shap(
                        noist_test_result_dict_current
                    )
                    exp_emb_list = []
                    for i in range(mask_ins_exp.shape[1]):
                        if mask_ins_exp[:, i].sum() == 0:
                            continue
                        exp_emb = gathered_val_vis[mask_ins_exp[:, i] == 1]
                        exp_emb_list.append(exp_emb.mean(axis=0))

                    exp_emb_numpy = np.array(exp_emb_list)

                    fig_scatter = self.plot_scatter(
                        gathered_val_vis, gathered_val_label, trainer
                    )
                    fig_scatter_hyper = self.plot_scatter_hyper(
                        gathered_val_vis,
                        gathered_val_label,
                        trainer,
                        exp_emb=exp_emb_numpy,
                    )
                    
                    fig_scatter_hyper_plotly = self.plot_scatter_hyper_plotly(
                        gathered_val_vis,
                        gathered_val_label,
                        trainer,
                        exp_emb=exp_emb_numpy,                        
                    )

                    acc_mean = self.get_svc_acc(
                        gathered_val_vis, gathered_val_label, trainer
                    )
                    acc_mean_high = self.get_svc_acc(
                        gathered_hight_vis, gathered_val_label, trainer
                    )

                    trainer.logger.log_metrics(
                        {
                            val_name: acc_mean,
                            val_name + "high": acc_mean_high,
                            "epoch": trainer.current_epoch,
                            val_name + " scatter": wandb.Image(fig_scatter),
                            val_name
                            + " fig_scatter_hyper": wandb.Image(fig_scatter_hyper),
                            val_name
                            + " fig_scatter_hyper_plotly": fig_scatter_hyper_plotly
                        }
                    )
                    plt.close()

                    print(
                        "------------------------------------------------------------------"
                    )
                    print(
                        "val_index",
                        val_index,
                        "SVC_val",
                        acc_mean,
                        "SVC_val_hight",
                        acc_mean_high,
                        "epoch",
                        trainer.current_epoch,
                    )
                    print(
                        "------------------------------------------------------------------"
                    )

                    if os.path.exists(self.dirpath):
                        os.makedirs(self.dirpath, exist_ok=True)

                    if acc_mean > self.best_acc:
                        self.best_acc = acc_mean
                        torch.save(
                            pl_module.state_dict(),
                            os.path.join(
                                self.dirpath,
                                f"best_model_{self.dataset}_acc{acc_mean}.pth",
                            ),
                        )
                # if trainer.current_epoch > 400 and acc_mean < 0.8:
                #     # stop the training
                #     trainer.should_stop = True

        self.val_high = {"val1": [], "val2": [], "val3": []}
        self.val_vis = {"val1": [], "val2": [], "val3": []}
        self.val_label = {"val1": [], "val2": [], "val3": []}
        self.noist_test_result_dict = {"val1": [], "val2": [], "val3": []}
