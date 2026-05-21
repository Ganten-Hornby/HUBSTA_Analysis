import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
# from model import DMTEVT_model, MyDataModule
# from model.nnet_new_exp_mix_in_lat import DMTEVT_model
from model.nnet_new_exp_mix_in_lat_exp_all import DMTEVT_model
from data_model.datamodel import MyDataModule
from dashfile.utils import encode_image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
cached_data = None
import math

from skimage.transform import rotate

def sankey_diagram(mask_class_exp, mask_exp_fea, class_name, feature_name, exp_name, title=''):
    
    num_class = mask_class_exp.shape[0]
    num_exp = mask_class_exp.shape[1]
    num_fea = mask_exp_fea.shape[1]
    
    source = []
    target = []
    
    for i in range(mask_class_exp.shape[0]):
        for j in range(mask_class_exp.shape[1]):
            if mask_class_exp[i, j] > 0:
                source.append( i )
                target.append( j + num_class )

    for i in range(mask_exp_fea.shape[0]):
        for j in range(mask_exp_fea.shape[1]):
            if mask_exp_fea[i, j] > 0:
                source.append(i+ num_class)
                target.append(j+ num_class + num_exp)
    cLIst = 10* ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(
            color = "black", width = 0.5),
            label = class_name + exp_name + feature_name,
            color = cLIst[:num_class] + ["#d62728"] * num_exp + ["#bcbd22"] * num_fea,
            # x=[0]* num_class + [1]* num_exp + [2]* num_fea,  # 节点的横坐标
            # y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]+ [0.1*i for i in range(num_fea)]
        ),
        link = dict(
        source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = target,
        value = [1] * len(source),
    ))])

    fig.update_layout(
        title_text="Basic Sankey Diagram", font_size=20)

    # output the html file
    fig.write_html("sankey_diagram.html")
    import pdb; pdb.set_trace()    
    
    


def polar_to_euclidean(r, theta):
    """
    将极坐标 (r, theta) 转换为欧式坐标 (x, y)

    参数:
    r (array-like): 径向距离
    theta (array-like): 角度，单位为弧度

    返回:
    x (array): x 坐标
    y (array): y 坐标
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def euclidean_to_polar(x, y):
    """
    将欧式坐标 (x, y) 转换为极坐标 (r, theta)

    参数:
    x (array-like): x 坐标
    y (array-like): y 坐标

    返回:
    r (array): 径向距离
    theta (array): 角度，单位为弧度
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def interpolate_polar(exp_r, exp_theta, fea_r, fea_theta, num_points=100):
    r_interp = np.linspace(exp_r, fea_r, num_points)
    theta_interp = np.linspace(exp_theta, fea_theta, num_points)
    return r_interp, theta_interp



def rotate_point(x, y, rotation_angle, rotation_center):
    """
    Rotate a point (x, y) around a given center by a certain angle.
    """
    theta = np.radians(rotation_angle)
    cx, cy = rotation_center
    
    # Translate points to the origin
    translated_x = x - cx
    translated_y = y - cy
    
    # Apply the rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_x = translated_x * cos_theta - translated_y * sin_theta
    rotated_y = translated_x * sin_theta + translated_y * cos_theta
    
    # Translate points back
    final_x = rotated_x + cx
    final_y = rotated_y + cy
    
    return final_x, final_y

def generate_heatmap_coordinates(matrix, rotation_angle, center, heatmapsize=1.0):
    """
    Generate the coordinates of the four corners of each cell in a heatmap.
    """
    rows, cols = matrix.shape
    coordinates = []
    
    rotation_center = (heatmapsize/28/2, heatmapsize/28/2)

    for i in range(rows):
        for j in range(cols):
            # Define the four corners of the current cell
            top_left = np.array((j, i))*heatmapsize/28
            top_right = np.array((j + 1, i))*heatmapsize/28
            bottom_left = np.array((j, i + 1))*heatmapsize/28
            bottom_right = np.array((j + 1, i + 1))*heatmapsize/28
            
            
            # Rotate the corners around the rotation center
            top_left_rot = center + rotate_point(*top_left, rotation_angle, rotation_center)
            top_right_rot = center + rotate_point(*top_right, rotation_angle, rotation_center)
            bottom_left_rot = center + rotate_point(*bottom_left, rotation_angle, rotation_center)
            bottom_right_rot = center + rotate_point(*bottom_right, rotation_angle, rotation_center)
            
            # Store the rotated coordinates
            coordinates.append((top_left_rot, top_right_rot, bottom_left_rot, bottom_right_rot))
    
    return coordinates

def floattocolor(index):
    colors = ["#f7fcf5","#f6fcf4","#f6fcf4","#f5fbf3","#f5fbf2","#f4fbf2","#f4fbf1","#f3faf0","#f2faf0","#f2faef","#f1faee","#f1faee","#f0f9ed","#f0f9ec","#eff9ec","#eef9eb","#eef8ea","#edf8ea","#ecf8e9","#ecf8e8","#ebf7e7","#ebf7e7","#eaf7e6","#e9f7e5","#e9f6e4","#e8f6e4","#e7f6e3","#e7f6e2","#e6f5e1","#e5f5e1","#e4f5e0","#e4f4df","#e3f4de","#e2f4dd","#e1f4dc","#e1f3dc","#e0f3db","#dff3da","#def2d9","#ddf2d8","#ddf2d7","#dcf1d6","#dbf1d5","#daf1d4","#d9f0d3","#d8f0d2","#d7efd1","#d6efd0","#d5efcf","#d4eece","#d4eece","#d3eecd","#d2edcb","#d1edca","#d0ecc9","#cfecc8","#ceecc7","#cdebc6","#ccebc5","#cbeac4","#caeac3","#c9eac2","#c8e9c1","#c6e9c0","#c5e8bf","#c4e8be","#c3e7bd","#c2e7bc","#c1e6bb","#c0e6b9","#bfe6b8","#bee5b7","#bde5b6","#bbe4b5","#bae4b4","#b9e3b3","#b8e3b2","#b7e2b0","#b6e2af","#b5e1ae","#b3e1ad","#b2e0ac","#b1e0ab","#b0dfaa","#aedfa8","#addea7","#acdea6","#abdda5","#aadca4","#a8dca3","#a7dba2","#a6dba0","#a5da9f","#a3da9e","#a2d99d","#a1d99c","#9fd89b","#9ed799","#9dd798","#9bd697","#9ad696","#99d595","#97d494","#96d492","#95d391","#93d390","#92d28f","#91d18e","#8fd18d","#8ed08c","#8ccf8a","#8bcf89","#8ace88","#88cd87","#87cd86","#85cc85","#84cb84","#82cb83","#81ca82","#80c981","#7ec980","#7dc87f","#7bc77e","#7ac77c","#78c67b","#77c57a","#75c479","#74c478","#72c378","#71c277","#6fc276","#6ec175","#6cc074","#6bbf73","#69bf72","#68be71","#66bd70","#65bc6f","#63bc6e","#62bb6e","#60ba6d","#5eb96c","#5db86b","#5bb86a","#5ab769","#58b668","#57b568","#56b467","#54b466","#53b365","#51b264","#50b164","#4eb063","#4daf62","#4caf61","#4aae61","#49ad60","#48ac5f","#46ab5e","#45aa5d","#44a95d","#42a85c","#41a75b","#40a75a","#3fa65a","#3ea559","#3ca458","#3ba357","#3aa257","#39a156","#38a055","#379f54","#369e54","#359d53","#349c52","#339b51","#329a50","#319950","#30984f","#2f974e","#2e964d","#2d954d","#2b944c","#2a934b","#29924a","#28914a","#279049","#268f48","#258f47","#248e47","#238d46","#228c45","#218b44","#208a43","#1f8943","#1e8842","#1d8741","#1c8640","#1b8540","#1a843f","#19833e","#18823d","#17813d","#16803c","#157f3b","#147e3a","#137d3a","#127c39","#117b38","#107a37","#107937","#0f7836","#0e7735","#0d7634","#0c7534","#0b7433","#0b7332","#0a7232","#097131","#087030","#086f2f","#076e2f","#066c2e","#066b2d","#056a2d","#05692c","#04682b","#04672b","#04662a","#03642a","#036329","#026228","#026128","#026027","#025e27","#015d26","#015c25","#015b25","#015a24","#015824","#015723","#005623","#005522","#005321","#005221","#005120","#005020","#004e1f","#004d1f","#004c1e","#004a1e","#00491d","#00481d","#00471c","#00451c","#00441b"]
    
    # index = 
    print(index)
    
    return colors[index]

def create_rotated_heatmap_scatter(data, rotation_angle, center):
    """
    Create a single trace for a rotated heatmap using filled squares.
    """
    h, w = data.shape
    # rotation_center = [h // 2, w // 2]
    
    # Determine the size of each square based on the heatmap size
    # square_size = 0.01
    
    heatmap_trace_list = []
    
    coordinates = generate_heatmap_coordinates(data, rotation_angle, center, heatmapsize=0.5)
    

    for i in range(h):
        for j in range(w):
            # Center coordinates of the square
            x_1, x_2, x_3, x_4 = coordinates[i*28+j]
            # x, y = rotate_coordinates(i, j, rotation_angle, rotation_center)
            # x_centered_l = center[0] - x * heatmap_size / 1000
            # x_centered_r = center[0] + x * heatmap_size / 1000
            # y_centered_l = center[1] + y * heatmap_size / 1000
            # y_centered_r = center[1] - y * heatmap_size / 1000
            
            # step_x = (x_centered_r-x_centered_l) / h
            # step_y = (y_centered_r-y_centered_l) / w
            
            
            corners = [
                x_1, x_3, x_4, x_2,  x_1, 
            ]
            
            x_coords = [corner[0] for corner in corners]
            y_coords = [corner[1] for corner in corners]
            colors = floattocolor(int(255*data[i, j]/data.max()))
            print(x_coords)
            print(y_coords)
            print(colors)
            trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                fill='toself',
                fillcolor=colors,
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='none',
            )
            
            heatmap_trace_list.append(trace)
    # print('data.max(), data.min()', data.max(), data.min())

    # heatmap_trace = go.Scatter(
    #     x=x_coords,
    #     y=y_coords,
    #     mode='lines',
    #     fill='toself',
    #     line=dict(color='rgba(0,0,0,0)'),
    #     marker=dict(
    #         # color=colors,
    #         # colorscale='Greens',
    #         fillcolor=colors,
    #         opacity=1.0,
    #         # colorbar=dict(title='Intensity'),
    #     ),
    #     showlegend=False,
    #     hoverinfo='none',
    # )
    
    return heatmap_trace_list

def ins_exp_relation(noist_test_result_dict):
    
    noist_test_result_dict = torch.tensor(noist_test_result_dict)
    
    num_ins = noist_test_result_dict.shape[0]
    num_exp = noist_test_result_dict.shape[2]

    mean_dis = noist_test_result_dict.mean(dim=1)
    mean_dis_topk = mean_dis.topk(3, dim=1).indices

    mask_exp_ins = torch.zeros((num_ins, num_exp))
    batch_indices = torch.arange(num_ins).unsqueeze(1).expand(-1, 3)
    mask_exp_ins[batch_indices, mean_dis_topk] = 1

    return mask_exp_ins

def ins_fea_relation(noist_test_result_dict):
    
    noist_test_result_dict = torch.tensor(noist_test_result_dict)
    
    num_ins = noist_test_result_dict.shape[0]
    num_exp = noist_test_result_dict.shape[2]

    mean_dis = noist_test_result_dict.mean(dim=1)
    mean_dis_topk = mean_dis.topk(3, dim=1).indices

    mask_exp_ins = torch.zeros((num_ins, num_exp))
    batch_indices = torch.arange(num_ins).unsqueeze(1).expand(-1, 3)
    mask_exp_ins[batch_indices, mean_dis_topk] = 1

    return mask_exp_ins

def noise_map_exp(model, data, num_exp=10, noise_level=0.1):
    exp_feature_num = int(data.shape[1] // num_exp)
    emb = model.vis(data)
    distance_list = []
    for i in range(num_exp):
        start_index = i * exp_feature_num
        end_index = (i + 1) * exp_feature_num
        noise_data_delta = torch.rand_like(data) * noise_level * data.std(dim=0)
        noise_data = torch.clone(data)
        noise_data[:, start_index:end_index] += noise_data_delta[:, start_index:end_index]
        noise_emb = model.vis(noise_data)
        distance = torch.norm(noise_emb - emb, dim=1)
        distance_list.append(distance)
    distance_tensor = torch.stack(distance_list, dim=1)
    return distance_tensor


def noise_map_fea(model, data, num_exp=10, noise_level=0.1):
    num_fea = data.shape[1]
    # exp_feature_num = int(data.shape[1] // num_exp)
    emb = model.forward(data)[2]
    distance_list = []
    for i in range(num_fea):
        start_index = i
        end_index = (i + 1) 
        noise_data_delta = torch.rand_like(data) * noise_level * data.std(dim=0)
        noise_data = torch.clone(data)
        noise_data[:, start_index:end_index] += noise_data_delta[:, start_index:end_index]
        noise_emb = model.forward(noise_data)[2]
        distance = torch.norm(noise_emb - emb, dim=1)
        distance_list.append(distance)
    distance_tensor = torch.stack(distance_list, dim=1)
    return distance_tensor

def extract_embeddings_from_loader(model, data_loader, embedding_key='embeddings', labels_key='labels'):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_datas = []
    all_exp_noist_test_result_dict = []
    # all_fea_noist_test_result_dict = []
    all_xmask = []

    print('start extract_embeddings_from_loader')
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='inference: '):
            data_input_item = batch['data_input_item'].cuda()
            label = batch['label'].cuda()
            x_masked, lat_high_dim_exp, lat_vis, lat_high_dim = model(data_input_item, tau=20)
            
            exp_noist_test_result_dict = []
            # fea_noist_test_result_dict = []
            for i in range(5):
                exp_noist_test_result = noise_map_exp(model, lat_high_dim, noise_level=i * 0.1 + 0.1)
                exp_noist_test_result_dict.append(exp_noist_test_result)
                
                # fea_noist_test_result = noise_map_fea(model, data_input_item, noise_level=i * 0.1 + 0.1)
                # # fea_noist_test_result_dict.append(fea_noist_test_result)
            
            exp_noist_test_result_dict = torch.stack(exp_noist_test_result_dict).cpu()
            # # fea_noist_test_result_dict = torch.stack(fea_noist_test_result_dict).cpu()
            
            exp_noist_test_result_dict = exp_noist_test_result_dict.transpose(0, 1)
            # # fea_noist_test_result_dict = fea_noist_test_result_dict.transpose(0, 1)
            
            all_datas.append(data_input_item.cpu().numpy())
            all_embeddings.append(lat_vis.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_exp_noist_test_result_dict.append(exp_noist_test_result_dict.cpu().numpy())
            # all_fea_noist_test_result_dict.append(fea_noist_test_result_dict.cpu().numpy())
            all_xmask.append(x_masked.cpu().numpy())
    
    
    all_datas = np.vstack(all_datas)
    all_embeddings = np.vstack(all_embeddings)
    all_exp_noist_test_result_dict = np.vstack(all_exp_noist_test_result_dict)
    all_xmask = np.vstack(all_xmask)
    all_labels = np.hstack(all_labels)

    return all_datas, all_embeddings, all_exp_noist_test_result_dict, all_xmask, all_labels

def load_data_visemb_from_lightning_ckpt(ckpt_path, dataset, T_num_layers=4, sample_rate_feature=0.2, num_input_dim=784, embedding_key='embeddings', labels_key='labels', sample_num=2000):
    global cached_data
    print(cached_data)
    if cached_data is None:
        model = DMTEVT_model(
            num_input_dim=num_input_dim,
            lr=0.001,
            nu_lat=0.1,
            nu_emb=0.1,
            exaggeration_lat=3,
            exaggeration_emb=3,
            sample_rate_feature=sample_rate_feature,
            T_hidden_size=512,
            weight_decay=0.000000001,
            num_use_moe=10,
            T_num_layers=T_num_layers,
            max_epochs=2000,
            test_noise=True,
        )
        model.load_state_dict(torch.load(ckpt_path))
        model = model.cuda()
        data_model = MyDataModule(
            data_name=dataset,
            pca_dim=64,
            batch_size=1000,
            num_workers=1,
            K=5,
            n_f_per_cluster=3,
            l_token=10,
            data_path='/zangzelin/data'
        )
        data_model.setup(stage='')
        data_loader = DataLoader(
            data_model.data_train,
            drop_last=False,
            shuffle=False,
            batch_size=1000,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
        )
        datas, embeddings, all_noist_test_result_dict, xmask, labels = extract_embeddings_from_loader(model, data_loader)
        # if sample_num is not None and sample_num < embeddings.shape[0]:
        #     indices = np.random.choice(embeddings.shape[0], sample_num, replace=False)
        #     datas = datas[indices]
        #     embeddings = embeddings[indices]
        #     labels = labels[indices]
        #     xmask = xmask[indices]
        #     all_noist_test_result_dict = all_noist_test_result_dict[indices]
    
        ins_exp_mask = ins_exp_relation(all_noist_test_result_dict)
        # ins_exp_mask = ins_fea_relation(all_noist_test_result_dict)
        
        xmask_list = []
        exp_emb_list = []
        for i in range(ins_exp_mask.shape[1]):
            # if ins_exp_mask[:, i].sum() == 0:
            #     continue
            exp_emb = embeddings[ins_exp_mask[:, i] == 1]
            exp_emb_list.append(exp_emb.mean(axis=0))
            xmask_list.append(xmask[:, i, :])
    
        exp_emb_list = np.vstack(exp_emb_list)
        xmask = np.stack(xmask_list, axis=1)
        fea_ins_weight = model.exp.weight.data
        fea_ins_mask = torch.zeros_like(fea_ins_weight, device=fea_ins_weight.device)
        top_k = 100
        _, top_indices = torch.topk(fea_ins_weight, top_k, dim=1)
        fea_ins_mask.scatter_(1, top_indices, 1)
        cached_data = (datas, embeddings, exp_emb_list, ins_exp_mask, fea_ins_mask, xmask, labels)
    return cached_data

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

def create_scatter_trace(gathered_val_vis_hy, gathered_val_label, encoded_images):
    scatter = go.Scatter(
        x=gathered_val_vis_hy[:, 0],
        y=gathered_val_vis_hy[:, 1],
        mode='markers',
        marker=dict(
            color=gathered_val_label,
            colorscale='Rainbow',
            size=2
        ),
        customdata=encoded_images,
        name='scatter',
        hoverinfo='text',
        text=[f"Label: {label}" for label in gathered_val_label]
    )
    return scatter

def create_exp_scatter_trace(exp_emb_hy, xmask_average, encoded_images_exp):
    exp_scatter = go.Scatter(
        x=exp_emb_hy[:, 0],
        y=exp_emb_hy[:, 1],
        mode='markers',
        marker=dict(
            color='red',
            size=10
        ),
        customdata=encoded_images_exp,
        name='exp_scatter',
        hoverinfo='text',
        text=["Exp"] * exp_emb_hy.shape[0]
    )
    return exp_scatter

def create_circle_trace(R, color='black', width=2, dash='solid'):
    circle = go.Scatter(
        x=[R * np.cos(theta) for theta in np.linspace(0, 2 * np.pi, 100)],
        y=[R * np.sin(theta) for theta in np.linspace(0, 2 * np.pi, 100)],
        mode='lines',
        line=dict(color=color, width=width, dash=dash)
    )
    return circle

def create_fea_scatter_trace(fea_emb_dict):

    x = [value[0] for value in fea_emb_dict.values()]
    y = [value[1] for value in fea_emb_dict.values()]
    text_list = list(fea_emb_dict.keys())

    fea_scatter = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            color='blue',
            size=5
        ),
        name='fea_scatter',
        hoverinfo='text',
        text=text_list
    )
    return fea_scatter

def create_heatmap_scatter_traces(exp_emb_hy, xmask_average):
    traces = []
    for i in range(exp_emb_hy.shape[0]):
        heat_data = xmask_average[i].reshape(28, 28)
        center = exp_emb_hy[i] * 1.3
        # angle = math.degrees(math.atan2(center[1], center[0]))
        angle = 0 # math.degrees(math.atan2(center[1], center[0]))
        heatmap_traces = create_rotated_heatmap_scatter(heat_data, angle, center)
        # traces.append(*heatmap_traces)
        traces += heatmap_traces
    return traces

def calculate_hyperbolic_embeddings(gathered_val_vis, R):
    u_mean = gathered_val_vis.mean(axis=0)
    u_std = gathered_val_vis.std(axis=0)
    gathered_val_vis_hy = euclidean_to_hyperbolic_matrix(gathered_val_vis, R=R)
    return gathered_val_vis_hy, u_mean, u_std

def create_scatter_fig(gathered_val_vis_hy, gathered_val_label, encoded_images):
    scatter_fig = go.Figure()
    scatter = create_scatter_trace(gathered_val_vis_hy, gathered_val_label, encoded_images)
    scatter_fig.add_trace(scatter)
    return scatter_fig

def calculate_exp_emb_hyperbolic(exp_emb, u_mean, u_std, R_emb):
    exp_emb_hy = (exp_emb - u_mean) / u_std
    exp_emb_hy = R_emb * exp_emb_hy / np.linalg.norm(exp_emb_hy, axis=1, keepdims=True)
    return exp_emb_hy

def create_exp_scatter_fig(exp_emb_hy, xmask, xmask_all, R_emb, scatter_fig):

    xmask_average = xmask.mean(axis=0)
    xmask_average = xmask_average / xmask_average.max()
    encoded_images_exp = [encode_image(xmask_average[i], i=i) for i in range(len(xmask_average))]

    exp_scatter = create_exp_scatter_trace(exp_emb_hy, xmask_average, encoded_images_exp)
    scatter_fig.add_trace(exp_scatter)

    circle = create_circle_trace(R_emb, width=2)
    scatter_fig.add_trace(circle)

    return scatter_fig, encoded_images_exp

def calculate_fea_emb(fea_ins_mask, exp_emb_hy, R_fea):
    fea_emb_dict = {}
    fea_emb_list = []
    for i in range(fea_ins_mask.shape[1]):
        if fea_ins_mask[:, i].sum() > 0:
            mask_fea_ins = fea_ins_mask[:, i].detach().cpu().numpy().astype(np.bool_)
            fea_emb = exp_emb_hy[mask_fea_ins].mean(axis=0)
            fea_emb_dict[f'fea_{i}'] = R_fea * fea_emb / np.linalg.norm(fea_emb)
            fea_emb_list.append(R_fea * fea_emb / np.linalg.norm(fea_emb))
        else:
            fea_emb_list.append([0, 0])
    return fea_emb_dict, fea_emb_list

def add_fea_scatter_to_fig(fea_emb_dict, scatter_fig):
    fea_scatter = create_fea_scatter_trace(fea_emb_dict)
    scatter_fig.add_trace(fea_scatter)
    return scatter_fig

def add_interpolated_lines(fea_ins_mask, exp_emb_hy, fea_emb_list, scatter_fig):
    x = []
    y = []
    for i in range(fea_ins_mask.shape[0]):
        for j in range(fea_ins_mask.shape[1]):
            if fea_ins_mask[i][j] > 0.5:
                exp_r, exp_t = euclidean_to_polar(exp_emb_hy[i][0], exp_emb_hy[i][1])
                fea_r, fea_t = euclidean_to_polar(fea_emb_list[j][0], fea_emb_list[j][1])

                r_interp, theta_interp = interpolate_polar(exp_r, exp_t, fea_r, fea_t)

                x_interp, y_interp = polar_to_euclidean(r_interp, theta_interp)

                x.extend(x_interp.tolist() + [None])
                y.extend(y_interp.tolist() + [None])

    scatter_fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode='lines',
        line=dict(color='grey', width=0.5),
    ))        
    return scatter_fig

def add_heatmap_traces(exp_emb_hy, xmask_average, scatter_fig):
    heatmap_traces = create_heatmap_scatter_traces(exp_emb_hy, xmask_average)
    for trace in heatmap_traces:
        scatter_fig.add_trace(trace)
    return scatter_fig

def create_initial_heatmaps(encoded_images, encoded_images_exp):
    initial_heatmap_data = encoded_images[0]
    heatmap_fig = go.Figure(data=go.Heatmap(z=initial_heatmap_data, colorscale='Greys'))
    initial_heatmap_exp = encoded_images_exp[0]
    secondary_heatmap_fig = go.Figure(data=go.Heatmap(z=initial_heatmap_exp, colorscale='Viridis'))
    return heatmap_fig, secondary_heatmap_fig

def plot_scatter_hyper(gathered_datas_all, gathered_val_vis, gathered_val_label, R=1.5, exp_emb=None, image_files=None, xmask_all=None, fea_ins_mask=None):
    gathered_val_vis_hy, u_mean, u_std = calculate_hyperbolic_embeddings(gathered_val_vis, R)

    if gathered_datas_all.shape[1] != 784:
        gathered_datas = gathered_datas_all[:, :784]
        xmask = xmask_all[:, :, :784]
    else:
        xmask = xmask_all
        
    # select high variance features
    import sklearn as sk
    import scanpy as sc
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    selector.fit(gathered_datas_all)
    variances = selector.variances_

    top_300_indices = np.argsort(variances)[-300:]
    
    data_path = '/zangzelin/data'
    sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
    feature_names_all = list(sadata.var.index)
    fea_name_top_300 = [feature_names_all[i] for i in top_300_indices]
    tissue_name = list(sadata.obs.tissue)
    
    import pickle
    HCL_set_list_label = pickle.load(open('HCL_set_list_label.pkl', 'rb'))
    
    
    # import pdb; pdb.set_trace()
    # Step 4: 使用这些索引选择相应的特征
    # gathered_datas_top_300 = gathered_datas_all[:, top_300_indices]
    
    xmask_all_mean = xmask_all.mean(axis=0)
    xmask_all_mean = xmask_all_mean[:, top_300_indices]
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(40, 10))
    sns.heatmap(
        xmask_all_mean, 
        cmap='Blues', 
        xticklabels=fea_name_top_300, 
        yticklabels=['exp{}'.format(i) for i in range(xmask_all_mean.shape[0])]
        )
    plt.savefig('xmask_exp_feature.png')
    
    class_exp_mask = []
    HCL_set_list_label_not_none = []
    for i, label in enumerate(HCL_set_list_label):
        # item = xmask_all[gathered_val_label==i].mean(axis=(0, 2))
        # import pdb; pdb.set_trace()
        if (gathered_val_label==i).sum() > 0:
            print(i)
            item = xmask_all[gathered_val_label==i].mean(axis=(0, 2))
            class_exp_mask.append(item)
            HCL_set_list_label_not_none.append(label)
    class_exp_mask = np.stack(class_exp_mask)
    
    plt.figure(figsize=(40, 10))
    sns.heatmap(
        class_exp_mask, 
        cmap='Reds', 
        xticklabels=['exp{}'.format(i) for i in range(xmask_all_mean.shape[0])], 
        yticklabels=HCL_set_list_label_not_none,
        )
    plt.savefig('xmask_class_exp.png')
    
    
    
    top_5_indices = np.argsort(xmask_all_mean, axis=1)[:, -5:]
    xmask_all_mean_bool = np.zeros_like(xmask_all_mean, dtype=bool)
    np.put_along_axis(xmask_all_mean_bool, top_5_indices, True, axis=1)
    
    class_exp_mask_T = class_exp_mask.T
    
    top_5_indices_T = np.argsort(class_exp_mask_T, axis=1)[:, -3:]
    class_exp_mask_bool_T = np.zeros_like(class_exp_mask_T, dtype=bool)
    np.put_along_axis(class_exp_mask_bool_T, top_5_indices_T, True, axis=1)
    class_exp_mask_bool = class_exp_mask_bool_T.T
    
    sankey_diagram(
        class_exp_mask_bool,
        xmask_all_mean_bool,
        HCL_set_list_label_not_none,
        fea_name_top_300,
        ['Exp {}'.format(i) for i in range(xmask_all_mean.shape[0])],
        'class_exp_mask_bool.html',
        )
    
    # import pdb; pdb.set_trace()
    encoded_images = [encode_image(gathered_datas[i], i=i) for i in range(len(gathered_val_vis))]
    scatter_fig = create_scatter_fig(gathered_val_vis_hy, gathered_val_label, encoded_images)

    if exp_emb is not None:
        R_emb = 2.3
        exp_emb_hy = calculate_exp_emb_hyperbolic(exp_emb, u_mean, u_std, R_emb)
        scatter_fig, encoded_images_exp = create_exp_scatter_fig(exp_emb_hy, xmask, xmask_all, R_emb, scatter_fig)

        R_fea = 1.6
        fea_emb_dict, fea_emb_list = calculate_fea_emb(fea_ins_mask, exp_emb_hy, R_fea)
        scatter_fig_ = add_fea_scatter_to_fig(fea_emb_dict, scatter_fig)

        scatter_fig = add_interpolated_lines(fea_ins_mask, exp_emb_hy, fea_emb_list, scatter_fig)
        scatter_fig = add_heatmap_traces(exp_emb_hy, xmask.mean(axis=0), scatter_fig)

        circle_r1 = create_circle_trace(R_fea, width=1)
        scatter_fig.add_trace(circle_r1)
        

    circle = create_circle_trace(R, dash='dash')
    scatter_fig.add_trace(circle)

    scatter_fig.update_layout(
        xaxis=dict(visible=False, range=[-3, 3]),
        yaxis=dict(visible=False, range=[-3, 3]),
        showlegend=False,
        width=1000,
        height=1000,
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
    )

    heatmap_fig, secondary_heatmap_fig = create_initial_heatmaps(encoded_images, encoded_images_exp if exp_emb is not None else [])

    return scatter_fig, heatmap_fig, secondary_heatmap_fig, encoded_images, encoded_images_exp if exp_emb is not None else []