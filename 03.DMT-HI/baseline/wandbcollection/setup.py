import wandb
import pandas as pd
import os
# 初始化 API
api = wandb.Api()

# 设置实体和项目
entity = "wangyuhao"  # 替换为你的WandB实体名
project = "DMTNML"  # 替换为你的WandB项目名

# 获取项目中的所有sweep（通过获取项目中的运行并过滤出sweep）
runs = api.runs(path=f"{entity}/{project}")

# 过滤出与sweep相关的runs，并提取唯一的sweep_id
sweep_ids = set(run.sweep.id for run in runs if run.sweep)

# 列出所有sweep的ID
targets=["fct","fknn","svc","svc_rbf","trust120"]
configs=["p1","p2"]
datasets=["coil20","coil100","mnist","kmnist","emnist","cifar10","cifar100","ACT","gast","SAM","HCL","MCA","NG20"]
methods=["_tsne","_umap","pumap","ivis","pacmap","hnne","evnet","sweep"]
results_train = {
    target: {dataset: {method: None for method in methods} for dataset in datasets}
    for target in targets
}
results_conf = {
    config: {dataset: {method: None for method in methods} for dataset in datasets}
    for config in configs
}
results_test = {
    target: {dataset: {method: None for method in methods} for dataset in datasets}
    for target in targets
}
image_dir = "downloaded_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
for sweep_id in sweep_ids:
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    for dataset in datasets:
        for method in methods:
            sweep_name=method+dataset+"_repeat7"
            if sweep.name == sweep_name or sweep.name == method+"_"+dataset+"_repeat7":
                print(sweep_name)
                sweep_runs = sweep.runs
                for target in targets:
                    if target not in ["fct","fknn","svc","svc_rbf"]:
                        max_train_target = None
                        max_val_target = None
                        max_test_target = None
                        for run in sweep_runs:
                            train_target = run.summary.get(f'train_{target}', None)
                            if train_target is not None:
                                if max_train_target is None or train_target < max_train_target:
                                    max_train_target = train_target
                            
                            # 存储最大值在结果字典中
                            if max_train_target is not None:
                                results_train[target][dataset][method] = max_train_target

                        for run in sweep_runs:
                            val_target = run.summary.get(f'validation_{target}', None)
                            test_target = run.summary.get(f'test_{target}', None)
                            if test_target is not None:
                                if max_val_target is None or val_target < max_val_target:
                                    max_val_target = val_target
                                    max_test_target = test_target
                            
                            # 存储最大值在结果字典中
                                if max_test_target is not None:
                                    results_test[target][dataset][method] = max_test_target
                            
                    else:
                        max_train_target = None
                        max_val_target = None
                        max_test_target = None
                        best_run = None
                        for run in sweep_runs:
                            train_target = run.summary.get(f'train_{target}', None)
                            if train_target is not None:
                                if max_train_target is None or train_target > max_train_target:
                                    max_train_target = train_target
                                    best_run = run
                            
                            # 存储最大值在结果字典中
                            if max_train_target is not None:
                                results_train[target][dataset][method] = max_train_target*100
                        
                        best_run_test = None

                        for run in sweep_runs:
                            val_target = run.summary.get(f'validation_{target}', None)
                            test_target = run.summary.get(f'test_{target}', None)
                            if test_target is not None:
                                if max_val_target is None or val_target > max_val_target:
                                    max_val_target = val_target
                                    max_test_target = test_target
                                    best_run_test = run
                            
                            # 存储最大值在结果字典中
                            if max_test_target is not None:
                                results_test[target][dataset][method] = max_test_target*100
                        
                        best_run_test=best_run

                        if best_run is not None and target == 'svc':
                            for config in configs:
                                results_conf[config][dataset][method] = best_run.config.get(config, None)
                            print(f"Checking images from run: {best_run.id} with max SVC: {max_train_target}")
                            files = best_run.files()
                            for file in files:
                                if "train_scatter" in file.name and file.name.endswith((".png", ".jpg", ".jpeg")):
                                    file_path_obj = file.download(root=image_dir,replace=True)  # 下载文件并获取文件对象
                                    file_path = file_path_obj.name  # 从文件对象获取路径
                                    
                                    # 因为文件可能保存在子目录中，拼接完整路径来处理重命名
                                    full_file_path = os.path.join(image_dir, file_path)
                                    new_path = os.path.join(image_dir, f"{sweep_name}.png")
                                    print(full_file_path)
                                    print(new_path)
                                    os.rename(file_path, new_path)  # 重命名文件
                                    print(f"Saved image as {new_path}")
                                    break  # 只下载一张符合条件的图片
                            for file in files:
                                if "train_fig_scatter_hyper" in file.name and file.name.endswith((".png", ".jpg", ".jpeg")):
                                    file_path_obj = file.download(root=image_dir,replace=True)  # 下载文件并获取文件对象
                                    file_path = file_path_obj.name  # 从文件对象获取路径
                                    
                                    # 因为文件可能保存在子目录中，拼接完整路径来处理重命名
                                    full_file_path = os.path.join(image_dir, file_path)
                                    new_path = os.path.join(image_dir, f"hyper_{sweep_name}.png")
                                    print(full_file_path)
                                    print(new_path)
                                    os.rename(file_path, new_path)  # 重命名文件
                                    print(f"Saved image as {new_path}")
                                    break  # 只下载一张符合条件的图片
                        if best_run_test is not None and target == 'svc':
                            print(f"Checking images from run: {best_run_test.id} with max SVC: {max_train_target}")
                            files = best_run_test.files()
                            for file in files:
                                if "test_scatter" in file.name and file.name.endswith((".png", ".jpg", ".jpeg")):
                                    file_path_obj = file.download(root=image_dir,replace=True)  # 下载文件并获取文件对象
                                    file_path = file_path_obj.name  # 从文件对象获取路径
                                    
                                    # 因为文件可能保存在子目录中，拼接完整路径来处理重命名
                                    full_file_path = os.path.join(image_dir, file_path)
                                    new_path = os.path.join(image_dir, f"test_{sweep_name}.png")
                                    print(full_file_path)
                                    print(new_path)
                                    os.rename(file_path, new_path)  # 重命名文件
                                    print(f"Saved image as {new_path}")
                                    break  # 只下载一张符合条件的图片
                            for file in files:
                                if "test_fig_scatter_hyper" in file.name and file.name.endswith((".png", ".jpg", ".jpeg")):
                                    file_path_obj = file.download(root=image_dir,replace=True)  # 下载文件并获取文件对象
                                    file_path = file_path_obj.name  # 从文件对象获取路径
                                    
                                    # 因为文件可能保存在子目录中，拼接完整路径来处理重命名
                                    full_file_path = os.path.join(image_dir, file_path)
                                    new_path = os.path.join(image_dir, f"test_hyper_{sweep_name}.png")
                                    print(full_file_path)
                                    print(new_path)
                                    os.rename(file_path, new_path)  # 重命名文件
                                    print(f"Saved image as {new_path}")
                                    break  # 只下载一张符合条件的图片

for target in targets:
    df = pd.DataFrame(results_train[target]).T  # 转置使datasets为行，methods为列
    
    # 保存到Excel文件，并确保行列名称被保存
    file_name = f"{target}_train_max_values.xlsx"
    df.to_excel(file_name, index=True, header=True)
    print(f"Saved {file_name}")

for target in targets:
    df = pd.DataFrame(results_test[target]).T  # 转置使datasets为行，methods为列
    
    # 保存到Excel文件，并确保行列名称被保存
    file_name = f"{target}_test_max_values.xlsx"
    df.to_excel(file_name, index=True, header=True)
    print(f"Saved {file_name}")

for config in configs:
    df = pd.DataFrame(results_conf[config]).T  # 转置使datasets为行，methods为列
    
    # 保存到Excel文件，并确保行列名称被保存
    file_name = f"{config}.xlsx"
    df.to_excel(file_name, index=True, header=True)
    print(f"Saved {file_name}")