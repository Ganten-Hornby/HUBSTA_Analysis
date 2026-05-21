import wandb
import yaml

# datasets=["ACT","coil20","gast","SAM","mnist","kmnist","HCL","emnist","MCA","coil100","cifar100"]
# methods=["ivis","_tsne","pumap","pacmap","_umap","evnet"]

datasets=["cifar10","NG20"]
methods=["hnne","evnet"]

for dataset in datasets:
    for method in methods:
        with open("baseline/sweep/"+method+".yaml", "r") as file:
            sweep_config = yaml.safe_load(file)
        if method == "pumap" and dataset in ["coil100","MCA","cifar10","cifar100"]:
            continue
        print(sweep_config)
        files = ['baseline/conf_new/'+dataset+'_tri.yaml','baseline/conf_new/' + method + '.yaml']
        merged_config = {}
        
        # 读取并合并配置文件
        for file in files:
            with open(file, 'r') as f:
                config = yaml.safe_load(f)
                merged_config.update(config)

        # 将合并后的配置写入临时文件
        with open('baseline/conf_new/tmp/'+method+dataset+'config.yaml', 'w') as f:
            yaml.safe_dump(merged_config, f)
        sweep_config["parameters"]["config"]["values"]=['baseline/conf_new/tmp/'+method+dataset+'config.yaml']
        sweep_config["name"]=method+dataset+"_repeat10"
        with open(f'/zangzelin/yuhao/gitlab/dmt_nml/baseline/other/{method}{dataset}.yaml', 'w') as f:
            yaml.safe_dump(sweep_config, f)
