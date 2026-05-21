import wandb
import yaml

datasets=["gast"]
methods=["tsne"]

for dataset in datasets:
    for method in methods:
        with open("baseline/sweep/"+method+".yaml", "r") as file:
            sweep_config = yaml.safe_load(file)
        print(sweep_config)
        files = ['baseline/conf_new/'+dataset+'.yaml','baseline/conf_new/' + method + '.yaml']
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
        sweep_config["name"]=method+dataset
        sweep_id = wandb.sweep(sweep_config)
        wandb.agent(sweep_id, count=30)
