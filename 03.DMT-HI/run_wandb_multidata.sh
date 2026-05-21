#!/bin/bash
wandb login --host=https://api.wandb.ai 1bfeca96066bdd13c317111756395d792b64ce56

# 定义你的 yaml 文件所在的目录
YAML_DIR="baseline/other"  # 将这里替换为你的 yaml 文件夹路径

# 遍历目录下的所有 yaml 文件
for yaml_file in "$YAML_DIR"/*.yaml; do
    # 获取 sweep ID
    # sweep_id=$(wandb sweep "$yaml_file" | grep -oE "wandb [^ ]+" | awk '{print $2}')
    # sweep_id=$(wandb sweep "$yaml_file" | grep -oP "Run sweep agent with: wandb agent \K[^ ]+")
    # sweep_id=$(wandb sweep "$yaml_file" | grep -o "wandb agent [^ ]*" | awk '{print $3}')
    # sweep_id=$(wandb sweep "$yaml_file" | grep -oP "wandb agent \K.*")
    # sweep_id=$(wandb sweep "$yaml_file" | sed -n 's/.*wandb agent //p')
    # output=$(wandb sweep sweep/nml5_val/ACT.yaml 2>&1)
    output=$(wandb sweep "$yaml_file" 2>&1)
    sweep_id=$(echo "$output" | grep -oP '(?<=wandb agent ).*')


    # 启动 wandb agent
    # wandb agent "$sweep_id" &
    echo '=================='
    echo "$sweep_id"

    # bash "run_wandb.sh $sweep_id 0,1,2,3 4"
    bash run_wandb.sh "$sweep_id" 0,1,2,3,4,5,6,7 12
    # echo '=================='

    # sleep 2m
    echo '=================='
    echo '=================='
    response=$(curl -s -H "Authorization: Bearer $API_KEY" \
  "https://api.wandb.ai/sweeps/$sweep_id")
    echo "$sweep_id"
    echo '=================='
    echo '=================='

    
    # 让 wandb agent 以后台进程的形式运行
    echo "Started wandb agent for sweep $sweep_id"
done
