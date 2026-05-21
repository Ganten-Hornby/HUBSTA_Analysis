import os
import socket
import wandb
import yaml
import subprocess
import time

# 查找一个随机可用端口
def get_random_open_port():
    """获取一个随机可用的端口"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))  # 让系统分配一个可用端口
    port = s.getsockname()[1]
    s.close()
    return port

datasets = ["SAM", "mnist"]
sweep_path = "/root/yuhao/gitlab/dmt_nml/sweep/nml5_val"

# 运行 WandB agent
def run_wandb_agent(sweep_id, gpus="0,1,2,3,4,5,6,7", cpus_per_program=10):
    """启动多个 WandB agent，并为每个 agent 分配一个随机端口"""
    gpu_list = gpus.split(',')
    cpu_start = 0
    processes = []

    for gpu_id in gpu_list:
        print(f"Starting wandb agent {sweep_id} on GPU {gpu_id} with {cpus_per_program} CPUs")

        # 查找一个随机端口，并将其设置到 WANDB_SERVER_PORT 环境变量中
        random_port = get_random_open_port()
        os.environ["WANDB_SERVER_PORT"] = str(random_port)  # 为当前进程设置随机端口
        print(f"Assigned random port {random_port} for GPU {gpu_id}")

        # 计算 CPU 范围
        cpu_end = cpu_start + cpus_per_program - 1
        cpu_range = f"{cpu_start}-{cpu_end}"

        # 使用 taskset 设置 CPU 亲和性，使用指定的 GPU 运行 WandB agent
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(["taskset", "-c", cpu_range, "wandb", "agent", sweep_id], env=env)
        processes.append(process)
        cpu_start += cpus_per_program

        # 每次启动后休眠 30 秒，避免过载
        time.sleep(30)

    monitor_processes(processes)

# 监控 WandB agent 进程
def monitor_processes(processes):
    """监控所有启动的 WandB agent 进程"""
    while processes:
        for process in processes[:]:
            retcode = process.poll()
            if retcode is not None:
                print(f"Process {process.pid} finished with exit code {retcode}")
                processes.remove(process)
        time.sleep(5)

# WandB 登录
def wandb_login():
    # 执行 WandB 登录命令
    api_key = "your_api_key"  # 替换为你的 WandB API key
    subprocess.run(["wandb", "login", "--host=https://api.wandb.ai", api_key])

# 创建 WandB sweep 并启动 agent
def create_sweep_and_run():
    # 登录 WandB
    wandb_login()

    for dataset in datasets:
        # 读取 YAML 文件，加载 sweep 配置
        with open(os.path.join(sweep_path, f"{dataset}.yaml"), "r") as file:
            sweep_config = yaml.safe_load(file)
            sweep_config["name"] = "sweep_" + dataset + "_repeat3"
            sweep_id = wandb.sweep(sweep_config)  # 创建 sweep 并获取 sweep_id

            print(f"Generated sweep ID: {sweep_id}")
            run_wandb_agent(sweep_id)

            # 等待 60 秒后再启动下一个 sweep
            print("Waiting for 60 seconds before running the next command...")
            time.sleep(60)

if __name__ == "__main__":
    create_sweep_and_run()
