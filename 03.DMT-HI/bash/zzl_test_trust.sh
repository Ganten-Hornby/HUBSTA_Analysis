# python main.py fit -c=baseline/conf_new/tmp/pacmapHCLconfig.yaml
# python main.py fit -c=baseline/conf_new/tmp/_tsneHCLconfig.yaml
# python main.py fit -c=baseline/conf_new/tmp/_umapHCLconfig.yaml

# python main.py fit -c=conf_new/nml4_zzl/HCL.yaml

# python main.py fit -c=baseline/conf_new/tmp/pacmapkmnistconfig.yaml
# python main.py fit -c=baseline/conf_new/tmp/_tsnekmnistconfig.yaml
# python main.py fit -c=baseline/conf_new/tmp/_umapkmnistconfig.yaml

# python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 1
python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 2
python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 3
python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 4
python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 5
python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 6
python main.py fit -c=conf_new/nml4_zzl/kmnist.yaml --model.num_use_moe 10
