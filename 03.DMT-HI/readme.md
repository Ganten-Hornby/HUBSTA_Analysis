## Configurating python environment

We recommend using conda for configuration. You can refer to our `install-env.sh` to configure the environment.

```bash
conda create -n nml python=3.9
conda activate nml
bash install_env.sh
```

## Run EVT

You can run EVT with a single line of code to get spatial clustering result and its latent embedding.

### Minimun replication

Running minimal replication can be done with the following command:

```bash
python main.py fit -c=conf_new/transf_cond_mnist.yaml
```

You can also use MLP version with the following command:

```bash 
python main.py fit -c=conf_new/cond_mnist.yaml
```
