import os
import numpy as np
from lightning import LightningModule
from lightning import LightningDataModule

from lightning.pytorch.cli import LightningCLI
import torch
torch.set_float32_matmul_precision('medium')

# only use one cpu
# torch.set_num_threads(1)

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--outputdir", default="outputdir")
        parser.link_arguments("data.init_args.num_positive_samples", "model.init_args.num_positive_samples")
        parser.link_arguments("data.init_args.data_name", "model.init_args.data_name")
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")


def main():
    config_path = os.path.join(
            os.path.dirname(__file__),
            "conf_new/huada_WS/HD_WS_TEST.yaml")
    print(config_path)
    cli = MyLightningCLI(
        LightningModule,
        LightningDataModule,
        save_config_callback=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={'default_config_files': [config_path]},
        run=False
        )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    output = cli.trainer.predict(cli.model, datamodule=cli.datamodule)
    high_dim = np.squeeze(np.concatenate([i[0] for i in output]))
    vis = np.concatenate([i[1] for i in output])
    print(f"high_dim shape:{high_dim.shape}")
    print(f"vis shape:{vis.shape}")
    os.makedirs(cli.config["outputdir"], exist_ok=True)
    np.save(os.path.join(cli.config["outputdir"], "X_dmt_highdim.npy"), high_dim)
    np.save(os.path.join(cli.config["outputdir"], "X_dmt.npy"), vis)


if __name__ == '__main__':
    main()