import os.path as osp
from datetime import datetime
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import KnowledgeGraph
from model import FusionModel


class LinkingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("data.num_relations", "model.num_relations", apply_on="instantiate")
        parser.link_arguments("data.num_nodes", "model.num_nodes", apply_on="instantiate")
        parser.link_arguments("trainer.max_epochs", "model.max_epochs", apply_on="parse")

    def before_instantiate_classes(self):
        dataset = osp.splitext(osp.basename(self.config.config[0].relative))[0]
        self.config.trainer.logger.init_args.name = self.config.data.dataset = dataset
        self.config.trainer.profiler.init_args.filename = dataset


if __name__ == "__main__":
    model_name = "exp_LaGR"
    cli = LinkingCLI(
        FusionModel,
        KnowledgeGraph,
        run=False,
        seed_everything_default=42,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "ddp",
            "precision": 32,
            "max_epochs": 20,
            "deterministic": True,
            "num_sanity_val_steps": 1,
            "logger": {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {
                    "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
                    "save_dir": model_name
                }
            },
            "profiler": {
                "class_path": "lightning.pytorch.profilers.SimpleProfiler",
                "init_args": {
                    "dirpath": model_name
                }
            },
            "callbacks": [
                ModelCheckpoint(every_n_epochs=1, monitor=FusionModel.MAIN_METRIC, mode="max"),
            ],
        },
        save_config_kwargs={"overwrite": True},
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
