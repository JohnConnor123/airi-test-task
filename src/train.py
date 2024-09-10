# flake8: noqa: F401
import os
import pickle
import sys
import warnings
from typing import Any

import hydra
import lightning as L
import mlflow
import pandas as pd
import torch
from hydra.utils import instantiate
from matplotlib import rcParams
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tensorboard import program
from torch.utils.data import DataLoader, TensorDataset

from lightning_model import MyLightningModel


sys.path[-1] = os.path.join(os.path.dirname(__file__))
from preprocessing import do_pipeline


current_dir = os.getcwd()
os.chdir(os.path.dirname(os.sep.join(__file__.split(os.sep)[:-1])))
cfg = OmegaConf.load("src/config/config.yaml")
os.chdir(current_dir)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def load_obj(cfg: DictConfig, **kwargs: dict) -> Any:
    """
    Extract an object from a given config
        Args:
            cfg: DictConfig to be transformed in object.
            **kwargs: Additional params.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("name", None)
    cfg = OmegaConf.create(cfg)
    print(cfg)
    return instantiate(cfg, **kwargs)


def init_tensorboard():
    tracking_address = "src\\logs\\tb_logs"  # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def get_dataloaders_from_csv_file(cfg, filename):
    X = pd.read_csv(filename)
    y = X.pop("ddg")
    X_train, X_test, y_train, y_test = train_test_split(
        torch.Tensor(X.to_numpy()),
        torch.Tensor(y.to_numpy()).unsqueeze(1),
        test_size=0.2,
        random_state=42,
    )
    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=cfg.data_preparation.batch_size
    )
    val_dataloader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=cfg.data_preparation.batch_size
    )
    return train_dataloader, val_dataloader


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    example_input_array = torch.ones((2, cfg.general.input_size))

    # train_dataloader, val_dataloader, test_dataloader = do_pipeline()
    with open(f"./{cfg.general.data_dir}/secondary/dataloaders/dataloaders.pkl", "rb") as file:
        train_dataloader, val_dataloader, test_dataloader = pickle.load(file)

    callbacks = [load_obj(callback) for callback in cfg.callbacks.values()]
    loggers = [load_obj(logger) for logger in cfg.logging.values()]

    model = load_obj(cfg.model)
    loss_fn = load_obj(cfg.loss)
    optimizer = load_obj(cfg.optimizer, params=model.parameters())

    """
    scheduler_kwargs = {
        "exponential": {"gamma": random.choice([0.85, 0.9, 0.93, 0.95])},
        "plateau": {
            "factor": random.choice([0.1, 0.2, 0.5, 0.8]),
            "patience": random.choice([5, 10, 14, 17, 20]),
        },
    }
    """

    lr_scheduler = load_obj(
        cfg.scheduler,
        optimizer=optimizer,  # **scheduler_kwargs[cfg.scheduler.name]
    )

    model = MyLightningModel(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        example_input_array=example_input_array,
    )

    trainer = L.Trainer(
        **cfg.trainer,
        logger=loggers,
        callbacks=callbacks,
    )

    mlflow.pytorch.autolog()
    if cfg.general.init_tensorboard_server:
        init_tensorboard()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    if cfg.mlflow.enable_system_metrics_logging:
        mlflow.enable_system_metrics_logging()

    if not mlflow.get_experiment_by_name(cfg.mlflow.experiment_name):
        mlflow.create_experiment(cfg.mlflow.experiment_name)

    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.set_experiment_tags(cfg.mlflow.run_tags)

    with mlflow.start_run(
        run_name=cfg.mlflow.run_name, description=cfg.mlflow.run_description
    ):
        trainer.fit(
            model=model.cuda(),
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        signature = mlflow.models.infer_signature(
            model_input=example_input_array.numpy(),
            model_output=model(example_input_array).detach().numpy(),
        )

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            input_example=example_input_array.numpy(),
            signature=signature,
            # registered_model_name=cfg.model.name,
        )

        hydra_run_paths = [
            "src/logs/hydra-multiruns/" + name
            for name in os.listdir("src/logs/hydra-multiruns/")
        ]
        hydra_full_path = max(hydra_run_paths, key=os.path.getmtime)
        mlflow.log_artifact(hydra_full_path, artifact_path="src/hydra")

        cfg = OmegaConf.to_container(cfg, resolve=True)
        # cfg["scheduler"].update(scheduler_kwargs[cfg["scheduler"]["name"]])
        mlflow.log_params(cfg)
        mlflow.log_artifact("src/config", artifact_path="src")

        mlflow.log_artifact("src/models", artifact_path="src")
        mlflow.log_artifact("src/utils", artifact_path="src")

        mlflow.log_artifact("src/lightning_model.py", artifact_path="src")
        mlflow.log_artifact("src/preprocessing.py", artifact_path="src")
        mlflow.log_artifact("src/train.py", artifact_path="src")


if __name__ == "__main__":
    main()
