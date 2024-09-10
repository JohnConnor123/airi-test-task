# flake8: noqa: F401
import datetime
import os
import pickle
import sys
from typing import Union

import mlflow
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


sys.path.append(os.sep.join(__file__.split(os.sep)[:-2]))
import preprocessing


current_dir = os.getcwd()
os.chdir(os.path.dirname(os.sep.join(__file__.split(os.sep)[:-2])))
cfg = OmegaConf.load("src/config/config.yaml")
os.chdir(current_dir)


def cache_dataloaders():
    train_dataloader, val_dataloader, test_dataloader = preprocessing.do_pipeline()

    os.makedirs(f"{cfg.general.data_dir}/secondary/dataloaders", exist_ok=True)

    # Save the dataloaders to a file
    with open(f"{cfg.general.data_dir}/secondary/dataloaders/dataloaders.pkl", "wb") as file:
        pickle.dump((train_dataloader, val_dataloader, test_dataloader), file)

    # Load the dataloaders from the file
    with open(f"{cfg.general.data_dir}/secondary/dataloaders/dataloaders.pkl", "rb") as file:
        train_dataloader, val_dataloader, test_dataloader = pickle.load(file)


def get_curr_time():
    now = datetime.datetime.now()
    return now.strftime("%d.%m.%Y %H-%M-%S")


def _union_mlflow_experiments():
    experiments = [
        f"RandomSearch for mlp network {name} loss" for name in ["mae", "mse", "huber"]
    ]
    exp_ids = []

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    for exp in experiments:
        exp = mlflow.get_experiment_by_name(exp)
        exp_ids.append(exp)
    from pprint import pprint

    pprint(exp_ids)

    run_ids = []
    for exp_id in exp_ids:
        mlflow.search_runs(experiment_ids=[exp_id])
        run_ids.append(run_ids)

    from pprint import pprint

    pprint(run_ids)
    out = []
    for _, run in run_ids:
        # Логирование параметров
        params = run["params"]
        for param, value in params.items():
            out.append(param, value)

        # Логирование метрик
        metrics = run["metrics"]
        for metric, value in metrics.items():
            out.append(metric, value)

        # Копирование артефактов, если необходимо
        artifact_uri = run["artifact_uri"]
        mlflow.log_artifact(artifact_uri)

    print(out)


def get_test_dataloader_from_y(
    y_test: torch.Tensor, X_test: Union[torch.Tensor, None] = None
) -> DataLoader:
    if X_test is None:
        X_test, _ = preprocessing.prepare_test_data()

    test_dataset = preprocessing.get_dataset(X_test, y_test.squeeze())
    test_dataloader = preprocessing.get_dataloader(test_dataset)
    return test_dataloader


if __name__ == "__main__":
    # train_dataloader, val_dataloader, test_dataloader = do_pipeline()
    with open(f"{cfg.general.data_dir}/secondary/dataloaders/dataloaders.pkl", "rb") as file:
        train_dataloader, val_dataloader, test_dataloader = pickle.load(file)
