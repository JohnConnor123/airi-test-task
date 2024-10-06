# flake8: noqa: F401
import datetime
import os
import pickle
import sys
from typing import Union

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

import preprocessing
from utils import compute_metrics


cfg = OmegaConf.load("src/config/config.yaml")


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


def get_test_dataloader_from_y(
    y_test: torch.Tensor, X_test: Union[torch.Tensor, None] = None
) -> DataLoader:
    if X_test is None:
        X_test, _ = preprocessing.prepare_test_data()

    test_dataset = preprocessing.get_dataset(X_test, y_test.squeeze())
    test_dataloader = preprocessing.get_dataloader(test_dataset)
    return test_dataloader


class LargeParamMLP(nn.Module):
    def __init__(
        self, input_size, hidden_sizes, output_size, activation, dropout_rate, batch_norm
    ):
        super(LargeParamMLP, self).__init__()
        layers = []
        in_size = input_size

        # Создаем скрытые слои
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_and_evaluate_model(
    model, optimizer, train_dataloader, val_dataloader, run_name, num_epochs=10
):
    with mlflow.start_run(run_name=run_name):
        example_input_array = torch.ones((2, 358))

        criterion = nn.MSELoss()
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=14, factor=0.2, verbose=True
        )

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            all_preds = []
            all_targets = []
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                all_preds.append(outputs.cpu().detach())
                all_targets.append(targets.cpu().detach())

            lr_scheduler.step(loss)
            all_preds = torch.cat(all_preds).squeeze()
            all_targets = torch.cat(all_targets, dim=0).squeeze()
            avg_train_loss = running_loss / len(train_dataloader)

            metrics = compute_metrics(all_targets, all_preds)
            metrics = {f"train_{key}": value for key, value in metrics.items()}
            metrics["train_loss"] = avg_train_loss
            mlflow.log_metrics(metrics=metrics, step=epoch)

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    all_preds.append(outputs.cpu().detach())
                    all_targets.append(targets.cpu().detach())
            all_preds = torch.cat(all_preds).squeeze()
            all_targets = torch.cat(all_targets, dim=0).squeeze()
            avg_val_loss = val_loss / len(val_dataloader)

            metrics = compute_metrics(all_targets, all_preds)
            metrics = {f"val_{key}": value for key, value in metrics.items()}
            metrics["val_loss"] = avg_val_loss
            mlflow.log_metrics(metrics=metrics, step=epoch)

        signature = mlflow.models.infer_signature(
            model_input=example_input_array.numpy(),
            model_output=model.cpu()(example_input_array).detach().numpy(),
        )

        mlflow.pytorch.log_model(
            model.cpu(),
            artifact_path="model",
            input_example=example_input_array.numpy(),
            signature=signature,
            # registered_model_name=cfg.model.name,
        )

        mlflow.log_artifact("src/config", artifact_path="src")

        mlflow.log_artifact("src/models", artifact_path="src")
        mlflow.log_artifact("src/utils", artifact_path="src")

        mlflow.log_artifact("src/preprocessing.py", artifact_path="src")
        mlflow.log_artifact("src/train.py", artifact_path="src")
        mlflow.log_artifact("src/lightning_model.py", artifact_path="src")
        return avg_val_loss


def objective(config):
    num_layers = config["num_layers"]
    hidden_sizes = [config["hidden_sizes"] for _ in range(num_layers)]
    activation = config["activation"]
    dropout_rate = config["dropout_rate"]
    batch_norm = config["batch_norm"]
    optimizer_name = config["optimizer"]
    learning_rate = config["lr"]

    print("hidden_sizes", hidden_sizes)

    model = LargeParamMLP(
        input_size=358,
        hidden_sizes=hidden_sizes,
        output_size=1,
        activation=activation,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
    ).cuda()

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    run_name = f"params={round(sum(p.numel() for p in model.parameters())/1e3, 2)}k, optimizer={optimizer_name},\
        dropout_rate={dropout_rate}, batch_norm={batch_norm}, lr={learning_rate},\
            num_layers={num_layers}, hidden_sizes={hidden_sizes}, activation={activation}"

    with open(
        rf"D:\\Python_Projects\\Jupyter\\test-job-tasks\\airi\\data\\secondary\\dataloaders\dataloaders.pkl",
        "rb",
    ) as file:
        train_dataloader, val_dataloader, test_dataloader = pickle.load(file)

    validation_loss = train_and_evaluate_model(
        model, optimizer, train_dataloader, val_dataloader, run_name=run_name, num_epochs=300
    )

    return validation_loss


def main():
    # train_dataloader, val_dataloader, test_dataloader = do_pipeline()
    mlflow.pytorch.autolog()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.enable_system_metrics_logging()

    experiment_name = "NAS for MLP"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    from ray import train, tune

    search_space = {
        "num_layers": tune.randint(2, 10),
        "hidden_sizes": tune.sample_from(lambda x: np.random.randint(64, 513)),
        "activation": tune.choice(["relu", "tanh", "sigmoid"]),
        "dropout_rate": tune.uniform(0.0, 0.5),
        "batch_norm": tune.choice([True, False]),
        "optimizer": tune.choice(["Adam", "RMSprop"]),
        "lr": tune.loguniform(1e-5, 1e-2),
    }

    tuner = tune.Tuner(objective, param_space=search_space)
    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min").config)


if __name__ == "__main__":
    # cache_dataloaders()
    main()
