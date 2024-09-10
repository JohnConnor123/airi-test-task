# flake8: noqa: F401
import lightning as L
import mlflow
import numpy as np
import torch

from utils import compute_metrics


class MyLightningModel(L.LightningModule):
    """Класс обертка для подсчета метрик"""

    def __init__(self, model, optimizer, loss_fn, lr_scheduler=None, example_input_array=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.example_input_array = example_input_array
        self.outs = {
            "train": {"y_pred": [], "y_true": [], "loss": []},
            "val": {"y_pred": [], "y_true": [], "loss": []},
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_ind):
        self.model.train()
        X, y = batch
        outputs = self.model(X)
        loss = self.loss_fn(outputs, y)

        self.outs["train"]["y_pred"].append(outputs.cpu().detach())
        self.outs["train"]["y_true"].append(y.cpu().detach())
        self.outs["train"]["loss"].append(loss.item())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        X, y = batch

        with torch.no_grad():
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)

        self.outs["val"]["y_pred"].append(outputs.cpu().detach())
        self.outs["val"]["y_true"].append(y.cpu().detach())
        self.outs["val"]["loss"].append(loss.cpu().detach())

        return loss.mean()

    def on_train_epoch_end(self):
        self._on_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        self._on_epoch_end(stage="val")

    def _on_epoch_end(self, stage):
        y_pred = torch.cat(self.outs[stage]["y_pred"], dim=0).squeeze()
        y_true = torch.cat(self.outs[stage]["y_true"], dim=0).squeeze()
        loss = self.outs[stage]["loss"]

        metrics = compute_metrics(y_true, y_pred)
        metrics = {f"{stage}_{key}": value for key, value in metrics.items()}
        metrics[f"{stage}_loss"] = np.mean(loss)
        metrics["lr"] = self.optimizers().optimizer.param_groups[0]["lr"]

        mlflow.log_metrics(metrics=metrics, step=self.current_epoch)
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        self.outs[stage]["y_true"] = []
        self.outs[stage]["y_pred"] = []

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return {"optimizer": self.optimizer}
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def log_hyperparams(self, cfg):
        mlflow.log_params(cfg)
