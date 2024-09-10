import pickle

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard import program
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


# from torchsummary import summary


torch.set_float32_matmul_precision("high")


def init_tensorboard():
    tracking_address = "src\\logs\\tb_logs\\VAE"  # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


# Define the Variational Autoencoder model
class LinearVAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Latent space layers
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_shape),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)

        z = mu + eps * std
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar


# Класс обертка для подсчета метрик
class MyLightningModel_VAE(L.LightningModule):
    def __init__(self, model=None, lr=1e-3):
        super().__init__()
        self.model = model
        self.example_input_array = torch.ones((1, 358))
        self.loss_fn = loss_vae
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        batch, _ = batch

        reconstruction, mu, log_dispersion = self.model(batch)
        loss = self.loss_fn(batch, mu, log_dispersion, reconstruction)

        self.log_dict({"train_loss": loss.mean()}, prog_bar=True, on_step=True, on_epoch=True)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        batch, _ = batch

        with torch.no_grad():
            reconstruction, mu, log_dispersion = self.model(batch)
            loss = self.loss_fn(batch, mu, log_dispersion, reconstruction)

        self.log_dict({"val_loss": loss.mean()}, prog_bar=True, on_step=True, on_epoch=True)

        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, factor=0.2, patience=14
                ),  # ExponentialLR(optimizer, gamma=0.93)
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# Функция для замены целевых значений на предсказания модели
def replace_labels_with_predictions(dataloader, model):
    all_inputs = []
    all_predictions = []

    model.eval()  # Перевод модели в режим предсказания (evaluation)
    with torch.no_grad():  # Отключение вычисления градиентов
        for inputs, _ in dataloader:
            inputs = inputs.to(model.device)  # Перенос данных на устройство модели (GPU/CPU)
            predictions = model(inputs)  # Предсказания модели
            all_inputs.append(inputs.cpu())
            all_predictions.append(predictions.cpu())

    # Конкатенация всех входных данных и предсказаний
    all_inputs = torch.cat(all_inputs, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    return all_predictions


def KL_divergence(mu, log_dispersion):
    """
    часть функции потерь, которая отвечает за "близость" латентных представлений разных людей
    """
    loss = -0.5 * torch.sum(1 + log_dispersion - mu.pow(2) - torch.exp(log_dispersion))
    return loss


def mse(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции
    """
    loss = nn.MSELoss()
    return loss(reconstruction, x)


def loss_vae(x, mu, log_dispersion, reconstruction, alpha=1, beta=2e-4):
    return alpha * KL_divergence(mu, log_dispersion) + beta * mse(x, reconstruction)


if __name__ == "__main__":
    # Create an instance of the VAE model
    input_shape = 358
    latent_dim = 16
    model = LinearVAE(input_shape, latent_dim)

    # Load the dataloaders from the file
    with open("data/secondary/dataloaders/dataloaders.pkl", "rb") as file:
        train_dataloader, val_dataloader, test_dataloader = pickle.load(file)

    # Define the loss function and optimizer
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger = TensorBoardLogger("src/logs/tb_logs", name="VAE", log_graph=False)
    init_tensorboard()
    max_epochs = 150

    VAE_MNIST = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                "lr-Adam", mode="min", stopping_threshold=1e-6, patience=max_epochs, verbose=1
            ),
            ModelCheckpoint(monitor="val_loss"),
        ],
    )  # сохраняю стату каждый батч

    VAE_MNIST.fit(
        model=MyLightningModel_VAE(model),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # visualize the latent space
    import numpy as np
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm
    from umap import UMAP

    concated_dataset = ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    dataloader = DataLoader(
        concated_dataset, batch_size=train_dataloader.batch_size, shuffle=True, num_workers=1
    )  # train_dataloader.num_workers)

    y_real = torch.ones(len(dataloader.dataset), 1)
    y_fake = torch.zeros(len(dataloader.dataset), 1)

    X_fake = torch.tensor([])
    X_real = torch.tensor([])
    for x_real, _ in tqdm(dataloader, desc="Generate fakes", total=len(dataloader)):
        X_real = torch.cat([X_real, x_real])
        noise = torch.rand(x_real.size(0), latent_dim)
        X_fake = torch.cat([X_fake, (model.decode(noise).detach())])

    fake_dataset = TensorDataset(X_fake, y_fake)
    real_dataset = TensorDataset(X_real, y_real)

    total = len(fake_dataset) + len(real_dataset)
    to_shuffle = list(range(total))
    np.random.shuffle(to_shuffle)

    model = torch.load(
        "src/logs/mlflow-registry/7/95bfc1be45b54eec9742b1341be7908c/artifacts/model/data/model.pth"
    )
    model.eval()
    y_fake = replace_labels_with_predictions(
        DataLoader(fake_dataset, batch_size=32, shuffle=True), model
    )

    X_full = np.concatenate([X_fake, X_real]).reshape(total, -1)[to_shuffle]
    y_full = np.concatenate([y_fake, y_real]).ravel()[to_shuffle]

    with open("data/secondary/dataloaders/dataloader-extended-x2-v1.pkl", "wb") as file:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train).squeeze(), torch.tensor(y_train).squeeze()
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test).squeeze(), torch.tensor(y_test).squeeze()
        )
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        pickle.dump((train_dataloader, test_dataloader, ""), file)

    embedding2d = UMAP(n_components=2).fit_transform(X_full)
    embedding3d = UMAP(n_components=3).fit_transform(X_full)
    print(embedding2d.shape, embedding3d.shape)

    # Babyplot
    from babyplots import Babyplot

    bp = Babyplot(background_color="#000000", turntable=True, rotation_rate=0.002)
    bp.add_plot(
        embedding3d.tolist(),
        "pointCloud",
        "categories",
        ["Real" if y else "Generated" for y in y_full],
        {
            "animationLoop": True,
            "animationDelay": 800,
            "showAxes": [True, True, True],
            "legendTitleFontColor": "#ffffff",
            "showLegend": True,
            "legendTitle": "Visualization of the fake and real data \
                with UMAP-projection onto 2D and 3D spaces",
            "legendTitleFontSize": 13,
            "colorScale": "jet",
            "fontColor": "#ffffff",
            "folded": True,  # True будет показывать проекцию из 3d в 2d
            "foldedEmbedding": embedding2d.tolist(),
        },
    )
    bp.save_as_html(
        "src/logs/babyplots/Real-fake data UMAP-rediction visualization, latent_dim=8.html",
        fullscreen=True,
    )
