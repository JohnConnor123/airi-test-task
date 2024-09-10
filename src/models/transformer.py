import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = Transformer(input_size=358, embed_dim=64, num_heads=4, num_layers=2)

    # Пример использования модели
    # X_train - тензор размерности (batch_size, 358)
    X_train = torch.randn((1, 358))
    outputs = model(X_train)
    print(outputs)
