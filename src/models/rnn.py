import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(
            1
        )  # Добавляем размерность для seq_length (теперь (batch_size, 1, input_dim))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :]  # Используем последний временной шаг
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = RNN(input_size=358, hidden_dim=64, num_layers=2)

    # Пример тензора входных данных (batch_size=32, input_size=358)
    X_train = torch.rand(32, 358)
    outputs = model(X_train)

    print(outputs)
