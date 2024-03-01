import torch
import torch.nn as nn


class MLP1(nn.Module):
    def __init__(self, dim_emb, class_nb, hidden_size=None, layers_nb=None):
        super(MLP1, self).__init__()
        self.L1 = nn.Linear(dim_emb, class_nb)

    def forward(self, X):
        out = self.L1(X)
        out = torch.permute(out, (0, 2, 1))
        return out


class MLP2(nn.Module):
    def __init__(self, dim_emb, class_nb, hidden_size, layers_nb=None):
        super(MLP2, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(dim_emb, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, class_nb),
        )

    def forward(self, x):
        x = self.seq(x)
        x = torch.permute(x, (0, 2, 1))
        return x


class Lstm(nn.Module):
    def __init__(self, dim_emb, class_nb, hidden_size, layers_nb):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_emb, hidden_size=hidden_size, num_layers=layers_nb)
        self.fc = nn.Linear(hidden_size, class_nb)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = torch.permute(out, (0, 2, 1))
        return out


class BiLSTM(nn.Module):
    def __init__(self, dim_emb, class_nb, hidden_size, layers_nb):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(dim_emb, hidden_size, layers_nb, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, class_nb)  # Multiply by 2 for bidirectional
        self.layers_nb = layers_nb
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, _, _ = x.size()
        h0 = torch.zeros(self.layers_nb * 2, batch_size, self.hidden_size).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.layers_nb * 2, batch_size, self.hidden_size).to(x.device)  # Initialize cell state

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = torch.permute(out, (0, 2, 1))
        return out
