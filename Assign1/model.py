import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        layer_dim=1,
        device="cpu",
    ):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        self.rnn = nn.RNN(input_dim, hidden_dim, device=self.device)
        self.pred_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out, _ = self.rnn(x)
        pred = self.pred_head(out[:, -1, :])
        return pred
