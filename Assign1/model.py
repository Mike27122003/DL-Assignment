import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim=1,
        device="cpu",
    ):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        self.rnn = nn.RNN(input_dim, hidden_dim, device=self.device)

    def forward(self, x: torch.Tensor):
        out = self.rnn(x)
        return out