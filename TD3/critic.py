import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, init_w: float = 3e-3):
        super().__init__()
        self.hidden1 = nn.Linear(state_dim, 256)
        self.hidden2 = nn.Linear(256 + action_dim, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.hidden1(s))
        x = torch.cat((s, action), dim=-1)
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        value = self.out(x)
        return value
