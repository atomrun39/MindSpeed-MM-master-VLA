import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        bsz, seq_len = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0, device=device)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)
        if enc.shape[-1] < self.embedding_dim:
            pad = self.embedding_dim - enc.shape[-1]
            enc = torch.cat([enc, torch.zeros(bsz, seq_len, pad, device=device, dtype=enc.dtype)], dim=-1)
        return enc


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        bsz, seq_len, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == bsz:
            timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        else:
            raise ValueError("Expected timesteps shape [B]")
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x
