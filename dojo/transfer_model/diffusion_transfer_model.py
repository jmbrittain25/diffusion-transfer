import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_transfer_model import BaseTransferModel


class DiffusionTransferModel(BaseTransferModel):

    def __init__(self, feat_dim=5, time_emb_dim=32, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.down1 = nn.Linear(feat_dim + time_emb_dim, hidden_dim)
        self.down2 = nn.Linear(hidden_dim, hidden_dim)
        self.up1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x, timestep, mask=None):
        t_emb = self.time_embed(timestep).unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, t_emb], dim=-1)
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))
        u1 = F.relu(self.up1(torch.cat([d2, d1], dim=-1)))
        pred = self.out(u1)
        if mask is not None:
            pred = pred * mask.unsqueeze(-1)
        return pred