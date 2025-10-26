import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):  # Denoiser
    def __init__(self, feat_dim=5, time_emb_dim=32, hidden_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.down1 = nn.Linear(feat_dim + time_emb_dim, hidden_dim)
        self.down2 = nn.Linear(hidden_dim, hidden_dim)
        self.up1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x, timestep):  # x: [batch, seq_len, feat_dim], timestep: [batch, 1]
        t_emb = self.time_embed(timestep).unsqueeze(1).repeat(1, x.shape[1], 1)  # Broadcast to seq
        x = torch.cat([x, t_emb], dim=-1)
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))
        u1 = F.relu(self.up1(torch.cat([d2, d1], dim=-1)))  # Skip connection
        return self.out(u1)

# Diffusion process (in training loop)
def add_noise(x, t, beta_schedule):  # Simplified; implement full DDPM noise schedule
    noise = torch.randn_like(x)
    alpha_t = 1 - beta_schedule[t]
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

class SimpleDiffusionUNet(nn.Module):
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

# Diffusion utils (simplified DDPM)
def get_beta_schedule(steps=1000):
    return np.linspace(1e-4, 0.02, steps)

def add_noise(x, t, betas):
    noise = torch.randn_like(x)
    cum_alpha = np.cumprod(1 - betas)
    sqrt_alpha_t = torch.sqrt(torch.tensor(cum_alpha[t], device=x.device))
    sqrt_one_minus_alpha_t = torch.sqrt(1 - sqrt_alpha_t**2)
    return sqrt_alpha_t.unsqueeze(1).unsqueeze(1) * x + sqrt_one_minus_alpha_t.unsqueeze(1).unsqueeze(1) * noise, noise

# Training loop
def train(model, loader, epochs=10, steps=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    betas = get_beta_schedule(steps)
    for epoch in range(epochs):
        for batch, mask in loader:
            t = torch.randint(0, steps, (batch.shape[0], 1), device=batch.device).float() / steps
            noisy, noise = add_noise(batch, t.long().squeeze().cpu().numpy(), betas)
            pred_noise = model(noisy, t, mask)
            loss = F.mse_loss(pred_noise, noise, reduction='none') * mask.unsqueeze(-1)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
