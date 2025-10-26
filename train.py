import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from astropy.constants import G, M_earth
import astropy.units as u

# Constants
MU = (G * M_earth).to(u.km**3 / u.s**2).value  # Gravitational parameter in km^3/s^2

class OrbitDataset(Dataset):
    def __init__(self, npz_file, data_size=None, max_len=512):
        """
        Loads trajectories from NPZ. Optionally subsamples to data_size.
        Pads sequences to max_len.
        """
        data = np.load(npz_file, allow_pickle=True)
        trajectories = data['trajectories']
        if data_size is not None and data_size < len(trajectories):
            indices = np.random.choice(len(trajectories), data_size, replace=False)
            trajectories = [trajectories[i] for i in indices]
        self.trajectories = trajectories
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        padded = np.pad(traj, ((0, self.max_len - traj.shape[0]), (0, 0)), mode='constant')
        mask = np.array([1] * traj.shape[0] + [0] * (self.max_len - traj.shape[0]))
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

class SimpleDiffusionUNet(nn.Module):
    def __init__(self, feat_dim=5, time_emb_dim=32, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
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

def get_beta_schedule(steps=1000):
    return np.linspace(1e-4, 0.02, steps)

def add_noise(x, t, betas):
    noise = torch.randn_like(x)
    cum_alpha = np.cumprod(1 - betas)
    sqrt_alpha_t = torch.sqrt(torch.tensor(cum_alpha[t.long().squeeze().cpu().numpy()], device=x.device))
    sqrt_one_minus_alpha_t = torch.sqrt(1 - sqrt_alpha_t**2)
    noisy = (sqrt_alpha_t.unsqueeze(1).unsqueeze(1) * x +
             sqrt_one_minus_alpha_t.unsqueeze(1).unsqueeze(1) * noise)
    return noisy, noise

def compute_physics_losses(pred, mask, mu=MU):
    """
    Computes physics-informed losses.
    Assumes pred shape [batch, seq_len, 5] with [x, y, vx, vy, t_norm]
    Only computes on masked (valid) parts.
    """
    # Extract valid parts (apply mask)
    mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
    x, y, vx, vy, _ = [pred[..., i] * mask[..., 0] for i in range(5)]

    # Radius r
    r = torch.sqrt(x**2 + y**2 + 1e-10)  # Avoid div by zero

    # Specific energy E = v^2 / 2 - mu / r
    v_sq = vx**2 + vy**2
    E = 0.5 * v_sq - mu / r

    # Specific angular momentum h magnitude (in 2D: x*vy - y*vx)
    h = torch.abs(x * vy - y * vx)

    # Losses: Variance over sequence (should be constant)
    # Mean over batch and seq, ignoring zero-masked
    valid_count = mask.sum(dim=[1, 2]) + 1e-10  # Per batch

    E_mean = E.sum(dim=1) / (valid_count)  # [batch]
    E_var = ((E - E_mean.unsqueeze(1))**2 * mask[..., 0]).sum(dim=1) / valid_count
    energy_loss = E_var.mean()  # Scalar

    h_mean = h.sum(dim=1) / valid_count
    h_var = ((h - h_mean.unsqueeze(1))**2 * mask[..., 0]).sum(dim=1) / valid_count
    ang_mom_loss = h_var.mean()

    return energy_loss, ang_mom_loss

def train_epoch(model, loader, optimizer, betas, device, phys_weight=0.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch, mask in loader:
        batch, mask = batch.to(device), mask.to(device)
        t = torch.randint(0, len(betas), (batch.shape[0],)).to(device)  # [batch]
        t_norm = (t / len(betas)).unsqueeze(1).float()  # Normalized [0,1]

        noisy, noise = add_noise(batch, t.cpu().numpy(), betas)
        pred_noise = model(noisy, t_norm, mask)

        # MSE loss on noise, masked
        mse_loss = F.mse_loss(pred_noise, noise, reduction='none') * mask.unsqueeze(-1)
        mse_loss = mse_loss.mean()

        # Physics losses on predicted clean (approx: noisy - pred_noise)
        pred_clean = noisy - pred_noise
        energy_loss, ang_mom_loss = compute_physics_losses(pred_clean, mask)
        phys_loss = energy_loss + ang_mom_loss

        loss = mse_loss + phys_weight * phys_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def evaluate(model, loader, betas, device):
    model.eval()
    metrics = {'rmse_pos': 0.0, 'rmse_vel': 0.0, 'energy_var': 0.0, 'ang_mom_var': 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch, mask in loader:
            batch, mask = batch.to(device), mask.to(device)
            t = torch.randint(0, len(betas), (batch.shape[0],)).to(device)
            t_norm = (t / len(betas)).unsqueeze(1).float()

            noisy, _ = add_noise(batch, t.cpu().numpy(), betas)
            pred = model(noisy, t_norm, mask)

            # Approximate clean for eval
            pred_clean = noisy - pred

            # RMSE on positions [0:2] and velocities [2:4], masked
            true_pos = batch[:, :, :2] * mask.unsqueeze(-1)
            pred_pos = pred_clean[:, :, :2] * mask.unsqueeze(-1)
            rmse_pos = torch.sqrt(((true_pos - pred_pos)**2).mean())

            true_vel = batch[:, :, 2:4] * mask.unsqueeze(-1)
            pred_vel = pred_clean[:, :, 2:4] * mask.unsqueeze(-1)
            rmse_vel = torch.sqrt(((true_vel - pred_vel)**2).mean())

            energy_loss, ang_mom_loss = compute_physics_losses(pred_clean, mask)

            metrics['rmse_pos'] += rmse_pos.item()
            metrics['rmse_vel'] += rmse_vel.item()
            metrics['energy_var'] += energy_loss.item()
            metrics['ang_mom_var'] += ang_mom_loss.item()
            num_batches += 1

    for k in metrics:
        metrics[k] /= num_batches
    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full dataset, subsample if specified
    full_dataset = OrbitDataset(args.data_file, data_size=args.data_size)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleDiffusionUNet(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    betas = get_beta_schedule(args.diffusion_steps)

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer, betas, device, args.phys_weight)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

    # Evaluate
    val_metrics = evaluate(model, val_loader, betas, device)
    print("Validation Metrics:", val_metrics)

    # Save model and metrics
    torch.save(model.state_dict(), args.output.replace('.json', '.pth'))
    with open(args.output, 'w') as f:
        json.dump(val_metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model for Orbital Trajectories")
    parser.add_argument('--data_file', type=str, default='orbits_2d_dataset.npz', help='Path to NPZ dataset')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension (model size)')
    parser.add_argument('--data_size', type=int, default=None, help='Subsample dataset size (None for full)')
    parser.add_argument('--phys_weight', type=float, default=0.0, help='Weight for physics losses')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON for metrics')
    args = parser.parse_args()
    main(args)