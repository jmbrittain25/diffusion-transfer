import numpy as np
import torch
import torch.nn.functional as F

from .base_model_trainer import BaseModelTrainer


class DiffusionModelTrainer(BaseModelTrainer):

    def __init__(self):
        super().__init__()

    def get_beta_schedule(self, steps=1000):
        return np.linspace(1e-4, 0.02, steps)

    def add_noise(self, x, t, betas):
        noise = torch.randn_like(x)
        cum_alpha = np.cumprod(1 - betas)
        sqrt_alpha_t = torch.sqrt(torch.tensor(cum_alpha[t], device=x.device))
        sqrt_one_minus_alpha_t = torch.sqrt(1 - sqrt_alpha_t**2)
        return sqrt_alpha_t.unsqueeze(1).unsqueeze(1) * x + sqrt_one_minus_alpha_t.unsqueeze(1).unsqueeze(1) * noise, noise

    def train_epoch(self,  model, loader, optimizer, betas, device, phys_weight=0.0):
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
        return total_loss / num_batches\

    def evaluate(self, model, loader, betas, device):
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
        