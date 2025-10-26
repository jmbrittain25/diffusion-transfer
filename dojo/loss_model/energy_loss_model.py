from .base_loss_model import BaseLossModel


class EnergyLossModel(BaseLossModel):

    def __init__(self):
        super().__init__()

    def __call__(self): # TODO - figure out what this should be!
        return


# TODO - integrate below code


MU = (G * M_earth).to(u.km**3 / u.s**2).value  # Gravitational parameter in km^3/s^2


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