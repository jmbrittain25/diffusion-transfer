from .base_loss_model import BaseLossModel


class MSELossModel(BaseLossModel):

    def __init__(self):
        super().__init__()

    def __call__(self): # TODO - figure out what this should be!
        return

# TODO - integrate below code


# MSE loss on noise, masked
mse_loss = F.mse_loss(pred_noise, noise, reduction='none') * mask.unsqueeze(-1)
mse_loss = mse_loss.mean()