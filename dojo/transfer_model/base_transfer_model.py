from abc import abstractmethod
import torch.nn as nn


class BaseTransferModel(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, timestep): # TODO - figure out what this should be!
        return
