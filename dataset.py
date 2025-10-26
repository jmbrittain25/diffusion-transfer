import numpy as np
import torch
from torch.utils.data import Dataset


class OrbitDataset(Dataset):
    def __init__(self, npz_file, max_len=512):  # Pad to max_len
        data = np.load(npz_file, allow_pickle=True)
        self.trajectories = data['trajectories']
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        padded = np.pad(traj, ((0, self.max_len - traj.shape[0]), (0, 0)), mode='constant')
        mask = np.array([1] * traj.shape[0] + [0] * (self.max_len - traj.shape[0]))
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
