import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple

from settings import Settings


class NeRFDataset(Dataset):
    def __init__(self, filepath, n_training, device):
        data = np.load(filepath)
        self.images = torch.from_numpy(data['images']).to(device)
        self.height, self.width = self.images.shape[1:3]
        self.poses = torch.from_numpy(data['poses']).to(device)
        self.focal = torch.from_numpy(data['focal']).to(device)
        self.n_training = n_training

        # original shape [n_training, 2, height, width, 3]
        all_rays = torch.stack([
            torch.stack(self.get_rays(self.height, self.width, self.focal, pose), 0)
            for pose in self.poses[:self.n_training]
        ])
        self.rays_o = all_rays[:, 0, ...].reshape(-1, 3)
        self.rays_d = all_rays[:, 1, ...].reshape(-1, 3)
        self.target_rgb = self.images.reshape(-1, 3)

    def __len__(self):
        return self.n_training * self.height * self.width

    def __getitem__(self, idx):
        return self.rays_o[idx], self.rays_d[idx], self.target_rgb[idx]
    
    def get_rays(self, 
        height, width, focal_length, 
        c2w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            * rays_o: [height, width, 3] ray origins
            * rays_d: [height, width, 3] ray directions
        """
        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(c2w),
            torch.arange(height, dtype=torch.float32).to(c2w),
            indexing='ij'
        )
        i = i.transpose(-1, -2)
        j = j.transpose(-1, -2)
        directions = torch.stack(
            [
                (i - width * 0.5) / focal_length, 
                -(j - height * 0.5) / focal_length,
                -torch.ones_like(i)
            ],
            dim=-1
        )

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d


def load_dataset(batch_size, device):
    settings = Settings()
    dataset = NeRFDataset("dataset/tiny_nerf_data.npz", settings.n_training, device)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, shuffle=False, sampler=sampler, batch_size=batch_size)
    return loader, sampler, dataset