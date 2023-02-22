from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def data_loader(n_training, device):
    # data = np.load("dataset/tiny_nerf_data.npz")
    data = np.load("/data2/cll/zju/cv/cv-2022/Homework4/dataset/tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]
    images = torch.from_numpy(data['images'][:n_training]).to(device)
    poses = torch.from_numpy(data['poses']).to(device)
    focal = torch.from_numpy(data['focal']).to(device)
    return images, poses, focal


def get_rays(height, width, focal_length, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input parameters:
        height: image height
        width: image width
        focal_length: focal length
        c2w: camera to world transformation matrix
    output:
        rays_o: shape (height, width, 3) origin of each ray
        rays_d: shape (height, width, 3) direction of each ray
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


def dataset_tests():
    data = np.load("dataset/tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    height = images.shape[1]
    width = images.shape[2]
    print("Basic information about the dataset:")
    print("Images shape: {}".format(images.shape))
    print("Image height: {}".format(height))
    print("Image width: {}".format(width))
    print("Poses shape: {}".format(poses.shape))
    print("Focal length: {}".format(focal))

    testimg_idx = 101
    testimg, testpose = images[testimg_idx], poses[testimg_idx]
    print("Test pose:")
    print(testpose)
    print("Test image:")
    plt.imshow(testimg)

    print("Test poses: position (x, y, z) and direction (r, p, y)")
    dirs = np.stack(
        [np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(),
                  origins[..., 1].flatten(),
                  origins[..., 2].flatten(),
                  dirs[..., 0].flatten(),
                  dirs[..., 1].flatten(),
                  dirs[..., 2].flatten(),
                  length=0.5,
                  normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()


def test_ray(device):
    n_training = 106
    images, poses, focal = data_loader(n_training, device)
    height = images.shape[1]
    width = images.shape[2]
    testimg_idx = 101
    testimg = images[testimg_idx]
    testpose = poses[testimg_idx]
    with torch.no_grad():
        ray_origin, ray_direction = get_rays(height, width, focal, testpose)

    print("Ray Origin [{}]: {}".format(ray_origin.shape, ray_origin[height // 2, width // 2, :]))
    print("Ray Direction [{}]: {}".format(ray_direction.shape, ray_direction[height // 2, width // 2, :]))


# dataset_tests()
# test_ray("cpu")