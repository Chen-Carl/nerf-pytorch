import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional
from matplotlib import pyplot as plt

from nerf import NeRFWrapper
from earlyStopping import EarlyStopping
from settings import Settings
from datasets import load_dataset


def render(model, settings, iter, dataset, train_psnrs, iternums, val_psnrs, img_id, args):
    height = dataset.height
    width = dataset.width
    focal = dataset.focal
    pose = dataset.poses[img_id]
    rays_o, rays_d = dataset.get_rays(height, width, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    outputs = model(rays_o, rays_d)

    def plot_samples(
        z_vals: torch.Tensor,
        z_hierarch: Optional[torch.Tensor] = None,
        ax: Optional[np.ndarray] = None
    ):
        """
        Plot stratified and (optional) hierarchical samples.
        """
        y_vals = 1 + np.zeros_like(z_vals)
        if ax is None:
            ax = plt.subplot()
        ax.plot(z_vals, y_vals, 'b-o')
        if z_hierarch is not None:
            y_hierarch = np.zeros_like(z_hierarch)
            ax.plot(z_hierarch, y_hierarch, 'r-o')
        ax.set_ylim([-1, 2])
        ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
        ax.axes.yaxis.set_visible(False)
        ax.grid(True)
        return ax

    fig, ax = plt.subplots(1, 4, figsize=(24, 4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
    ax[0].imshow(outputs["rgb_map"].reshape([dataset.height, dataset.width, 3]).detach().cpu().numpy())
    ax[0].set_title(f'Iteration: {iter + 1}')
    ax[1].imshow(dataset.images[img_id].detach().cpu().numpy())
    ax[1].set_title(f'Target')
    ax[2].plot(range(0, len(train_psnrs)), train_psnrs, 'r')
    ax[2].plot(iternums, val_psnrs, 'b')
    ax[2].set_title('PSNR (train=red, val=blue')
    z_vals_strat = outputs['z_vals_stratified'].view((-1, settings.n_samples))
    z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
    if 'z_vals_hierarchical' in outputs:
        z_vals_hierarch = outputs['z_vals_hierarchical'].view(
            (-1, settings.n_samples_hierarchical))
        z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
    else:
        z_sample_hierarch = None
    plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
    ax[3].margins(0)

    if args.local_rank == 0:
        if settings.imshow:
            plt.show()  
        else:
            plt.imsave(
                "results/iter_{}_{}_{}.png".format(iter + 1, img_id, int(round(time.time() * 1000))), 
                outputs["rgb_map"].reshape([dataset.height, dataset.width, 3]).detach().cpu().numpy()
            )
    return outputs


def train(args):
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", args.local_rank)

    settings = Settings()
    model = NeRFWrapper(settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    warmup_stopper = EarlyStopping(patience=50)
    loader, sampler, dataset = load_dataset(settings.batch_size, device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=[args.local_rank])

    train_psnrs = []
    val_psnrs = []
    iternums = []

    with tqdm(total=settings.n_iters, position=args.local_rank * 2) as pbar:
        for i in range(settings.n_iters):
            model.train()
            sampler.set_epoch(i)
            for rays_o, rays_d, rgb in loader:
                outputs = model(rays_o, rays_d)
                # Check for any numerical issues.
                for k, v in outputs.items():
                    if torch.isnan(v).any():
                        print(f"! [Numerical Alert] {k} contains NaN.")
                    if torch.isinf(v).any():
                        print(f"! [Numerical Alert] {k} contains Inf.")

                loss = F.mse_loss(outputs["rgb_map"], rgb)
                pbar.set_description("GPU {} Loss: {:.4f}".format(args.local_rank, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                psnr = -10.0 * torch.log10(loss)
                train_psnrs.append(psnr.item())

            if (i + 1) % settings.display_rate == 0:
                model.eval()
                img_ids = [101, 102]
                for img_id in img_ids:
                    render(model, settings, i, dataset, train_psnrs, iternums, val_psnrs, img_id, args)

                if args.local_rank == 0:
                    torch.save(model.state_dict(), "models/checkpoints/checkpoint_{}.pth".format(int(round(time.time() * 1000))))
                                    
            pbar.update(1)

    return True, train_psnrs, val_psnrs, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    args = parser.parse_args()
    settings = Settings()
    for i in range(settings.n_restarts):
        success, train_psnrs, val_psnrs, model = train(args)
        if success:
            if args.local_rank == 0:
                torch.save(model.state_dict(), "models/nerf_model_{}.pth".format(int(round(time.time() * 1000))))
            print("Training successful. Model saved.")
            break

main()