import pickle
from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import Settings
from data_processing import data_loader, get_rays
from model import init_models, nerf_forward
from utils import plot_samples, crop_center

settings = Settings()
n_iters = settings.n_iters
one_image_per_step = settings.one_image_per_step
center_crop = settings.center_crop
center_crop_iters = settings.center_crop_iters
batch_size = settings.batch_size
display_rate = settings.display_rate
warmup_iters = settings.warmup_iters
warmup_min_fitness = settings.warmup_min_fitness
n_samples_hierarchical = settings.n_samples_hierarchical
n_samples = settings.n_samples
near = settings.near
far = settings.far
kwargs_sample_stratified = settings.kwargs_sample_stratified
kwargs_sample_hierarchical = settings.kwargs_sample_hierarchical
chunksize = settings.chunksize
device = settings.device

n_training = 100
model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
images, poses, focal = data_loader(n_training, device)


def train():
    testimg_idx = 99
    testimg, testpose = images[testimg_idx], poses[testimg_idx]
    # Shuffle rays across all images.
    if not one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack([
            torch.stack(get_rays(height, width, focal, p), 0)
            for p in poses[:n_training]
        ], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    with tqdm(total=n_iters) as pbar:
        for i in range(n_iters):
            model.train()

            if one_image_per_step:
                # Randomly pick an image as the target.
                target_img_idx = np.random.randint(images.shape[0])
                target_img = images[target_img_idx].to(device)
                if center_crop and i < center_crop_iters:
                    target_img = crop_center(target_img)
                height, width = target_img.shape[:2]
                target_pose = poses[target_img_idx].to(device)
                rays_o, rays_d = get_rays(height, width, focal, target_pose)
                rays_o = rays_o.reshape([-1, 3])
                rays_d = rays_d.reshape([-1, 3])
            else:
                # Random over all images.
                batch = rays_rgb[i_batch:i_batch + batch_size]
                batch = torch.transpose(batch, 0, 1)
                rays_o, rays_d, target_img = batch
                height, width = target_img.shape[:2]
                i_batch += batch_size
                # Shuffle after one epoch
                if i_batch >= rays_rgb.shape[0]:
                    rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                    i_batch = 0
            target_img = target_img.reshape([-1, 3])

            # Run one iteration of TinyNeRF and get the rendered RGB image.
            outputs = nerf_forward(
                rays_o,
                rays_d,
                near,
                far,
                encode,
                model,
                kwargs_sample_stratified=kwargs_sample_stratified,
                n_samples_hierarchical=n_samples_hierarchical,
                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                fine_model=fine_model,
                viewdirs_encoding_fn=encode_viewdirs,
                chunksize=chunksize
            )

            # Check for any numerical issues.
            for k, v in outputs.items():
                if torch.isnan(v).any():
                    print(f"! [Numerical Alert] {k} contains NaN.")
                if torch.isinf(v).any():
                    print(f"! [Numerical Alert] {k} contains Inf.")

            # Backprop!
            rgb_predicted = outputs['rgb_map']
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            psnr = -10.0 * torch.log10(loss)
            train_psnrs.append(psnr.item())
            pbar.set_description("Loss: {:.4f}".format(loss.item()))

            # Evaluate testimg at given display rate.
            if (i + 1) % display_rate == 0:
                model.eval()
                height, width = testimg.shape[:2]
                rays_o, rays_d = get_rays(height, width, focal, testpose)
                rays_o = rays_o.reshape([-1, 3])
                rays_d = rays_d.reshape([-1, 3])
                outputs = nerf_forward(
                    rays_o,
                    rays_d,
                    near,
                    far,
                    encode,
                    model,
                    kwargs_sample_stratified=kwargs_sample_stratified,
                    n_samples_hierarchical=n_samples_hierarchical,
                    kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                    fine_model=fine_model,
                    viewdirs_encoding_fn=encode_viewdirs,
                    chunksize=chunksize
                )

                rgb_predicted = outputs['rgb_map']
                loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
                val_psnr = -10. * torch.log10(loss)
                val_psnrs.append(val_psnr.item())
                iternums.append(i)

                # Plot example outputs
                fig, ax = plt.subplots(1, 4, figsize=(24, 4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
                ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
                ax[0].set_title(f'Iteration: {i}')
                ax[1].imshow(testimg.detach().cpu().numpy())
                ax[1].set_title(f'Target')
                ax[2].plot(range(0, i + 1), train_psnrs, 'r')
                ax[2].plot(iternums, val_psnrs, 'b')
                ax[2].set_title('PSNR (train=red, val=blue')
                z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
                z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
                if 'z_vals_hierarchical' in outputs:
                    z_vals_hierarch = outputs['z_vals_hierarchical'].view(
                        (-1, n_samples_hierarchical))
                    z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
                else:
                    z_sample_hierarch = None
                plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
                ax[3].margins(0)
                plt.show()

            pbar.update(1)

            # Check PSNR for issues and stop if any are found.
            if i == warmup_iters - 1:
                if val_psnr < warmup_min_fitness:
                    print(
                        f'Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...'
                    )
                    return False, train_psnrs, val_psnrs
            elif i < warmup_iters:
                if warmup_stopper is not None and warmup_stopper(i, psnr):
                    print(
                        f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...'
                    )
                    return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs


def save_models(model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper, settings):
    # using pickle save all parameters
    with open(settings.model_path, 'wb') as f:
        pickle.dump(model.state_dict(), f)
    with open(settings.fine_model_path, 'wb') as f:
        pickle.dump(fine_model.state_dict(), f)
    with open(settings.encode_path, 'wb') as f:
        pickle.dump(encode.state_dict(), f)
    with open(settings.encode_viewdirs_path, 'wb') as f:
        pickle.dump(encode_viewdirs.state_dict(), f)
    with open(settings.optimizer_path, 'wb') as f:
        pickle.dump(optimizer.state_dict(), f)
    with open(settings.warmup_stopper_path, 'wb') as f:
        pickle.dump(warmup_stopper.state_dict(), f)


def main():
    settings = Settings()
    n_restarts = settings.n_restarts
    for _ in range(n_restarts):
        model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
        success, train_psnrs, val_psnrs = train()
        if success and val_psnrs[-1] >= warmup_min_fitness:
            save_models(model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper, settings)
            print('Training successful!')
            break

    print("Done!")


main()