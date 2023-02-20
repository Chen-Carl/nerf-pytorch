from typing import Optional, Tuple

import torch


def stratified_sample(
    rays_o : torch.Tensor,
    rays_d : torch.Tensor,
    near : float,
    far : float,
    n_samples : int,
    perturb : Optional[bool] = True,
    inverse_depth : Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input parameters:
        * rays_o: shape [N_rays, 3] origin of each ray
        * rays_d: shape [N_rays, 3] direction of each ray
        * near: float distance to near plane
        * far: float distance to far plane
        * n_samples: int number of samples
        * perturb: bool whether to perturb the samples
        * inverse_depth: bool whether to use inverse depth
    return:
        * pts: shape [N_rays, n_samples, 3] points along each ray
        * z_vals: shape [N_rays, n_samples] depth values of each point
    """
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    if not inverse_depth:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    
    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.cat([mids, z_vals[-1:]])
        lower = torch.cat([z_vals[:1], mids])
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def hierarchical_sample(
    rays_o : torch.Tensor,
    rays_d : torch.Tensor,
    z_vals : torch.Tensor,
    weights : torch.Tensor,
    n_samples : int,
    perturb : Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    input parameters:
        * rays_o: shape [N_rays, 3] origin of each ray
        * rays_d: shape [N_rays, 3] direction of each ray
        * z_vals: shape [N_rays, n_samples] depth values of each point
        * weights: shape [N_rays, n_samples] weights of each point
        * n_samples: int number of samples
        * perturb: bool whether to perturb the samples
    return:
        * pts: shape [N_rays, n_samples * 2, 3] points along each ray
        * z_vals_combined: shape [N_rays, n_samples * 2] depth values of each point
        * new_z_samples: shape [N_rays, n_samples] new depth values of each point
    """
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb)
    new_z_samples = new_z_samples.detach()

    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    return pts, z_vals_combined, new_z_samples


def sample_pdf(
    bins : torch.Tensor,
    weights : torch.Tensor,
    n_samples : int,
    perturb : Optional[bool] = False
) -> torch.Tensor:
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
    u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + (bins_g[..., 1] - bins_g[..., 0]) * t
    return samples