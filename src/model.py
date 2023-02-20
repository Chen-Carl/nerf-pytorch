from typing import Callable, Optional, Tuple, List

import torch
import torch.nn as nn

from PositionalEncoder import PositionalEncoder
from EarlyStopping import EarlyStopping
from NeRF import NeRF
from Sampling import stratified_sample, hierarchical_sample
from VolumnRendering import raw2outputs
from settings import Settings

settings = Settings()
device = settings.device
d_input = settings.d_input
n_freqs = settings.n_freqs
log_space = settings.log_space
use_viewdirs = settings.use_viewdirs
n_freqs_views = settings.n_freqs_views
d_filter = settings.d_filter
n_layers = settings.n_layers
skip = settings.skip
use_fine_model = settings.use_fine_model
lr = settings.lr


def init_models():
    # Encoders
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # View direction encoders
    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views, log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers, d_filter, skip, d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF(encoder.d_output, n_layers, d_filter, skip,
                          d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    optimizer = torch.optim.Adam(model_params, lr=lr)
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper


def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:
    return [
        inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)
    ]


def prepare_chunks(
        points: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunksize: int = 2**15
) -> List[torch.Tensor]:
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


def prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    kwargs_sample_stratified: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model=None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    input parameters:
        rays_o: shape [N_rays, 3] origin of each ray
        rays_d: shape [N_rays, 3] direction of each ray
        near: float distance to near plane
        far: float distance to far plane
        encoding_fn: function that encodes a point into a vector
        coarse_model: coarse model
        kwargs_sample_stratified: dict of kwargs for stratified sampling
        n_samples_hierarchical: int number of samples for hierarchical sampling
        kwargs_sample_hierarchical: dict of kwargs for hierarchical sampling
        fine_model: fine model
        viewdirs_encoding_fn: function that encodes a view direction into a vector
        chunksize: int size of chunks to split the input into
    return:

    """
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = stratified_sample(rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    outputs = {'z_vals_stratified': z_vals}

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = hierarchical_sample(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
            **kwargs_sample_hierarchical
        )

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize)
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['rgb_map_0'] = rgb_map_0
        outputs['depth_map_0'] = depth_map_0
        outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs
