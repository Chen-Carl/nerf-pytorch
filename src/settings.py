import torch

class Settings:
    def __init__(self):
        # Encoders
        self.d_input = 3             # Number of input dimensions
        self.n_freqs = 10            # Number of encoding functions for samples
        self.log_space = True        # If set, frequencies scale in log space
        self.use_viewdirs = True     # If set, use view direction as input
        self.n_freqs_views = 4       # Number of encoding functions for views

        # Stratified sampling
        self.n_samples = 64          # Number of spatial samples per ray
        self.perturb = True          # If set, applies noise to sample positions
        self.inverse_depth = False   # If set, samples points linearly in inverse depth
        self.near = 2.0
        self.far = 6.0

        # Model
        self.d_filter = 128          # Dimensions of linear layer filters
        self.n_layers = 2            # Number of layers in network bottleneck
        self.skip = []               # Layers at which to apply input residual
        self.use_fine_model = True   # If set, creates a fine model
        self.d_filter_fine = 128     # Dimensions of linear layer filters of fine network
        self.n_layers_fine = 6       # Number of layers in fine network bottleneck

        # Hierarchical sampling
        self.n_samples_hierarchical = 64     # Number of samples per ray
        self.perturb_hierarchical = False    # If set, applies noise to sample positions

        # Optimizer
        self.lr = 5e-4               # Learning rate

        # Training
        self.n_training = 100
        self.n_iters = 2000
        self.batch_size = 2**12          # Number of rays per gradient step (power of 2)
        self.one_image_per_step = True   # One image per gradient step (disables batching)
        self.chunksize = 2**12           # Modify as needed to fit in GPU memory
        self.center_crop = True          # Crop the center of image (one_image_per_)
        self.center_crop_iters = 50      # Stop cropping center after this many epochs
        self.display_rate = 100          # Display test output every X epochs
        self.imshow = True               # If set, displays test output using matplotlib

        # Early Stopping
        self.warmup_iters = 100          # Number of iterations during warmup phase
        self.warmup_min_fitness = 10.0   # Min val PSNR to continue training at warmup_iters
        self.n_restarts = 10             # Number of times to restart if training stalls

        # We bundle the kwargs for various functions to pass all at once.
        self.kwargs_sample_stratified = {
            'n_samples': self.n_samples,
            'perturb': self.perturb,
            'inverse_depth': self.inverse_depth
        }
        self.kwargs_sample_hierarchical = {
            'perturb': self.perturb
        }
