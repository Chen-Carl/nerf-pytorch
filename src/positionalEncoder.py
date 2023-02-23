import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    Sine-cosine positional encoder for input points.
    """
    def __init__(self, d_input, n_freqs, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.log_space = log_space
        self.embed_fns = [lambda x: x]

        # Define frequencies in linear scale or log scale
        if self.log_space:
            freq_bands = torch.logspace(0., 1., self.n_freqs, base=2.)
        else:
            freq_bands = torch.linspace(1, 2**(self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

