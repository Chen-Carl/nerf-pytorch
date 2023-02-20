from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, 
        d_input : int = 3,
        n_layers : int = 8,
        d_filter : int = 256,
        skip : Tuple[int] = (4, ),
        d_viewdirs : Optional[int] = None,
    ):
        """
        parameters:
            d_input: dimension of input points
            n_layers: number of layers
            d_filter: dimension of hidden layers
            skip: skip connections
            d_viewdirs: dimension of view directions
        """
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.d_viewdirs = d_viewdirs
        self.activation = F.relu

        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)]
        )
        for i in range(n_layers - 1):
            if i in self.skip:
                self.layers.append(nn.Linear(d_filter + self.d_input, d_filter))
            else:
                self.layers.append(nn.Linear(d_filter, d_filter))
        
        if self.d_viewdirs is not None:
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            self.output = nn.Linear(d_filter, 4)

    def forward(self, 
        x : torch.Tensor, 
        viewdirs : Optional[torch.Tensor] = None
    ):
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")
        
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)
        
        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)
            x = self.rgb_filters(x)
            x = torch.cat([x, viewdirs], dim=-1)
            x = self.activation(self.branch(x))
            x = self.output(x)
            x = torch.cat([x, alpha], dim=-1)
        else:
            x = self.output(x)
            
        return x
            