"""
DualRazorNorm: Learnable Blending Operation for Dual-Path Architectures
========================================================================

The Razor blending operation:
    y = s * [(c1 - W1) * LN(x_denoiser) + (c2 - W2) * x_main]

Where W1, W2 are learned per-layer scalars and s is a learned output scale.
"""

import torch
import torch.nn as nn


class DualRazorNorm(nn.Module):
    """
    Dual RazorNorm: Blend two signals with independent learnable weights.

    Output = s * [(c - W1) * LayerNorm(x_denoiser) + (c' - W2) * x_main]

    Supports dimension mismatch: if denoiser has smaller dimension, it gets
    projected up to match the main signal dimension before blending.

    W1 controls normalized signal contribution
    W2 controls raw signal contribution
    """
    def __init__(self, dim: int, eps: float = 1e-5, init_w1: float = 0.1, init_w2: float = 1.5,
                 use_scaling: bool = True, dim_denoiser: int = None):
        super().__init__()
        self.dim = dim
        self.dim_denoiser = dim_denoiser or dim
        self.use_scaling = use_scaling

        self.ln = nn.LayerNorm(dim, eps=eps)
        self.ln_denoiser = nn.LayerNorm(self.dim_denoiser, eps=eps) if self.dim_denoiser != dim else self.ln

        if self.dim_denoiser != dim:
            self.proj_denoiser = nn.Linear(self.dim_denoiser, dim)
        else:
            self.proj_denoiser = None

        if use_scaling:
            self.output_scale_0 = nn.Parameter(torch.ones(1) * 1.5)
            self.output_scale_1 = nn.Parameter(torch.ones(1) * -1.0)
        else:
            self.output_scale_0 = None
            self.output_scale_1 = None

        self.W1 = nn.Parameter(torch.ones(1) * init_w1)
        self.W2 = nn.Parameter(torch.ones(1) * init_w2)

    def forward(self, x_main: torch.Tensor, x_denoiser: torch.Tensor, i: int) -> torch.Tensor:
        """
        Args:
            x_main: Main signal (full dimension)
            x_denoiser: Denoiser signal (potentially smaller dimension)
            i: Layer index for alternating strategy

        For even layers (i%2==0): x_main=SSM, x_denoiser=Transformer (Post-LN)
        For odd layers (i%2==1): x_main=Transformer, x_denoiser=SSM (Pre-LN)
        """
        if i % 2 == 0:
            # Even layers: Post-LN destructive interference
            normalized = self.ln_denoiser(x_denoiser)
            if self.proj_denoiser is not None:
                normalized = self.proj_denoiser(normalized)
            output = (1.4 - self.W1) * normalized + (0.6 - self.W2) * x_main
            if self.output_scale_0 is not None:
                output = output * self.output_scale_0
        else:
            # Odd layers: Pre-LN constructive/destructive interference
            normalized = self.ln_denoiser(x_denoiser)
            if self.proj_denoiser is not None:
                normalized = self.proj_denoiser(normalized)
            if i < 80:
                output = (1.1 - self.W1) * normalized + (0.5 - self.W2) * x_main
            else:
                output = (1.5 - self.W1) * normalized + (0.5 - self.W2) * x_main
            if self.output_scale_1 is not None:
                output = output * self.output_scale_1

        return output
