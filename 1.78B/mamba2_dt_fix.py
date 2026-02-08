"""
Mamba2 dt Bounded Fix
=====================

Applies sigmoid-bounded dt to Mamba2 instances, preventing the dt death spiral
while preserving fast CUDA kernels via dt_softplus=False.

Usage:
    from mamba2_dt_fix import wrap_mamba2_with_bounded_dt

    ssm = Mamba2(d_model=512, ...)
    ssm = wrap_mamba2_with_bounded_dt(ssm, dt_min=0.001, dt_max=0.1)
"""

import torch
import torch.nn as nn


def wrap_mamba2_with_bounded_dt(mamba2_module, dt_min=0.001, dt_max=0.1):
    """
    Wrap a Mamba2 module to use bounded sigmoid dt.

    Replaces unbounded softplus with:
        dt = dt_min + (dt_max - dt_min) * sigmoid(dt_raw + dt_bias)

    This keeps dt bounded to [dt_min, dt_max], preventing the death spiral
    while preserving native CUDA kernels (called with dt_softplus=False).

    Args:
        mamba2_module: A Mamba2 instance from mamba_ssm
        dt_min: Minimum dt value (default: 0.001)
        dt_max: Maximum dt value (default: 0.1)

    Returns:
        Wrapped module with bounded dt
    """
    mamba2_module._dt_min = dt_min
    mamba2_module._dt_max = dt_max
    mamba2_module._original_forward = mamba2_module.forward

    def bounded_dt_forward(hidden_states, inference_params=None, **kwargs):
        """Forward pass with bounded sigmoid dt instead of unbounded softplus."""
        if hasattr(mamba2_module, 'dt_activation'):
            original_activation = mamba2_module.dt_activation

            def bounded_activation(dt_raw):
                return dt_min + (dt_max - dt_min) * torch.sigmoid(dt_raw)

            mamba2_module.dt_activation = bounded_activation
            try:
                result = mamba2_module._original_forward(hidden_states, inference_params, **kwargs)
            finally:
                mamba2_module.dt_activation = original_activation
            return result

        return mamba2_module._original_forward(hidden_states, inference_params, **kwargs)

    mamba2_module.forward = bounded_dt_forward
    return mamba2_module
