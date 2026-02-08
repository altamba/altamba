"""
Mamba dt Bounded Fix
====================

Applies sigmoid-bounded dt to Mamba1/Mamba2 instances, preventing the dt death
spiral while preserving fast CUDA kernels.

Usage:
    from mamba2_dt_fix import wrap_mamba2_with_bounded_dt, wrap_mamba1_with_bounded_dt

    ssm2 = Mamba2(d_model=512, ...)
    ssm2 = wrap_mamba2_with_bounded_dt(ssm2, dt_min=0.001, dt_max=0.1)

    ssm1 = Mamba(d_model=512, ...)
    ssm1 = wrap_mamba1_with_bounded_dt(ssm1, dt_min=0.001, dt_max=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def wrap_mamba1_with_bounded_dt(mamba1_module, dt_min=0.001, dt_max=0.1):
    """
    Wrap a Mamba (v1) module to use bounded sigmoid dt.

    Mamba1 applies softplus inside the CUDA kernel (delta_softplus=True) and
    uses dt_proj.weight directly (not as a module call). This wrapper forces
    the slow path, applies bounded sigmoid manually, and passes
    delta_softplus=False to the kernel.

    Final dt = dt_min + (dt_max - dt_min) * sigmoid(W @ dt_raw + bias)

    Args:
        mamba1_module: A Mamba instance from mamba_ssm
        dt_min: Minimum dt value (default: 0.001)
        dt_max: Maximum dt value (default: 0.1)

    Returns:
        Wrapped module with bounded dt
    """
    from einops import rearrange
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None

    mamba1_module._original_forward = mamba1_module.forward

    def bounded_dt_forward(hidden_states, inference_params=None, **kwargs):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = mamba1_module._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = mamba1_module.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            mamba1_module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if mamba1_module.in_proj.bias is not None:
            xz = xz + rearrange(mamba1_module.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(mamba1_module.A_log.float())
        x, z = xz.chunk(2, dim=1)

        if conv_state is not None:
            conv_state.copy_(F.pad(x, (mamba1_module.d_conv - x.shape[-1], 0)))

        if causal_conv1d_fn is None:
            x = mamba1_module.act(mamba1_module.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(mamba1_module.conv1d.weight, "d 1 w -> d w"),
                bias=mamba1_module.conv1d.bias,
                activation=mamba1_module.activation,
            )

        x_dbl = mamba1_module.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [mamba1_module.dt_rank, mamba1_module.d_state, mamba1_module.d_state], dim=-1
        )
        dt = mamba1_module.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        # Bounded sigmoid instead of softplus: apply bias ourselves
        dt = dt + mamba1_module.dt_proj.bias.to(dtype=dt.dtype).unsqueeze(0).unsqueeze(-1)
        dt = dt_min + (dt_max - dt_min) * torch.sigmoid(dt)

        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x, dt, A, B, C, mamba1_module.D.float(),
            z=z, delta_bias=None, delta_softplus=False,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = mamba1_module.out_proj(y)
        return out

    mamba1_module.forward = bounded_dt_forward
    return mamba1_module
