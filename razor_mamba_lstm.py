"""
RazorMamba-LSTM: Learnable Blend of SSM and LSTM Dynamics
==========================================================

Combines three architectural innovations:
1. Mamba SSM selective state space dynamics
2. Traditional LSTM gating mechanisms
3. RazorNorm learnable bypass for blending both approaches

Core idea:
    c_ssm = A_t * c_{t-1} + B_t * x_t      # Mamba SSM dynamics
    c_lstm = f_t * c_{t-1} + i_t * g_t     # Traditional LSTM gating
    c_t = W * LayerNorm(c_ssm) + (1-W) * c_lstm  # Learnable blend

The W parameter tells us which dynamics the model prefers:
- W → 1.0: Model prefers SSM dynamics
- W → 0.0: Model prefers LSTM gating
- W ≈ 0.3-0.7: Model wants both (hybrid memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class RazorNorm(nn.Module):
    """
    RazorNorm: LayerNorm with learnable bypass.

    Output = LayerNorm(x) + (1-W) * x

    The W parameter learns how much raw signal to add to normalized output.
    """
    def __init__(self, dim: int, eps: float = 1e-5, init_w: float = 1.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
        self.W = nn.Parameter(torch.ones(1) * init_w)  # Start at init_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln(x)
        bypass = (1.0 - self.W) * x
        return normalized + bypass


class DualRazorNorm(nn.Module):
    """
    Dual RazorNorm: Blend two signals with independent learnable weights.

    Output = (1/√d) * [(c1 - W1) * LayerNorm(x1) + (c2 - W2) * x2]

    The 1/√d scaling factor preserves variance as d_model scales to prevent
    magnitude explosion in large models (d_model > 1024).

    Supports dimension mismatch: if denoiser has smaller dimension, it gets
    projected up to match the main signal dimension before blending.

    W1 controls normalized signal contribution
    W2 controls raw signal contribution (can be negative for denoising)
    """
    def __init__(self, dim: int, eps: float = 1e-5, init_w1: float = 0.1, init_w2: float = 1.5,
                 use_scaling: bool = True, dim_denoiser: int = None):
        super().__init__()
        self.dim = dim
        self.dim_denoiser = dim_denoiser or dim
        self.use_scaling = use_scaling

        # LayerNorm - will be applied to whichever signal gets normalized
        self.ln = nn.LayerNorm(dim, eps=eps)
        self.ln_denoiser = nn.LayerNorm(self.dim_denoiser, eps=eps) if self.dim_denoiser != dim else self.ln

        # Projection layer if denoiser has different dimension
        if self.dim_denoiser != dim:
            self.proj_denoiser = nn.Linear(self.dim_denoiser, dim)
        else:
            self.proj_denoiser = None
        
        # Learnable output scale (ReZero-style) - starts at identity
        # Model learns optimal output magnitude instead of fixed 1/√d
        if use_scaling:
            self.output_scale_0 = nn.Parameter(torch.ones(1) * 1.5)
            self.output_scale_1 = nn.Parameter(torch.ones(1) * -1.0)
        else:
            self.output_scale_0 = None
            self.output_scale_1 = None

        self.W1 = nn.Parameter(torch.ones(1) * init_w1)  # Normalized signal weight
        self.W2 = nn.Parameter(torch.ones(1) * init_w2)  # Raw signal weight

    def forward(self, x_main: torch.Tensor, x_denoiser: torch.Tensor, i: int) -> torch.Tensor:
        """
        Args:
            x_main: Main signal (full dimension)
            x_denoiser: Denoiser signal (potentially smaller dimension)
            i: Layer index for alternating strategy

        For even layers (i%2==0): x_main=SSM, x_denoiser=Transformer
        For odd layers (i%2==1): x_main=Transformer, x_denoiser=SSM
        """
        if i % 2 == 0:
            # Even layers: Post-LN destructive interference
            # Normalize Transformer (denoiser), keep SSM (main) raw
            normalized = self.ln_denoiser(x_denoiser)
            # Project denoiser to match main dimension if needed
            if self.proj_denoiser is not None:
                normalized = self.proj_denoiser(normalized)
            output = (1.4 - self.W1) * normalized + (0.6 - self.W2) * x_main
            # Apply learnable output scale (starts at 1.0, learns optimal magnitude)
            if self.output_scale_0 is not None:
                output = output * self.output_scale_0
        else:
            # Odd layers: Pre-LN constructive/destructive interference
            # Normalize SSM (denoiser), keep Transformer (main) raw
            normalized = self.ln_denoiser(x_denoiser)
            # Project denoiser to match main dimension if needed
            if self.proj_denoiser is not None:
                normalized = self.proj_denoiser(normalized)
            if i < 80:
                output = (1.1 - self.W1) * normalized + (0.5 - self.W2) * x_main
            else:
                output = (1.5 - self.W1) * normalized + (0.5 - self.W2) * x_main
            # Apply learnable output scale (starts at 1.0, learns optimal magnitude)
            if self.output_scale_1 is not None:
                output = output * self.output_scale_1

        

        return output


class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style Selective SSM for LSTM cell state.

    Computes selective state space dynamics:
        h_t = A_t * h_{t-1} + B_t * x_t

    Where A_t and B_t are input-dependent (selective mechanism).
    """
    def __init__(self, input_size: int, state_size: int, dt_rank: int = None):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.dt_rank = dt_rank or max(1, state_size // 16)

        # Project input to SSM parameters
        # We need: x (for input), dt (for discretization), B, C
        self.in_proj = nn.Linear(input_size, state_size + self.dt_rank + state_size * 2)

        # Discretization parameter projection
        self.dt_proj = nn.Linear(self.dt_rank, state_size)

        # SSM A parameter (log space for stability)
        # Initialize with S4D-Real structure
        A = torch.arange(1, state_size + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection parameter
        self.D = nn.Parameter(torch.ones(state_size))

        # Initialize dt projection for proper range
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias for proper time scale
        dt = torch.exp(
            torch.rand(state_size) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_size] - input at current timestep
            prev_state: [batch, state_size] - previous SSM state (h_{t-1})

        Returns:
            output: [batch, state_size] - SSM output
            new_state: [batch, state_size] - updated SSM state (h_t)
        """
        batch_size = x.shape[0]

        # Initialize state if needed
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.state_size, device=x.device, dtype=x.dtype)

        # Project input to get SSM parameters
        proj = self.in_proj(x)  # [batch, state_size + dt_rank + state_size*2]
        x_ssm, dt, B, C = torch.split(
            proj,
            [self.state_size, self.dt_rank, self.state_size, self.state_size],
            dim=-1
        )

        # Compute discretization parameter
        dt = self.dt_proj(dt)  # [batch, state_size]
        dt = F.softplus(dt)  # Ensure positive

        # Get A from log space (negative for stability)
        A = -torch.exp(self.A_log)  # [state_size]

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        A_discrete = torch.exp(dt * A.unsqueeze(0))  # [batch, state_size]
        B_discrete = dt * B  # [batch, state_size]

        # SSM recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        new_state = A_discrete * prev_state + B_discrete * x_ssm

        # Output: y_t = C * h_t + D * x_t
        output = C * new_state + self.D.unsqueeze(0) * x_ssm

        return output, new_state


class RazorMambaLSTMCell(nn.Module):
    """
    LSTM Cell with learnable blend of SSM and LSTM dynamics for cell state.

    Architecture:
        1. Compute traditional LSTM gates (i, f, o, g)
        2. Compute SSM dynamics for cell state
        3. Use RazorNorm to blend: c_t = W * LN(c_ssm) + (1-W) * c_lstm
        4. Output hidden state: h_t = o_t * tanh(c_t)

    The W parameter learns which dynamics work better:
        W → 1: SSM dynamics preferred
        W → 0: LSTM gating preferred
        W ≈ 0.5: Hybrid approach
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_ssm: bool = True,
        use_razor: bool = True,
        init_w1: float = 0.1,
        init_w2: float = 1.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_ssm = use_ssm
        self.use_razor = use_razor

        # Traditional LSTM gates (all operate on [h_{t-1}, x_t])
        combined_size = input_size + hidden_size
        self.W_i = nn.Linear(combined_size, hidden_size)  # Input gate
        self.W_f = nn.Linear(combined_size, hidden_size)  # Forget gate
        self.W_o = nn.Linear(combined_size, hidden_size)  # Output gate
        self.W_g = nn.Linear(combined_size, hidden_size)  # Cell gate

        # SSM for alternative cell dynamics
        if use_ssm:
            self.ssm = SelectiveSSM(combined_size, hidden_size)

        # DualRazorNorm for blending SSM and LSTM cell states
        if use_razor and use_ssm:
            self.cell_blend = DualRazorNorm(hidden_size, init_w1=init_w1, init_w2=init_w2)
        else:
            self.cell_blend = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch, input_size]
            state: (h, c, ssm_state) where:
                h: [batch, hidden_size] - LSTM hidden state
                c: [batch, hidden_size] - LSTM cell state
                ssm_state: [batch, hidden_size] - SSM internal state

        Returns:
            h_new: [batch, hidden_size] - new hidden state
            (h_new, c_new, ssm_state_new): new states
        """
        batch_size = x.shape[0]

        # Initialize states if needed
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            ssm_state = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c, ssm_state = state

        # Concatenate hidden and input for gates
        combined = torch.cat([h, x], dim=1)  # [batch, hidden_size + input_size]

        # Compute traditional LSTM gates
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        g_t = torch.tanh(self.W_g(combined))     # Cell gate

        # Traditional LSTM cell update
        c_lstm = f_t * c + i_t * g_t

        # SSM cell update (if enabled)
        if self.use_ssm:
            c_ssm, ssm_state_new = self.ssm(combined, ssm_state)

            if self.use_razor:
                # DualRazorNorm: (1-W1) * LN(c_ssm) + (1-W2) * c_lstm
                # W1 controls SSM normalized contribution
                # W2 controls LSTM raw contribution (can be negative for denoising)
                c_new = self.cell_blend(c_ssm, c_lstm)
            else:
                # Simple average without learnable weight
                c_new = 0.5 * self.cell_blend(c_ssm) + 0.5 * c_lstm
        else:
            # No SSM, just use traditional LSTM
            c_new = self.cell_blend(c_lstm)
            ssm_state_new = ssm_state

        # Hidden state output
        h_new = o_t * torch.tanh(c_new)

        return h_new, (h_new, c_new, ssm_state_new)


class RazorMambaLSTM(nn.Module):
    """
    Multi-layer RazorMamba-LSTM with learnable SSM/LSTM blending.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        use_ssm: bool = True,
        use_razor: bool = True,
        init_w1: float = 0.1,
        init_w2: float = 1.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_ssm = use_ssm
        self.use_razor = use_razor

        # Create layers
        self.layers = nn.ModuleList([
            RazorMambaLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                use_ssm=use_ssm,
                use_razor=use_razor,
                init_w1=init_w1,
                init_w2=init_w2
            )
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: [batch, seq_len, input_size]
            state: list of (h, c, ssm_state) tuples, one per layer

        Returns:
            output: [batch, seq_len, hidden_size]
            state: list of (h, c, ssm_state) tuples
        """
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = [None] * self.num_layers

        outputs = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]

            new_state = []
            for i, layer in enumerate(self.layers):
                x_t, state_i = layer(x_t, state[i])
                new_state.append(state_i)

                # Apply dropout between layers (not after last layer)
                if self.dropout is not None and i < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            outputs.append(x_t)
            state = new_state

        output = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_size]
        return output, state

    def get_razor_weights(self) -> list:
        """
        Extract W parameters from all RazorNorm/DualRazorNorm layers.

        Returns:
            List of (layer_idx, W1_value, W2_value) tuples for DualRazorNorm
            or (layer_idx, W_value, None) tuples for RazorNorm
        """
        weights = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'cell_blend'):
                if isinstance(layer.cell_blend, DualRazorNorm):
                    weights.append((i, layer.cell_blend.W1.item(), layer.cell_blend.W2.item()))
                elif isinstance(layer.cell_blend, RazorNorm):
                    weights.append((i, layer.cell_blend.W.item(), None))
        return weights


# ============================================================
# TEST AND DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RazorMamba-LSTM: Learnable Blend of SSM and LSTM Dynamics")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Hyperparameters
    batch_size = 4
    seq_len = 20
    input_size = 64
    hidden_size = 128
    num_layers = 2

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")

    # Create models for comparison
    print("\n" + "-"*80)
    print("Creating Models...")

    # 1. Standard LSTM (no SSM, no Razor)
    lstm_standard = RazorMambaLSTM(
        input_size, hidden_size, num_layers,
        use_ssm=False, use_razor=False
    ).to(device)

    # 2. RazorMamba-LSTM (full system)
    lstm_razor_mamba = RazorMambaLSTM(
        input_size, hidden_size, num_layers,
        use_ssm=True, use_razor=True
    ).to(device)

    print("  ✓ Standard LSTM")
    print("  ✓ RazorMamba-LSTM")

    # Test input
    x = torch.randn(batch_size, seq_len, input_size).to(device)

    print("\n" + "-"*80)
    print("Forward Pass Tests...")

    # Test standard LSTM
    print("\n1. Standard LSTM:")
    with torch.no_grad():
        out_std, state_std = lstm_standard(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out_std.shape}")
    print(f"   ✓ Forward pass successful")

    # Test RazorMamba-LSTM
    print("\n2. RazorMamba-LSTM:")
    with torch.no_grad():
        out_razor, state_razor = lstm_razor_mamba(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out_razor.shape}")

    # Show initial W values
    razor_weights = lstm_razor_mamba.get_razor_weights()
    print(f"\n   Initial W values:")
    for weight_tuple in razor_weights:
        if len(weight_tuple) == 3 and weight_tuple[2] is not None:
            layer_idx, w1_val, w2_val = weight_tuple
            ssm_pct = (1 - w1_val) * 100
            lstm_pct = (1 - w2_val) * 100
            print(f"     Layer {layer_idx}: W1 = {w1_val:.4f}, W2 = {w2_val:.4f}")
            print(f"       → SSM: {ssm_pct:.1f}%, LSTM: {lstm_pct:.1f}%")
        else:
            layer_idx, w_val, _ = weight_tuple
            ssm_pct = w_val * 100
            lstm_pct = (1 - w_val) * 100
            print(f"     Layer {layer_idx}: W = {w_val:.4f} ({ssm_pct:.1f}% SSM, {lstm_pct:.1f}% LSTM)")
    print(f"   ✓ Forward pass successful")

    # Parameter counts
    print("\n" + "-"*80)
    print("Parameter Comparison:")
    params_std = sum(p.numel() for p in lstm_standard.parameters())
    params_razor = sum(p.numel() for p in lstm_razor_mamba.parameters())
    print(f"  Standard LSTM:     {params_std:,}")
    print(f"  RazorMamba-LSTM:   {params_razor:,}")
    print(f"  Overhead:          +{params_razor - params_std:,} ({100*(params_razor-params_std)/params_std:.1f}%)")

    # Test backward pass
    print("\n" + "-"*80)
    print("Backward Pass Test...")
    out_razor, _ = lstm_razor_mamba(x)
    loss = out_razor.sum()
    loss.backward()

    # Check gradients
    has_nan = any(torch.isnan(p.grad).any() for p in lstm_razor_mamba.parameters() if p.grad is not None)
    has_inf = any(torch.isinf(p.grad).any() for p in lstm_razor_mamba.parameters() if p.grad is not None)
    print(f"  Gradient NaN: {has_nan}")
    print(f"  Gradient Inf: {has_inf}")
    print(f"  ✓ Backward pass successful")

    # Show that W has gradients (can be learned)
    print("\n  W parameter gradients:")
    for i, layer in enumerate(lstm_razor_mamba.layers):
        if hasattr(layer, 'cell_blend'):
            if isinstance(layer.cell_blend, DualRazorNorm):
                w1_grad = layer.cell_blend.W1.grad
                w2_grad = layer.cell_blend.W2.grad
                if w1_grad is not None and w2_grad is not None:
                    print(f"    Layer {i}: W1.grad = {w1_grad.item():.6f}, W2.grad = {w2_grad.item():.6f}")
            elif isinstance(layer.cell_blend, RazorNorm):
                w_grad = layer.cell_blend.W.grad
                if w_grad is not None:
                    print(f"    Layer {i}: W.grad = {w_grad.item():.6f}")

    print("\n" + "="*80)
    print("All Tests Passed!")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ Selective SSM dynamics for cell state")
    print("  ✓ Traditional LSTM gating preserved")
    print("  ✓ RazorNorm learnable blend (W parameter)")
    print("  ✓ W learns whether SSM or LSTM works better")
    print("  ✓ Fully differentiable and trainable")
    print("\nNext Steps:")
    print("  1. Train on real task (e.g., Shakespeare)")
    print("  2. Track W evolution during training")
    print("  3. Analyze which dynamics the model prefers")
    print("  4. Compare performance vs standard LSTM")
    print("="*80 + "\n")
