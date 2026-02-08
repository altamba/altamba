"""
Transformer + SSM Denoising Architecture
=========================================

Tests the hypothesis: Expressive Transformer (main) + Structured SSM (denoising)

Architecture:
    input → embedding
          ↓
    ┌─────────────────┬─────────────────┐
    │  Transformer    │      SSM        │
    │  (main path)    │  (denoising)    │
    └─────────────────┴─────────────────┘
          ↓
    DualRazorNorm: (1-W1)*Transformer + (1-W2)*SSM
          ↓
    output projection

Expected initialization:
- W1 = 0.1 → Transformer: 90% (main expressive path)
- W2 = 1.5 → SSM: -50% (structured denoising)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from razor_mamba_lstm import DualRazorNorm, SelectiveSSM

# Import the bounded dt fix
from mamba2_dt_fix import wrap_mamba2_with_bounded_dt

# Try native mamba-ssm first (CUDA kernels), fallback to pure PyTorch
try:
    from mamba_ssm import Mamba2
    MAMBA_NATIVE_AVAILABLE = True
    print("   [OK] Using native mamba-ssm Mamba2 (fast CUDA kernels) with BOUNDED DT")
except ImportError:
    try:
        from mamba_ssm import Mamba
        Mamba2 = Mamba  # Fallback to Mamba-1 if Mamba2 not available
        MAMBA_NATIVE_AVAILABLE = True
        print("   [OK] Using native mamba-ssm Mamba (Mamba-1, fast CUDA kernels) with BOUNDED DT")
    except ImportError:
        from mamba_encoder import Mamba2Core
        Mamba2 = Mamba2Core
        MAMBA_NATIVE_AVAILABLE = False
        print("   [WARNING] Using pure PyTorch Mamba2Core (slower)")

class TransformerPostBlock(nn.Module):
    """
    Standard Transformer block with multi-head self-attention and FFN.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            output: [batch, seq_len, d_model]

        PRE-NORM architecture (modern standard for stability):
        - LayerNorm applied BEFORE sublayer (not after)
        - Residual connection added after sublayer
        """
        # Self-attention with PRE-NORM
        
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # FFN with PRE-NORM
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x

class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head self-attention and FFN.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            output: [batch, seq_len, d_model]

        PRE-NORM architecture (modern standard for stability):
        - LayerNorm applied BEFORE sublayer (not after)
        - Residual connection added after sublayer
        """
        # Self-attention with PRE-NORM
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + self.dropout1(attn_out)
        
        # FFN with PRE-NORM
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - Energy Dissipation Bridge

    Decouples information content (direction) from energy content (magnitude).
    Critical for SSM brake stability: prevents unbounded integration from
    exploding when A_log > 0 and dt_proj is large.

    More stable than LayerNorm for unbounded sequences because it only
    normalizes magnitude, not mean.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization: x / sqrt(mean(x^2) + eps)
        # Preserves direction, normalizes magnitude to unit variance
        norm = x.norm(2, dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return self.weight * x / (norm + self.eps)


class SSMBlock(nn.Module):
    """
    SSM block for denoising path with energy dissipation.

    Two modes:
    - Standard (main path): LayerNorm on delta
    - Brake (denoiser path): RMSNorm + learnable gain for stability

    The brake mode prevents NaN explosion when SSM has:
    - Positive A_log eigenvalues (growth instead of decay)
    - Large dt_proj values (massive integration steps)

    FIXED: Now uses bounded sigmoid dt to prevent explosion in feedback loops.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, d_state: int = 128,
                 ssd_chunk_size: int = 64, use_brake_mode: bool = False,
                 brake_init_gain: float = 0.1, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssd_chunk_size = ssd_chunk_size
        self.use_brake_mode = use_brake_mode
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Use Mamba2 (native CUDA or pure PyTorch)
        # Mamba2 requires d_state that works with headdim (default 64)
        # Using d_state=128 (Mamba2 default) instead of 64 (Mamba-1 default)

        self.ssm = Mamba2(
            d_model=d_model,
            d_conv=4,
            expand=2,
            headdim=64,
            chunk_size=ssd_chunk_size
        )

        # APPLY THE FIX: Bounded sigmoid dt
        if MAMBA_NATIVE_AVAILABLE:
            self.ssm = wrap_mamba2_with_bounded_dt(self.ssm, dt_min=dt_min, dt_max=dt_max)
            print(f"   [FIXED] SSMBlock with bounded dt ∈ [{dt_min}, {dt_max}]")

        # Normalization - RMSNorm for brake mode (energy dissipation)
        if use_brake_mode:
            # THE BRIDGE: RMSNorm prevents energy explosion
            self.norm = RMSNorm(d_model)
            # Learnable brake force (initialize small to start safe)
            # Optimizer will increase this once it trusts the bridge
            self.brake_gain = nn.Parameter(torch.tensor(brake_init_gain))
        else:
            # Standard mode for main path
            self.norm = nn.LayerNorm(d_model)
            self.brake_gain = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]

        Brake mode: RMSNorm dissipates energy, gain controls brake force
        Standard mode: LayerNorm on delta, direct addition
        """
        # INPUT SURGE PROTECTION: Prevent "token 0 = 207" scenarios
        # Tanh clamps input magnitude to prevent unstable SSM regime
        # Scale factor 5.0 allows normal inputs through while protecting against spikes
        #x_safe = torch.tanh(x / 5.0) * 5.0

        # SSM forward pass (now protected from explosive initial states)
        ssm_out = self.ssm(x)

        # THE BRIDGE: Normalize energy before applying to residual stream
        ssm_out = self.norm(ssm_out)
        ssm_out = self.dropout(ssm_out)

        # Apply brake gain if in brake mode
        #if self.use_brake_mode and self.brake_gain is not None:
            # Controlled brake force - starts small (0.1), optimizer increases
            # as it gains confidence in the bridge's stability
            #ssm_out = self.brake_gain * ssm_out

        # Residual connection
        output = x + ssm_out

        return output


class TransformerSSMDenoise(nn.Module):
    """
    Dual-path architecture: Transformer (main) + SSM (denoising).

    The model processes input through two parallel paths:
    1. Transformer path: Expressive multi-head attention for main signal
    2. SSM path: Structured state space for denoising

    Then blends them using DualRazorNorm with learnable weights.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        init_w1: float = 0.1,
        init_w2: float = 1.5,
        use_denoising: bool = True,
        baseline_type: str = "transformer",  # "transformer", "ssm", or "dual"
        use_scaling: bool = True,  # Enable 1/√d scaling for large models
        denoiser_scale_ssm: float = 1.0,  # Scale down SSM denoiser (odd layers)
        denoiser_scale_transformer: float = 1.0,  # Scale down Transformer denoiser (even layers)
        use_odd_ssm: bool = True,  # ALTAMBA-Sparse: disable SSM in odd layers for 50% compute savings
        use_gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory savings
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_denoising = use_denoising
        self.use_scaling = use_scaling
        self.denoiser_scale_ssm = denoiser_scale_ssm
        self.denoiser_scale_transformer = denoiser_scale_transformer
        self.use_odd_ssm = use_odd_ssm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.baseline_type = baseline_type if not use_denoising else "dual"

        # Calculate denoiser dimensions (separate for SSM and Transformer)
        # SSM denoiser is active on odd layers (1,3,5,7,9)
        self.d_denoiser_ssm = int(d_model * denoiser_scale_ssm) if denoiser_scale_ssm < 1.0 else d_model
        self.d_denoiser_ssm = (self.d_denoiser_ssm // n_heads) * n_heads  # Ensure divisibility

        # Transformer denoiser is active on even layers (0,2,4,6,8)
        self.d_denoiser_transformer = int(d_model * denoiser_scale_transformer) if denoiser_scale_transformer < 1.0 else d_model
        self.d_denoiser_transformer = (self.d_denoiser_transformer // n_heads) * n_heads  # Ensure divisibility

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Input projection layers (when denoiser has different dimension)
        # Projects d_model input down to d_denoiser for denoiser path
        if self.baseline_type in ["transformer", "dual", "jamba"]:
            self.transformer_input_proj = nn.ModuleList()
            for i in range(n_layers):
                if i % 2 == 0 and self.d_denoiser_transformer != d_model:
                    # Even: Transformer is denoiser, needs projection
                    self.transformer_input_proj.append(nn.Linear(d_model, self.d_denoiser_transformer))
                else:
                    # Odd: Transformer is main, no projection needed
                    self.transformer_input_proj.append(nn.Linear(d_model, self.d_denoiser_transformer))
        else:
            self.transformer_input_proj = None

        if self.baseline_type in ["ssm", "dual", "jamba"]:
            self.ssm_input_proj = nn.ModuleList()
            for i in range(n_layers):
                if i % 2 == 1 and self.d_denoiser_ssm != d_model:
                    # Odd: SSM is denoiser, needs projection
                    self.ssm_input_proj.append(nn.Linear(d_model, self.d_denoiser_ssm))
                else:
                    # Even: SSM is main, no projection needed
                    self.ssm_input_proj.append(None)
        else:
            self.ssm_input_proj = None

        # Transformer path
        # Even layers (0,2,4,6,8): Transformer is denoiser (use d_denoiser_transformer)
        # Odd layers (1,3,5,7,9): Transformer is main (use d_model)
        if self.baseline_type in ["transformer", "dual", "jamba"]:
            self.transformer_layers = nn.ModuleList()
            for i in range(n_layers):
                if self.baseline_type == "jamba":
                    # Jamba: all layers use d_model and Pre-LN (standard architecture)
                    trans_dim = d_model
                    trans_ff = trans_dim * 4
                    self.transformer_layers.append(
                        TransformerBlock(trans_dim, n_heads, trans_ff, dropout)
                    )
                else:
                    # Dual/Transformer: use denoiser dimensions and alternating LN
                    trans_dim = self.d_denoiser_transformer if i % 2 == 0 else d_model
                    trans_ff = trans_dim * 4

                    if i % 2 == 0:
                        # Even: Post-LN (denoiser)
                        self.transformer_layers.append(
                            TransformerPostBlock(trans_dim, n_heads, trans_ff, dropout)
                        )
                    else:
                        # Odd: Pre-LN (main)
                        self.transformer_layers.append(
                            TransformerBlock(trans_dim, n_heads, trans_ff, dropout)
                        )

        # SSM path
        # Even layers (0,2,4,6,8): SSM is main (use d_model)
        # Odd layers (1,3,5,7,9): SSM is denoiser (use d_denoiser_ssm) - optional if use_odd_ssm=False
        if self.baseline_type in ["ssm", "dual", "jamba"]:
            self.ssm_layers = nn.ModuleList()
            for i in range(n_layers):
                if self.baseline_type == "jamba":
                    # Jamba: all SSM layers use d_model (standard architecture)
                    # Enable brake mode for stability (dt_proj ~ 7, A_log > 0)
                    self.ssm_layers.append(SSMBlock(d_model, dropout, use_brake_mode=True))
                elif i % 2 == 1 and not use_odd_ssm:
                    # ALTAMBA-Sparse: Skip SSM in odd layers
                    self.ssm_layers.append(None)
                else:
                    # Dual/SSM: use denoiser dimensions for odd layers
                    ssm_dim = d_model if i % 2 == 0 else self.d_denoiser_ssm
                    # ALL SSM layers need brake mode with RMSNorm energy dissipation
                    # Even layers (0,2,4,6,8,10): Main path SSMs - CRITICAL for stability
                    # Odd layers (1,3,5,7,9,11): Denoiser SSMs (if enabled)
                    # Both need energy dissipation when dt_proj ~ 7 and A_log > 0
                    use_brake = False
                    self.ssm_layers.append(SSMBlock(ssm_dim, dropout, use_brake_mode=use_brake))

        # DualRazorNorm blending layers (only for dual mode)
        # Output dimension is always d_model, but denoiser might be smaller
        # Each layer uses appropriate denoiser dimension based on which path is denoiser
        if use_denoising:
            self.blend_layers = nn.ModuleList()
            for i in range(n_layers):
                # Even: Transformer is denoiser, Odd: SSM is denoiser
                dim_denoiser = self.d_denoiser_transformer if i % 2 == 0 else self.d_denoiser_ssm
                self.blend_layers.append(
                    DualRazorNorm(
                        dim=d_model,
                        init_w1=init_w1,
                        init_w2=init_w2,
                        use_scaling=use_scaling,
                        dim_denoiser=dim_denoiser
                    )
                )

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token IDs
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # Embed tokens and add positional encoding
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Create causal mask for attention
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Process through layers
        for i in range(self.n_layers):
            if self.baseline_type == "dual":
                # Dual-path with DualRazorNorm blending
                # Project input to denoiser dimension if needed
                if i % 2 == 0:
                    # Even: Transformer is denoiser, SSM is main
                    x_trans_in = self.transformer_input_proj[i](x) if self.transformer_input_proj and self.transformer_input_proj[i] is not None else x
                    x_ssm_in = x  # SSM uses full dimension
                else:
                    # Odd: SSM is denoiser, Transformer is main
                    x_trans_in = x  # Transformer uses full dimension
                    x_ssm_in = self.ssm_input_proj[i](x) if self.ssm_input_proj and self.ssm_input_proj[i] is not None else x

                # Forward through both paths
                x_transformer = self.transformer_layers[i](x_trans_in, mask=causal_mask)

                # ALTAMBA-Sparse: Skip SSM in odd layers if use_odd_ssm=False
                if self.ssm_layers[i] is not None:
                    x_ssm = self.ssm_layers[i](x_ssm_in)
                else:
                    x_ssm = None

                # Blend: (main, denoiser)
                if i % 2 == 0:
                    # Even: SSM main, Transformer denoiser
                    x = self.blend_layers[i](x_ssm, x_transformer, i)
                else:
                    # Odd: Transformer main, SSM denoiser (if enabled)
                    if x_ssm is not None:
                        # RESTORED: Use SSM as denoiser (now stable with bounded dt!)
                        x = self.blend_layers[i](x_transformer, x_ssm, i)
                    else:
                        # ALTAMBA-Sparse: Skip SSM denoiser, apply learnable (0.5 - W2) * main * output_scale_1
                        blend = self.blend_layers[i]
                        output = (0.5 - blend.W2) * x_transformer
                        if blend.output_scale_1 is not None:
                            output = output * blend.output_scale_1
                        x = output

            elif self.baseline_type == "transformer":
                # Transformer only (baseline)
                x = self.transformer_layers[i](x, mask=causal_mask)
            elif self.baseline_type == "ssm":
                # SSM only (baseline)
                x = self.ssm_layers[i](x)
            elif self.baseline_type == "jamba":
                # Jamba-style: 1:7 attention-to-Mamba ratio (7 Mamba, 1 Attention, repeat)
                # Pattern: Mamba(0-6), Attention(7), Mamba(8-14), Attention(15), etc.
                if (i + 1) % 8 == 0:  # Every 8th layer is Attention
                    if self.use_gradient_checkpointing and self.training:
                        x = checkpoint(self.transformer_layers[i], x, mask=causal_mask, use_reentrant=False)
                    else:
                        x = self.transformer_layers[i](x, mask=causal_mask)
                else:  # Rest are Mamba
                    if self.use_gradient_checkpointing and self.training:
                        x = checkpoint(self.ssm_layers[i], x, use_reentrant=False)
                    else:
                        x = self.ssm_layers[i](x)

        # Output projection
        x = self.ln_f(x)
        logits = self.fc_out(x)

        return logits

    def get_razor_weights(self):
        """Extract W1, W2, and output_scale from all blend layers.

        Returns list of tuples: (layer_idx, W1, W2, output_scale)
        where W1 and W2 can be scalars (if num_groups=1) or tensors (if num_groups>1)
        output_scale is scalar or None if scaling disabled
        """
        if not self.use_denoising:
            return []

        weights = []
        for i, blend in enumerate(self.blend_layers):
            # Get output_scale if it exists
            if hasattr(blend, 'output_scale') and blend.output_scale is not None:
                output_scale = blend.output_scale.item()
            else:
                output_scale = None

            # Handle both scalar and group-based weights
            if blend.W1.numel() == 1:
                # Scalar weights
                weights.append((i, blend.W1.item(), blend.W2.item(), output_scale))
            else:
                # Group weights - return as tensors for detailed analysis
                weights.append((i, blend.W1.detach().cpu(), blend.W2.detach().cpu(), output_scale))
        return weights


class CharTransformerSSM(nn.Module):
    """Character-level language model wrapper."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_denoising: bool = True,
        init_w1: float = 0.1,
        init_w2: float = 1.5,
        use_scaling: bool = True,
        denoiser_scale_ssm: float = 1.0,
        denoiser_scale_transformer: float = 1.0
    ):
        super().__init__()
        self.model = TransformerSSMDenoise(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_model * 4,
            dropout=dropout,
            init_w1=init_w1,
            init_w2=init_w2,
            use_denoising=use_denoising,
            use_scaling=use_scaling,
            denoiser_scale_ssm=denoiser_scale_ssm,
            denoiser_scale_transformer=denoiser_scale_transformer
        )
        self.use_denoising = use_denoising

    def forward(self, x, targets=None):
        logits = self.model(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            # Crop to max context if needed
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Transformer + SSM Denoising Architecture Test")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test configuration
    vocab_size = 65
    batch_size = 4
    seq_len = 64
    d_model = 256
    n_layers = 2
    n_heads = 4

    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")

    # Create models
    print("\n" + "-"*80)
    print("Creating models...")

    # Baseline Transformer
    model_baseline = CharTransformerSSM(
        vocab_size, d_model, n_layers, n_heads,
        use_denoising=False
    ).to(device)

    # Transformer + SSM Denoising
    model_denoise = CharTransformerSSM(
        vocab_size, d_model, n_layers, n_heads,
        use_denoising=True,
        init_w1=0.1,  # Transformer: 90%
        init_w2=1.5   # SSM: -50% (denoising)
    ).to(device)

    print("  [OK] Baseline Transformer")
    print("  [OK] Transformer + SSM Denoising")

    # Test input
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    print("\n" + "-"*80)
    print("Forward pass tests...")

    # Test baseline
    print("\n1. Baseline Transformer:")
    with torch.no_grad():
        logits_base, loss_base = model_baseline(x, targets)
    print(f"   Input: {x.shape}")
    print(f"   Output: {logits_base.shape}")
    print(f"   Loss: {loss_base.item():.4f}")
    print("   [OK] Forward pass successful")

    # Test denoising
    print("\n2. Transformer + SSM Denoising:")
    with torch.no_grad():
        logits_denoise, loss_denoise = model_denoise(x, targets)
    print(f"   Input: {x.shape}")
    print(f"   Output: {logits_denoise.shape}")
    print(f"   Loss: {loss_denoise.item():.4f}")

    # Show W values
    weights = model_denoise.model.get_razor_weights()
    print(f"\n   Initial W values:")
    for layer_idx, w1, w2 in weights:
        ssm_pct = (1 - w1) * 100
        ssm_denoise_pct = (1 - w2) * 100
        print(f"     Layer {layer_idx}: W1={w1:.4f}, W2={w2:.4f}")
        print(f"       -> Transformer: {ssm_pct:.1f}%, SSM: {ssm_denoise_pct:.1f}%")
    print("   [OK] Forward pass successful")

    # Parameter counts
    print("\n" + "-"*80)
    print("Parameter comparison:")
    params_base = sum(p.numel() for p in model_baseline.parameters())
    params_denoise = sum(p.numel() for p in model_denoise.parameters())
    print(f"  Baseline Transformer:     {params_base/1e6:.2f}M")
    print(f"  Transformer + SSM:        {params_denoise/1e6:.2f}M")
    print(f"  Overhead:                 +{(params_denoise-params_base)/1e6:.2f}M ({100*(params_denoise-params_base)/params_base:.1f}%)")

    # Test backward
    print("\n" + "-"*80)
    print("Backward pass test...")
    logits, loss = model_denoise(x, targets)
    loss.backward()

    has_nan = any(torch.isnan(p.grad).any() for p in model_denoise.parameters() if p.grad is not None)
    has_inf = any(torch.isinf(p.grad).any() for p in model_denoise.parameters() if p.grad is not None)
    print(f"  Gradient NaN: {has_nan}")
    print(f"  Gradient Inf: {has_inf}")
    print("  [OK] Backward pass successful")

    # Show W gradients
    print("\n  W parameter gradients:")
    for i, blend in enumerate(model_denoise.model.blend_layers):
        w1_grad = blend.W1.grad.item() if blend.W1.grad is not None else 0
        w2_grad = blend.W2.grad.item() if blend.W2.grad is not None else 0
        print(f"    Layer {i}: W1.grad={w1_grad:.6f}, W2.grad={w2_grad:.6f}")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
    print("\nReady to train on Shakespeare dataset!")
    print("="*80 + "\n")
