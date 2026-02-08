"""
Dual Model Training: Baseline vs Denoising (Side-by-side)
==========================================================

Trains both models simultaneously on the same batches:
1. Baseline Transformer (no denoising)
2. Transformer + SSM Denoising

Perfect comparison - same data, same steps, same everything.
Only difference: the denoising architecture.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import glob
import tiktoken
from transformer_ssm_denoise import TransformerSSMDenoise
from torch.optim.lr_scheduler import CosineAnnealingLR

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# GPT-4 Tokenizer Setup
# =============================================================================
print("⏳ Initializing GPT-4 Tokenizer (cl100k_base)...")
enc = tiktoken.get_encoding("cl100k_base")
vocab_size = ((enc.n_vocab + 63) // 64) * 64  # Round to 100288
print(f"✅ Tokenizer ready. Vocab size: {vocab_size}\n")

# Hyperparameters
batch_size = 8         # Large batch for big GPU
block_size = 256
max_iters = 15000
eval_interval = 25
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50

# Model architecture
d_model = 2560
n_layers = 12
n_heads = 8
dropout = 0.2

# Baseline configuration
baseline_type = "jamba"  # "transformer", "ssm", or "jamba" (1:7 attention-to-Mamba ratio)
match_parameters = True  # Adjust baseline d_model to match dual model parameters

# Denoising configuration - REVERSED (SSM main + Transformer denoiser)
init_w1 = 1.5   # Transformer: -50% (denoiser)
init_w2 = -0.5  # SSM: 110% (main signal)

# Scaling configuration for large models
use_scaling = True  # Enable learnable output scaling (ReZero-style, starts at 1.0)
denoiser_scale_ssm = 0.75          # Scale down SSM denoiser (odd layers: 1,3,5,7,9)
denoiser_scale_transformer = 0.75  # Scale down Transformer denoiser (even layers: 0,2,4,6,8)

# ALTAMBA-Sparse: Disable odd-layer SSMs (50% SSM compute savings)
use_odd_ssm = True  # Set False for sparse mode (empirically validated at 2B: W1≈1.5 → SSM contrib ≈ 0%)

# Gradient checkpointing (for Jamba baseline memory savings)
use_gradient_checkpointing_baseline = True  # Enable for Jamba baseline to save memory

# Mixed precision
use_amp = True
gradient_accumulation_steps = 8

# Dataset
pile_path = os.path.join(script_dir, "pile")

# Checkpointing
save_checkpoints = True
checkpoint_interval = 900
checkpoint_dir_baseline = os.path.join(script_dir, "checkpoints_baseline")
checkpoint_dir_denoising = os.path.join(script_dir, "checkpoints_denoising")
os.makedirs(checkpoint_dir_baseline, exist_ok=True)
os.makedirs(checkpoint_dir_denoising, exist_ok=True)

torch.manual_seed(1337)

print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")


class PileDataset:
    """Streaming dataset for The Pile."""
    def __init__(self, pile_dir, block_size):
        self.pile_dir = pile_dir
        self.block_size = block_size

        # Find all JSON files
        self.json_files = []
        for item in os.listdir(pile_dir):
            item_path = os.path.join(pile_dir, item)
            if os.path.isdir(item_path):
                files = glob.glob(os.path.join(item_path, "*.json"))
                self.json_files.extend(files)
            elif item.endswith('.json'):
                self.json_files.append(item_path)

        self.json_files.sort(reverse=True)
        print(f"Found {len(self.json_files)} JSON files in {pile_dir}")

        self.lines_per_cycle = 409
        self.reload_interval = 160

        self.train_pool = []
        self.val_pool = []
        self.last_reload_step = -1

        self.current_file_idx = 0
        self.current_file_handle = None

        self._open_next_file()
        self._read_next_chunk(skip_cycles=0)

        print(f"✅ Pools ready: Train {len(self.train_pool):,} tokens, Val {len(self.val_pool):,} tokens\n")

    def _open_next_file(self):
        if self.current_file_handle:
            self.current_file_handle.close()

        if self.current_file_idx >= len(self.json_files):
            self.current_file_idx = 0

        filepath = self.json_files[self.current_file_idx]
        print(f"📂 Opening file {self.current_file_idx + 1}/{len(self.json_files)}")

        self.current_file_handle = open(filepath, 'r', encoding='utf-8')
        self.current_file_idx += 1

    def _read_next_chunk(self, skip_cycles=0):
        import json

        train_tokens = []
        val_tokens = []
        lines_read = 0

        while lines_read < self.lines_per_cycle:
            line = self.current_file_handle.readline()

            if not line:
                self._open_next_file()
                continue

            try:
                data = json.loads(line)
                text = data.get('text', '')
                tokens = enc.encode(text)

                if lines_read % 10 == 0:
                    val_tokens.extend(tokens)
                else:
                    train_tokens.extend(tokens)

                lines_read += 1
            except:
                continue

        self.train_pool = train_tokens
        self.val_pool = val_tokens

    def get_batch(self, split, batch_size, global_step):
        cycle_num = global_step // self.reload_interval
        if cycle_num > self.last_reload_step:
            print(f"\n🔄 Reloading data (step {global_step}, cycle {cycle_num})...")
            self._read_next_chunk(skip_cycles=cycle_num)
            self.last_reload_step = cycle_num

        pool = self.train_pool if split == 'train' else self.val_pool

        if len(pool) < self.block_size + 1:
            x = torch.zeros(batch_size, self.block_size, dtype=torch.long, device=device)
            y = torch.zeros(batch_size, self.block_size, dtype=torch.long, device=device)
            return x, y

        max_start = len(pool) - self.block_size - 1
        starts = torch.randint(0, max_start, (batch_size,))

        x = torch.stack([torch.tensor(pool[i:i+self.block_size], dtype=torch.long) for i in starts])
        y = torch.stack([torch.tensor(pool[i+1:i+self.block_size+1], dtype=torch.long) for i in starts])

        return x.to(device), y.to(device)


def calculate_matched_d_model(vocab_size, d_model_dual, n_layers, n_heads, dropout, baseline_type, init_w1, init_w2, use_scaling_param=True, denoiser_scale_ssm_param=1.0, denoiser_scale_transformer_param=1.0, use_odd_ssm_param=True):
    """
    Calculate the d_model for baseline to match dual-path parameter count.

    Uses binary search to find the baseline d_model that gives approximately
    the same total parameters as the dual-path model.
    """
    # First, count parameters in dual model
    temp_dual = TransformerSSMDenoise(
        vocab_size=vocab_size,
        d_model=d_model_dual,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model_dual * 4,
        dropout=dropout,
        init_w1=init_w1,
        init_w2=init_w2,
        use_denoising=True,
        baseline_type="dual",
        use_scaling=use_scaling_param,
        denoiser_scale_ssm=denoiser_scale_ssm_param,
        denoiser_scale_transformer=denoiser_scale_transformer_param,
        use_odd_ssm=use_odd_ssm_param
    )
    target_params = sum(p.numel() for p in temp_dual.parameters())
    del temp_dual

    # Binary search for matching d_model
    low, high = d_model_dual // 2, d_model_dual * 2
    best_d_model = d_model_dual
    best_diff = float('inf')

    for _ in range(20):  # 20 iterations should be enough
        mid = (low + high) // 2
        # Mamba2 requires: (d_model * expand) % headdim == 0
        # With expand=2 and headdim=64: d_model % 32 == 0
        mid = (mid // 32) * 32

        temp_baseline = TransformerSSMDenoise(
            vocab_size=vocab_size,
            d_model=mid,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=mid * 4,
            dropout=dropout,
            use_denoising=False,
            baseline_type=baseline_type,
            use_scaling=use_scaling_param
        )
        baseline_params = sum(p.numel() for p in temp_baseline.parameters())
        del temp_baseline

        diff = abs(baseline_params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_d_model = mid

        if baseline_params < target_params:
            low = mid + 32
        else:
            high = mid - 32

        # If we're within 1% of target, that's good enough
        if diff / target_params < 0.01:
            break

    return best_d_model, target_params


class TransformerModel(nn.Module):
    """Wrapper for Transformer models."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        use_denoising: bool,
        init_w1: float = 0.1,
        init_w2: float = 1.5,
        baseline_type: str = "transformer",
        use_scaling: bool = True,
        denoiser_scale_ssm: float = 1.0,
        denoiser_scale_transformer: float = 1.0,
        use_odd_ssm: bool = True,
        use_gradient_checkpointing: bool = False
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
            baseline_type=baseline_type,
            use_scaling=use_scaling,
            denoiser_scale_ssm=denoiser_scale_ssm,
            denoiser_scale_transformer=denoiser_scale_transformer,
            use_odd_ssm=use_odd_ssm,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.use_denoising = use_denoising

    def forward(self, x, targets=None):
        logits = self.model(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss


@torch.no_grad()
def estimate_loss(model, dataset, global_step):
    """Estimate loss on train and val."""
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split, batch_size, global_step)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss = model(X, Y)
            else:
                _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


def save_checkpoint(model, optimizer, scaler, step, loss, filename):
    """Save model checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filename)


if __name__ == "__main__":
    print("="*80)
    print("DUAL TRAINING: Baseline vs Denoising (Side-by-Side)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  batch_size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"  baseline_type: {baseline_type.upper()}")

    # Load dataset
    print("\n" + "="*80)
    print("Loading The Pile dataset...")
    print("="*80 + "\n")
    dataset = PileDataset(pile_path, block_size)

    # Create BOTH models
    print("\n" + "="*80)
    print("Creating BOTH models...")
    print("="*80 + "\n")

    # Calculate matched d_model for baseline if requested
    if match_parameters:
        if denoiser_scale_ssm < 1.0 or denoiser_scale_transformer < 1.0:
            print(f"Calculating baseline d_model to match dual-path parameters (SSM={denoiser_scale_ssm}, Transformer={denoiser_scale_transformer})...")
        else:
            print("Calculating baseline d_model to match dual-path parameters...")
        d_model_baseline, target_params = calculate_matched_d_model(
            vocab_size, d_model, n_layers, n_heads, dropout,
            baseline_type, init_w1, init_w2, use_scaling, denoiser_scale_ssm, denoiser_scale_transformer, use_odd_ssm
        )
        print(f"Baseline d_model adjusted: {d_model} → {d_model_baseline} (target: {target_params/1e6:.2f}M params)\n")
    else:
        d_model_baseline = d_model

    model_baseline = TransformerModel(
        vocab_size, d_model_baseline, n_layers, n_heads, dropout,
        use_denoising=False,
        baseline_type=baseline_type,
        use_scaling=use_scaling,
        denoiser_scale_ssm=1.0,
        denoiser_scale_transformer=1.0,
        use_odd_ssm=True,
        use_gradient_checkpointing=use_gradient_checkpointing_baseline
    ).to(device)

    model_denoising = TransformerModel(
        vocab_size, d_model, n_layers, n_heads, dropout,
        use_denoising=True,
        init_w1=init_w1,
        init_w2=init_w2,
        use_scaling=use_scaling,
        denoiser_scale_ssm=denoiser_scale_ssm,
        denoiser_scale_transformer=denoiser_scale_transformer,
        use_odd_ssm=use_odd_ssm
    ).to(device)

    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_denoising = sum(p.numel() for p in model_denoising.parameters())

    print(f"Baseline model ({baseline_type.upper()}):  {params_baseline/1e6:.2f}M parameters")
    print(f"Denoising model (DUAL):  {params_denoising/1e6:.2f}M parameters")

    # Show denoiser breakdown if scaling is enabled
    if denoiser_scale_ssm < 1.0 or denoiser_scale_transformer < 1.0:
        print(f"  → SSM denoiser scale (odd layers):         {denoiser_scale_ssm:.2f}x ({int(d_model * denoiser_scale_ssm)} dims)")
        print(f"  → Transformer denoiser scale (even layers): {denoiser_scale_transformer:.2f}x ({int(d_model * denoiser_scale_transformer)} dims)")
        avg_scale = (denoiser_scale_ssm + denoiser_scale_transformer) / 2
        print(f"  → Estimated savings: ~{(1.0 - avg_scale) * 25:.1f}% from full-size denoisers")

    if match_parameters:
        diff_pct = abs(params_baseline - params_denoising) / params_denoising * 100
        print(f"Parameter match:         {diff_pct:.2f}% difference")
    else:
        print(f"Overhead:                +{(params_denoising-params_baseline)/1e6:.2f}M ({100*(params_denoising-params_baseline)/params_baseline:.1f}%)")

    # Optimizers
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=learning_rate)
    optimizer_denoising = torch.optim.AdamW(model_denoising.parameters(), lr=learning_rate)

    scheduler_baseline = CosineAnnealingLR(
    optimizer_baseline,
    T_max=max_iters,
    eta_min=learning_rate * 0.1)
    
    scheduler_denoising = CosineAnnealingLR(
    optimizer_denoising,
    T_max=max_iters,
    eta_min=learning_rate * 0.1)

    scaler_baseline = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    scaler_denoising = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    # Training loop
    print("\n" + "="*80)
    print("Training BOTH models simultaneously...")
    print("="*80 + "\n")

    model_baseline.train()
    model_denoising.train()

    best_val_baseline = float('inf')
    best_val_denoising = float('inf')

    for step in range(max_iters):
        # Evaluation
        if step % eval_interval == 0 or step == max_iters - 1:
            losses_baseline = estimate_loss(model_baseline, dataset, step)
            losses_denoising = estimate_loss(model_denoising, dataset, step)

            # Calculate improvement
            improvement = ((losses_baseline['val'] - losses_denoising['val']) / losses_baseline['val']) * 100

            print(f"Step {step:5d}")
            print(f"  Baseline:  Train {losses_baseline['train']:.4f} | Val {losses_baseline['val']:.4f}")
            print(f"  Denoising: Train {losses_denoising['train']:.4f} | Val {losses_denoising['val']:.4f}", end="")

            # Show W values
            if hasattr(model_denoising.model, 'get_razor_weights'):
                weights = model_denoising.model.get_razor_weights()
                if weights:
                    # Handle both scalar and group-based weights
                    def get_mean(w):
                        """Get mean value whether w is scalar or tensor."""
                        if isinstance(w, (int, float)):
                            return w
                        else:  # Tensor
                            return w.mean().item()

                    avg_w1 = sum(get_mean(w[1]) for w in weights) / len(weights)
                    avg_w2 = sum(get_mean(w[2]) for w in weights) / len(weights)

                    # Average output_scale if present
                    if len(weights[0]) > 3 and weights[0][3] is not None:
                        avg_scale = sum(w[3] for w in weights if w[3] is not None) / len(weights)
                        print(f" | W1: {avg_w1:.3f}, W2: {avg_w2:.3f}, scale: {avg_scale:.3f}")
                    else:
                        print(f" | W1: {avg_w1:.3f}, W2: {avg_w2:.3f}")

                    # Show per-layer values every 100 steps
                    if step % 100 == 0:
                        print(f"  Per-layer W values:")
                        for weight_data in weights:
                            layer_idx, w1, w2 = weight_data[0], weight_data[1], weight_data[2]
                            output_scale = weight_data[3] if len(weight_data) > 3 else None

                            w1_mean = get_mean(w1)
                            w2_mean = get_mean(w2)
                            trans_contrib = (1 - w1_mean) * 100
                            ssm_contrib = (1 - w2_mean) * 100

                            scale_str = f", scale={output_scale:.3f}" if output_scale is not None else ""
                            print(f"    Layer {layer_idx}: W1={w1_mean:+.3f} ({trans_contrib:+6.1f}%), W2={w2_mean:+.3f} ({ssm_contrib:+6.1f}%){scale_str}")

            print(f"  → Improvement: {improvement:+.2f}%")
            print()

            # Save best models
            if losses_baseline['val'] < best_val_baseline:
                best_val_baseline = losses_baseline['val']

            if losses_denoising['val'] < best_val_denoising:
                best_val_denoising = losses_denoising['val']
                if save_checkpoints:
                    save_checkpoint(model_denoising, optimizer_denoising, scaler_denoising,
                                  step, best_val_denoising,
                                  os.path.join(checkpoint_dir_denoising, "best_model.pt"))

        # Get THE SAME batch for both models
        X, Y = dataset.get_batch('train', batch_size, step)

        # Train BASELINE
        optimizer_baseline.zero_grad()
        for _ in range(gradient_accumulation_steps):
            X_batch, Y_batch = dataset.get_batch('train', batch_size, step)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss = model_baseline(X_batch, Y_batch)
                loss = loss / gradient_accumulation_steps
                scaler_baseline.scale(loss).backward()
            else:
                _, loss = model_baseline(X_batch, Y_batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()

        if use_amp:
            scaler_baseline.unscale_(optimizer_baseline)
            torch.nn.utils.clip_grad_norm_(model_baseline.parameters(), 1.0)
            scaler_baseline.step(optimizer_baseline)
            scaler_baseline.update()
            scheduler_baseline.step()
        else:
            torch.nn.utils.clip_grad_norm_(model_baseline.parameters(), 1.0)
            optimizer_baseline.step()

        # Train DENOISING
        optimizer_denoising.zero_grad()
        for _ in range(gradient_accumulation_steps):
            X_batch, Y_batch = dataset.get_batch('train', batch_size, step)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss = model_denoising(X_batch, Y_batch)
                loss = loss / gradient_accumulation_steps
                scaler_denoising.scale(loss).backward()
            else:
                _, loss = model_denoising(X_batch, Y_batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()

        if use_amp:
            scaler_denoising.unscale_(optimizer_denoising)
            torch.nn.utils.clip_grad_norm_(model_denoising.parameters(), 1.0)
            scaler_denoising.step(optimizer_denoising)
            scaler_denoising.update()
            scheduler_denoising.step()
        else:
            torch.nn.utils.clip_grad_norm_(model_denoising.parameters(), 1.0)
            optimizer_denoising.step()

        # Periodic checkpoints
        if save_checkpoints and (step + 1) % checkpoint_interval == 0:
            save_checkpoint(model_denoising, optimizer_denoising, scaler_denoising,
                          step, losses_denoising['train'],
                          os.path.join(checkpoint_dir_denoising, f"checkpoint_step_{step+1}.pt"))

    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Baseline:  Val Loss = {best_val_baseline:.4f}")
    print(f"Denoising: Val Loss = {best_val_denoising:.4f}")
    final_improvement = ((best_val_baseline - best_val_denoising) / best_val_baseline) * 100
    print(f"\nFinal Improvement: {final_improvement:+.2f}%")
    print("="*80)
