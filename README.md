# ALTAMBA: Alternating Mamba with Peristaltic Normalization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18521311.svg)](https://doi.org/10.5281/zenodo.18521311)

**Learnable Destructive Interference via The Razor Blending Operation**

ALTAMBA is a dual-path Transformer+SSM architecture that runs both paths in parallel at every layer, blending them through learned destructive and constructive interference. It achieves **8-9% validation loss improvement** over parameter-matched Jamba baselines across three scales (402M, 1.08B, 1.78B).

## Key Ideas

- **The Razor** -- a learnable blending operation `(c - W) * LN(x_denoiser) + (c' - W') * x_main` with per-layer parameters and output scaling that discovers optimal interference coefficients automatically.
- **Peristaltic Normalization** -- alternating Post-LayerNorm (variance clamping for precise denoising) and Pre-LayerNorm (flexible variance for gradient flow) with phase-alternating output scaling (+1.5x / -1.0x).
- **Bounded-dt fix** for Mamba-1 and Mamba-2 -- replacing unbounded softplus with sigmoid-bounded dt, preventing the dt death spiral while preserving CUDA kernels.

## Results

| Scale | Baseline Val | ALTAMBA Val | Improvement |
|-------|-------------|-------------|-------------|
| 402M  | 3.1866      | 2.8886      | 9.35%       |
| 1.08B | 2.9771      | 2.6974      | 9.40%       |
| 1.78B | 2.4427      | 2.2554      | 7.66%       |

Trained on Common Pile (arXiv subset) with GPT-4 tokenizer (cl100k_base).

## Model Weights

Pre-trained 1.78B checkpoint (fp16): [ameritusweb/altamba-1.78b on Hugging Face](https://huggingface.co/ameritusweb/altamba-1.78b)

## Demos

This repository includes two demo configurations:

### 1.78B (Mamba-2)

The original ALTAMBA demo at 1.78B parameters using Mamba-2 for all SSM layers. Located in `1.78B/`.

- `d_model=2560`, 12 layers, 8 heads
- Mamba-2 with bounded-dt fix (sigmoid-bounded, `dt_softplus=False`)
- Jamba baseline (1:7 attention-to-Mamba ratio)

### 418M (Mamba-1)

A smaller 418M parameter demo using Mamba-1 (selective scan) for the dual-path SSM layers, while the Jamba baseline still uses Mamba-2. Located in `418M/`.

- `d_model=1024`, 12 layers, 8 heads
- Mamba-1 for ALTAMBA dual-path SSM layers (more expressive selective scan)
- Mamba-2 for the Jamba baseline
- Bounded-dt fix for both Mamba-1 and Mamba-2
- Configurable skip connections over even layers (`use_even_skip`)

Mamba-1's original selective mechanism with input-dependent B, C, and full `d_inner`-dimensional dt provides greater expressiveness for the denoising role compared to Mamba-2's structured state space duality (SSD).

### DualRazorNorm differences

The two demos use different Razor constants and output scale initializations:

| | 1.78B | 418M |
|---|---|---|
| Even layer (Post-LN) | `(1.4 - W1) * LN(denoiser) + (0.6 - W2) * main` | `(1.0 - W1) * LN(denoiser) + (1.0 - W2) * main` |
| Odd layer (Pre-LN) | `(1.1 - W1) * LN(denoiser) + (0.5 - W2) * main` | `(1.5 - W1) * LN(denoiser) + (0.5 - W2) * main` |
| Output scale init | `+1.5` (even) / `-1.0` (odd) | `+1.0` (even) / `+1.0` (odd) |

The 1.78B demo uses asymmetric constants and phase-alternating output scaling (+1.5x / -1.0x) tuned for Mamba-2. The 418M demo uses symmetric constants with neutral output scaling, which pairs better with Mamba-1's selective scan.

## Files

Each demo directory contains the same four files:

| File | Description |
|------|-------------|
| `transformer_ssm_denoise.py` | Core ALTAMBA architecture (TransformerSSMDenoise) |
| `dual_razor_norm.py` | DualRazorNorm blending operation |
| `mamba2_dt_fix.py` | Bounded-dt fix for Mamba-1 and Mamba-2 |
| `train_dual.py` | Training script (side-by-side baseline vs ALTAMBA) |

Additional files at root:

| File | Description |
|------|-------------|
| `altamba_paper.tex` | Paper source |
| `ALTAMBA.pdf` | Compiled paper |
| `figures/` | Publication figures extracted from checkpoint |

## Quick Start

### Training

Each demo trains ALTAMBA side-by-side against a parameter-matched Jamba baseline on the same batches. Place your data in a `pile/` directory next to the scripts (JSON files with a `"text"` field per line), then run:

```bash
# 1.78B demo (Mamba-2)
cd 1.78B
python train_dual.py

# 418M demo (Mamba-1)
cd 418M
python train_dual.py
```

Configuration (hyperparameters, model size, dataset path, etc.) is set at the top of each `train_dual.py`. Key options in the 418M demo:

| Option | Default | Description |
|--------|---------|-------------|
| `use_mamba1` | `True` | Use Mamba-1 selective scan for dual-path SSM layers |
| `use_even_skip` | `False` | Add skip connections over even layers |
| `denoiser_scale_ssm` | `0.75` | Bottleneck compression for SSM denoiser |
| `denoiser_scale_transformer` | `0.75` | Bottleneck compression for Transformer denoiser |
| `baseline_type` | `"jamba"` | Baseline architecture (`"transformer"`, `"ssm"`, or `"jamba"`) |
| `match_parameters` | `True` | Adjust baseline d_model to match ALTAMBA parameter count |

### Using the model directly

**1.78B (Mamba-2):**
```python
from transformer_ssm_denoise import TransformerSSMDenoise

model = TransformerSSMDenoise(
    vocab_size=100288,
    d_model=2560,
    n_layers=12,
    n_heads=8,
    dropout=0.2,
    mode="dual",
    denoiser_scale_ssm=0.75,
    denoiser_scale_transformer=0.75,
)
```

**418M (Mamba-1):**
```python
from transformer_ssm_denoise import TransformerSSMDenoise

model = TransformerSSMDenoise(
    vocab_size=100288,
    d_model=1024,
    n_layers=12,
    n_heads=8,
    dropout=0.2,
    mode="dual",
    denoiser_scale_ssm=0.75,
    denoiser_scale_transformer=0.75,
    use_mamba1=True,
)
```

## Citation

```bibtex
@misc{seto2026altamba,
    title={ALTAMBA: Alternating Mamba with Peristaltic Normalization},
    author={Scott Seto},
    year={2026},
    doi={10.5281/zenodo.18521311},
}
```

## License

Apache 2.0
