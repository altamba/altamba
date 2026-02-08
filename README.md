# ALTAMBA: Alternating Mamba with Peristaltic Normalization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18521311.svg)](https://doi.org/10.5281/zenodo.18521311)

**Learnable Destructive Interference via The Razor Blending Operation**

ALTAMBA is a dual-path Transformer+SSM architecture that runs both paths in parallel at every layer, blending them through learned destructive and constructive interference. It achieves **8-9% validation loss improvement** over parameter-matched Jamba baselines across three scales (402M, 1.08B, 1.78B).

## Key Ideas

- **The Razor** -- a learnable blending operation `(c - W) * LN(x_denoiser) + (c' - W') * x_main` with per-layer parameters and output scaling that discovers optimal interference coefficients automatically.
- **Peristaltic Normalization** -- alternating Post-LayerNorm (variance clamping for precise denoising) and Pre-LayerNorm (flexible variance for gradient flow) with phase-alternating output scaling (+1.5x / -1.0x).
- **Bounded-dt fix** for Mamba-2 -- replacing unbounded softplus with sigmoid-bounded dt, preserving native CUDA kernels via `dt_softplus=False`.

## Results

| Scale | Baseline Val | ALTAMBA Val | Improvement |
|-------|-------------|-------------|-------------|
| 402M  | 3.1866      | 2.8886      | 9.35%       |
| 1.08B | 2.9771      | 2.6974      | 9.40%       |
| 1.78B | 2.4427      | 2.2554      | 7.66%       |

Trained on Common Pile (arXiv subset) with GPT-4 tokenizer (cl100k_base).

## Files

| File | Description |
|------|-------------|
| `transformer_ssm_denoise.py` | Core ALTAMBA architecture (TransformerSSMDenoise) |
| `razor_mamba_lstm.py` | DualRazorNorm blending operation |
| `mamba2_dt_fix.py` | Bounded-dt fix for Mamba-2 |
| `train_dual.py` | Training script (side-by-side baseline vs ALTAMBA) |
| `altamba_paper.tex` | Paper source |
| `ALTAMBA.pdf` | Compiled paper |
| `figures/` | Publication figures extracted from checkpoint |

## Quick Start

```python
from transformer_ssm_denoise import TransformerSSMDenoise

model = TransformerSSMDenoise(
    vocab_size=100288,
    d_model=2560,
    n_layers=12,
    n_heads=8,
    dropout=0.2,
    mode="dual",                      # "dual", "transformer", "ssm", "jamba"
    denoiser_scale_ssm=0.75,          # bottleneck compression
    denoiser_scale_transformer=0.75,
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
