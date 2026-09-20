# One-Bit Stochastically Quantized Random Features in SGD

**Statistical Analysis of Stochastically Quantized Random Features in SGD for Memory-Constrained Kernel Learning**
Master's thesis — MSc Industrial and Applied Mathematics (Data Science), Université Grenoble Alpes / Grenoble INP – Ensimag.
Research carried out at GIPSA-Lab under the supervision of Antoine Chatalic and Simon Barthelmé. Defended September 2026.

## Overview

Random Fourier features (RFFs) reduce kernel methods from $O(n^2)$ memory to $O(nM)$, but storing $nM$ floats can still be too much on memory-constrained systems. This work studies least-squares regression solved by mini-batch SGD on RFFs **stochastically rounded to one bit per coordinate**:

$$
\varphi^q_M(x,\gamma)_j = \pm\sqrt{2/M} \quad \text{w.p. } \tfrac12\big(1 \pm \cos(\omega_j^\top x + \xi_j)\big),
\qquad \mathbb{E}_\gamma[\varphi^q_M(x,\gamma)] = \varphi_M(x).
$$

Rounding perturbs the covariance operator by a diagonal term, $C^q_M = C_M + \bar D$ with $\|\bar D\|_{op} \le 2/M$, which is propagated through the SGD recursion.

## Main result

Building on Carratino, Rudi & Rosasco (2018), the excess risk of the quantized estimator satisfies

$$
\mathbb{E}_{J,\gamma}\|f^q_T - f _{\mathcal H}\|^2_{L^2(\rho_X)}
\;\lesssim\; \sigma^2_{\mathcal H}\,\Xi_T
+ \frac{U^2}{M^2}\,\Theta_T^2
+ \mathbb{E}_J\|f_T - f_{\mathcal H}\|^2_{L^2(\rho_X)} ,
$$

where the variance term is of order $\mu \log T / b$ and the bias term of order $\mu T / M^2$.

With $\mu T \asymp \sqrt n$ and $M = \tilde O(\sqrt n)$, this gives $\tilde O(n^{-1/2})$: one bit per coordinate preserves the minimax rate (Caponnetto & De Vito, 2007) **with the same number of features as full precision**. The bound relies on a spectral assumption on the mini-batch operators, verified in the population case and left open in general.

## Experiments

Datasets: **CASP** and **SUPERCONDUCT** (OpenML), Gaussian and Matérn ($\nu \in \{1/2, 3/2\}$) kernels, single-pass and multi-pass schedules.

| Estimator | Features | Storage |
|---|---|---|
| FP | full precision, recomputed | — |
| SQ | stochastic 1-bit, noise redrawn each step (analysed) | — |
| FQ | stochastic 1-bit, noise frozen | $nM$ bits |
| DQ | deterministic sign quantization | $nM$ bits |

Findings:
- SQ is ~10% worse than FP at $M=10$; the gap closes by $M \simeq 10^3$.
- FQ and SQ agree within 0.01 test MSE in both regimes, although FQ is outside the analysis.
- DQ is competitive with, and often better than, FP (it targets a larger RKHS).
- At $M = 10^4$, the stored feature matrix goes from **320 MB to 10 MB**.

## Repository structure

```
thesis/        LaTeX sources and PDF
src/           feature maps, quantizers, SGD estimators
experiments/   scripts reproducing the figures of Chapter 4
```

## Citation

```bibtex
@mastersthesis{bortolasi2026quantized,
  author = {Bortolasi, Matteo},
  title  = {Statistical Analysis of Stochastically Quantized Random Features in SGD for Memory-Constrained Kernel Learning},
  school = {Universit{\'e} Grenoble Alpes, Grenoble INP -- Ensimag},
  year   = {2026}
}
```
