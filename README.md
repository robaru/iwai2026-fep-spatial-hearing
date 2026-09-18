# Framing Spatial Hearing within the Free Energy Principle

A hierarchical model of perceptual learning in human sound localisation, cast as trial-by-trial belief revision under the free energy principle (FEP). The listener is described by a generative model of localisation responses whose beliefs — lateral precision, polar precision and hemifield disambiguation — are revised after every trial by natural-gradient descent on variational free energy, driven by visual feedback. Fitted to the localisation-training data of Majdak, Goupell & Laback (2010), the model captures group-average learning trajectories and separates three learning components with distinct timescales.

This repository contains the code that produces the results, figures and supplementary analyses of:

> Barumerli, R., Meyer-Kahlen, N., & Picinali, L. (2026). **Framing Spatial Hearing within the Free Energy Principle: A Hierarchical Model of Perceptual Learning.** *International Workshop on Active Inference (IWAI 2026)*, extended abstract, poster + spotlight presentation.

![Group-average behavioural trajectories (manual pointing, n = 5): lateral dispersion, polar dispersion and hemifield error rate over training trials; solid lines are the empirical means with ±1 SE ribbons, dashed lines the FEP model.](figures/fig_behavioral.png)

*Group-average behavioural trajectories, manual-pointing group (n = 5). Solid: empirical mean ± 1 SE; dashed: FEP model. Left to right: lateral dispersion (SD of lateral error), polar dispersion (SD of polar error excluding hemifield errors, cos²α_t-weighted) and hemifield error rate (cos²α_t-weighted). Produced by `MAIN_fep_simulation.ipynb` (`figures/fig_behavioral.pdf`).*

## Contents

| File | What it does |
|---|---|
| `MAIN_fep_simulation.ipynb` | Main analysis. States the model equations, loads the data, estimates the baseline beliefs from Experiment 1, fits the learning rates per subject, and produces Figure 3 of the paper (`figures/fig_behavioral.pdf`, `figures/fig_param_avg.pdf`) together with the goodness-of-fit numbers. Outputs are stored in the notebook. |
| `robustness_analysis.ipynb` | Supplementary analyses of Appendix A.3: held-out model comparison against reduced models, robustness of the flat-polar-precision result to the cos² scaling, and a parametric bootstrap for the learning rates. Reproduces Table 1 of the paper and `figures/fig_reviewer.pdf`. |
| `data/download_data.py` | Downloads `data.mat` from the AMT auxiliary data and verifies its SHA-256 checksum. The data file itself is not part of the repository. |
| `figures/` | The figures as they appear in the paper. |
| `environment.yml` | Conda environment with pinned versions. |

## Setup

```bash
conda env create -f environment.yml
conda activate iwai2026-fep
python data/download_data.py
```

The download script fetches `https://amtoolbox.org/amt-1.0.0/auxdata/majdak2010/data.mat` (about 1.2 MB) into `data/` and checks that its SHA-256 is

```
d3df2bd84b565b7e55833dd40d21c7090b70714473bf54e92186c8b38082cee6
```

## Reproducing the paper

**Figure 3 and the R² values** — run `MAIN_fep_simulation.ipynb` top to bottom (about 5 minutes; the learning-rate fits use all cores through `joblib`). The notebook prints the fitted learning rates per subject and the group-average R² for lateral dispersion, polar dispersion and hemifield error rate (0.80, 0.72, 0.86 in the paper).

**Table 1 and Appendix A.3** — run `robustness_analysis.ipynb` (about 20 minutes on 14 cores with the default 100 bootstrap replicates per subject; set the environment variable `N_BOOT` to a smaller value for a quick check). The notebook prints the held-out comparison, the cos² robustness table and Table 1 in the paper's layout, and saves `figures/fig_reviewer.pdf`.

The learning-rate optimiser (BADS) is stochastic, so refitted learning rates and derived numbers reproduce the paper up to optimiser noise; the conclusions do not depend on it.

## Model summary

Directions are expressed in interaural-polar coordinates φ = (α, β): lateral angle α along the interaural axis, polar angle β around the cone of confusion. Subscripts *t* and *r* denote target and response. The listener model factorises as


p(φ_r | φ_t, θ) = VM(α_r; α_t, κ_L) · [ w · VM(β_r; β_t, κ̃_P) + (1 − w) · VM(β_r; π − β_t, κ̃_P) ],

where VM is the von Mises distribution, κ̃_P = κ_P cos²(α) accounts for the collapse of the cone of confusion at lateral angles, and the beliefs are θ = (κ_L, κ_P, w): lateral precision, polar precision and the probability of resolving the correct hemifield.

With a Laplace approximation and a flat prior the variational free energy reduces to the negative log-likelihood, and beliefs are revised by natural-gradient descent, θ_{i+1} = θ_i − η I(θ_i)⁻¹ ∇F_i, with the Fisher information I(κ) = A′(κ) for a von Mises concentration (A(κ) = I₁(κ)/I₀(κ)) and I(ψ) = w(1 − w) for the logit ψ of the hemifield weight. The three resulting update rules, one per belief, are stated in the notebooks and in Appendix A of the paper. Initial beliefs θ₀ are fitted per subject on the last 100 trials of the no-feedback block; the three learning rates (η_L, η_P, η_w) are the only free parameters of the learning model and are fitted by Bayesian adaptive direct search (BADS).

### On the lateral angle in the cos² factor

The cone of confusion enters the model twice. At perception time the listener has no access to the true lateral angle, so the generative model of the response conditions on the inferred cone, α_r. At update time the visual feedback has revealed the true direction, so the free energy and the κ_P update are evaluated with the true cone, α_t — the same factor used in the likelihood with which the experimenter fits θ₀ and the learning rates, and in the cos²-weighted behavioural metrics. The code implements this convention: `cos2 = cos(lat_targ)**2` in the update and in the fitting likelihood.

## Data

The data are those of Majdak, P., Goupell, M. J., & Laback, B. (2010). *3-D localization of virtual sound sources: Effects of visual environment, pointing method, and training.* Attention, Perception, & Psychophysics, 72(2), 454–469. https://doi.org/10.3758/APP.72.2.454. They belong to the original authors and are distributed as auxiliary data of the [Auditory Modeling Toolbox](https://amtoolbox.org); this repository only downloads them. The paper uses the five listeners of the manual-pointing group; Experiment 1 (no feedback) provides the baseline beliefs and Experiment 2 (visual feedback) the learning trajectories.

## Licence

The code is released under the MIT licence (see `LICENSE`). The data are not covered by this licence.
