"""
demo_one_listener.py
====================

A step-by-step tutorial of the IWAI 2026 model ("Framing Spatial Hearing within
the Free Energy Principle"), run on ONE simulated listener instead of real data.

What happens, top to bottom:
  1. settings (the hidden listener's schedule, number of trials)
  2. the generative model: likelihood of a response given target and beliefs
  3. the belief update: free energy gradient, Fisher information, natural-gradient step
  4. simulate one listener: a hidden "true listener" whose precision and hemifield
     weight follow a prescribed schedule produces all responses (a baseline block
     and a feedback block)
  5. fit the agent: starting beliefs from the baseline block, then learning rates
     by maximum likelihood of the feedback-block responses, with beliefs rolled
     forward trial by trial
  6. behavioural metrics in 50-trial blocks (data vs. fitted model)
  7. plot

Why a hidden listener with a schedule? If the responses were drawn from the
agent's own beliefs (a closed loop), the score (prediction error) would have zero
mean and the beliefs would only diffuse as a random walk. Learning, as in the real
data, needs responses that are better than the current beliefs predict, so here
the listener improves on a fixed schedule and the agent has to catch up.

Reference implementation: MAIN_fep_simulation.ipynb (cells 4, 6, 12, 14, 17).
Equations: Appendix A of the paper (A1-A6). Where the two differ, this script
follows the notebook and says so in a comment marked "NOTEBOOK vs PAPER".

Only numpy, scipy and matplotlib. Runtime: well under a minute.

Conventions (as in the notebook and in Majdak et al. 2010):
  - interaural-polar coordinates: lateral angle alpha in [-90, 90] deg,
    polar angle beta in [-90, 270) deg (0 = front, 90 = up, 180 = back)
  - all computations in radians; degrees only for printing and plotting
  - beliefs theta = (kappa_L, kappa_P, w); w is updated through its logit psi
"""

import time
import numpy as np
import scipy.optimize
from scipy.special import ive, expit, logit
import matplotlib.pyplot as plt

t_start = time.time()


# =============================================================================
# 1. SETTINGS
# =============================================================================
rng = np.random.default_rng(9)         # one seed for everything -> reproducible
                                       # (seed 9: fitted learning rates closest to the median over seeds 1-10; see 5b)

N_BASELINE = 100     # baseline trials (no feedback); the notebook uses the last 100
                     # trials of Experiment 1 to estimate the starting beliefs
N_TRIALS   = 750     # feedback trials (the paper's Fig. 3 is truncated at 750)
BLOCK      = 50      # trials per block for the behavioural metrics
N_PRED     = 20      # simulated response sets from the fitted model (panel b)

# The hidden listener. Its parameters change over the feedback block on a
# prescribed schedule, independent of the agent's beliefs: saturating
# exponentials  x(j) = x_end + (x_start - x_end) exp(-j / tau),  j = feedback trial,
# with distinct timescales as in the paper (fast lateral, slow polar precision).
# The start values are close to the baseline fit of listener NH16
# (notebook cell 10: kappa_P = 4.31, w = 0.71, kappa_L = 8.54); the baseline block
# uses the start values. There are no "true learning rates": the hidden listener
# has a schedule, only the agent has learning rates, and those are fitted.
KL_START, KL_END, TAU_L = 8.5, 18.0, 80.0     # lateral concentration kappa_L (SD ~20 -> ~14 deg), fast
KP_START, KP_END, TAU_P = 4.3, 7.0, 400.0     # polar concentration kappa_P (SD ~30 -> ~23 deg), slow
W_START,  W_END,  TAU_W = 0.70, 0.90, 250.0   # hemifield weight w = P(correct hemifield)

# Target directions as in Majdak et al. (2010): uniform on the sphere,
# all azimuths, elevations between -30 and +80 deg.
EL_MIN, EL_MAX = np.deg2rad(-30.0), np.deg2rad(80.0)


# =============================================================================
# 2. THE GENERATIVE MODEL (likelihood of one response)
# =============================================================================
# Von Mises density:  VM(x; mu, kappa) = exp(kappa cos(x - mu)) / (2 pi I0(kappa))
# We use log I0(kappa) = log(ive(0, kappa)) + kappa, which does not overflow.
#
# Lateral (A2):  p(alpha_r | alpha_t, kappa_L) = VM(alpha_r; alpha_t, kappa_L)
# Polar   (A3):  p(beta_r | ..., kappa_P, w)  = w     VM(beta_r; beta_t,      kappa_P~)
#                                             + (1-w) VM(beta_r; pi - beta_t, kappa_P~)
#                with kappa_P~ = kappa_P cos^2(alpha).
#
# Which alpha goes into cos^2? The LISTENER perceives with the inferred cone
# alpha_r. The EXPERIMENTER, who fits the model, conditions on the true cone
# alpha_t; this is the likelihood used for theta_0, for the learning rates,
# inside the update and in the simulation of section 4 (notebook cells 4, 12,
# 14, 17). The function below is the experimenter's one.

def log_lik_trial(alpha_t, beta_t, alpha_r, beta_r, kappa_L, kappa_P, w):
    """Log-likelihood of a response (alpha_r, beta_r) to a target (alpha_t, beta_t)
    under beliefs (kappa_L, kappa_P, w). Returns (lateral part, polar part).
    Works on single trials and on arrays of trials. Angles in radians."""
    # lateral: one von Mises centred on the target
    log_p_lat = kappa_L * np.cos(alpha_r - alpha_t) - (np.log(2 * np.pi) + np.log(ive(0, kappa_L)) + kappa_L)
    # polar: two von Mises, correct hemifield (beta_t) and mirrored (pi - beta_t)
    k_eff = np.maximum(kappa_P * np.cos(alpha_t) ** 2, 1e-6)      # cone of confusion
    log_norm = np.log(2 * np.pi) + np.log(ive(0, k_eff)) + k_eff
    vm_c = np.exp(k_eff * np.cos(beta_r - beta_t) - log_norm)
    vm_m = np.exp(k_eff * np.cos(beta_r - (np.pi - beta_t)) - log_norm)
    log_p_pol = np.log(np.maximum(w * vm_c + (1 - w) * vm_m, 1e-300))
    return log_p_lat, log_p_pol


# =============================================================================
# 3. THE BELIEF UPDATE (one natural-gradient step on the free energy)
# =============================================================================
# Free energy with a flat prior and a Laplace posterior (A4):
#     F_i = -log p(response_i | target_i, theta_i) + const
# Natural-gradient step (A5):
#     theta_{i+1} = theta_i - eta * I(theta_i)^{-1} * dF_i/dtheta
# Fisher information:
#     von Mises concentration:  I(kappa) = A'(kappa) = 1 - A(kappa)^2 - A(kappa)/kappa,
#                               A(kappa) = I1(kappa)/I0(kappa) = E[cos(x - mu)]
#     logit psi of w:           I(psi) = w (1 - w)
# Because the likelihood factorises, the three updates do not interact.
#
# Polar prediction error: with kappa_P~ = kappa_P cos^2(alpha_t) the exact derivative is
#     dF/dkappa_P = -cos^2(alpha_t) [r - A(kappa_P~)],
# and the update (notebook cell 12) uses exactly this gradient: the expected
# alignment A is evaluated at the scaled concentration kappa_P~. Only the
# preconditioner keeps the unscaled Fisher information A'(kappa_P) (the exact
# Fisher of the scaled model would divide by cos^2 and up-weight lateral trials).
# The prediction error r - A(kappa_P~) is positive on average when the responses
# are more concentrated than the current belief predicts, and negative when they
# are less: kappa_P moves towards the listener's actual polar precision.

def update_beliefs(kappa_L, kappa_P, psi, alpha_t, beta_t, alpha_r, beta_r, eta_L, eta_P, eta_w):
    """One trial of learning after visual feedback (notebook cell 12, Eqs. A6-A8).
    Takes the current beliefs (kappa_L, kappa_P, psi = logit w), the target and the
    response, and the learning rates. Returns the beliefs for the next trial."""
    A_L = ive(1, kappa_L) / ive(0, kappa_L)                  # Bessel ratio A(kappa_L)
    A_P = ive(1, kappa_P) / ive(0, kappa_P)                  # Bessel ratio A(kappa_P), for the Fisher term
    fisher_L = max(1 - A_L ** 2 - A_L / kappa_L, 1e-4)       # I(kappa_L), floored as in notebook
    fisher_P = max(1 - A_P ** 2 - A_P / kappa_P, 1e-4)       # I(kappa_P)
    # lateral: prediction error = observed alignment - expected alignment
    grad_L = -(np.cos(alpha_r - alpha_t) - A_L)              # dF/dkappa_L
    kappa_L_new = np.clip(kappa_L - eta_L * grad_L / fisher_L, 0.01, 200.0)
    # polar: the two mixture components evaluated at the response
    w, cos2 = expit(psi), np.cos(alpha_t) ** 2               # true cone after feedback
    k_eff = max(kappa_P * cos2, 1e-6)
    log_norm = np.log(2 * np.pi) + np.log(ive(0, k_eff)) + k_eff
    vm_c = np.exp(k_eff * np.cos(beta_r - beta_t) - log_norm)
    vm_m = np.exp(k_eff * np.cos(beta_r - (np.pi - beta_t)) - log_norm)
    p = w * vm_c + (1 - w) * vm_m
    if p < 1e-300:                                           # response impossible: skip (notebook)
        return kappa_L_new, kappa_P, psi
    # r = responsibility-weighted cosine alignment between response and target
    r = (w * np.cos(beta_r - beta_t) * vm_c + (1 - w) * np.cos(beta_r - (np.pi - beta_t)) * vm_m) / p
    A_eff = ive(1, k_eff) / ive(0, k_eff)                    # expected alignment A(kappa_P~)
    grad_P = -cos2 * (r - A_eff)                             # dF/dkappa_P (exact gradient)
    kappa_P_new = np.clip(kappa_P - eta_P * grad_P / fisher_P, 0.01, 200.0)
    # hemifield: dF/dpsi = -w(1-w)(vm_c - vm_m)/p; dividing by I(psi) = w(1-w) cancels it
    psi_new = psi + eta_w * (vm_c - vm_m) / p
    return kappa_L_new, kappa_P_new, psi_new


# =============================================================================
# 4. SIMULATE ONE LISTENER
# =============================================================================
N_ALL = N_BASELINE + N_TRIALS

# 4a. Targets: uniform on the sphere within the elevation range.
# Uniform on a sphere = uniform azimuth and uniform sin(elevation).
az = rng.uniform(0, 2 * np.pi, N_ALL)
el = np.arcsin(rng.uniform(np.sin(EL_MIN), np.sin(EL_MAX), N_ALL))
x, y, z = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)   # x front, y left, z up
alpha_t = np.arcsin(y)                               # lateral angle
beta_t = np.arctan2(z, x)                            # polar angle in (-pi, pi]
beta_t = np.where(beta_t < -np.pi / 2, beta_t + 2 * np.pi, beta_t)   # -> [-90, 270) deg

# 4b. The hidden listener's schedule. Entry j is the listener's state on
# feedback trial j (entry N_TRIALS is the state after the last trial, for the plot).
# During the baseline block the listener stays at the start values.
j = np.arange(N_TRIALS + 1)
kL_true = KL_END + (KL_START - KL_END) * np.exp(-j / TAU_L)
kP_true = KP_END + (KP_START - KP_END) * np.exp(-j / TAU_P)
w_true = W_END + (W_START - W_END) * np.exp(-j / TAU_W)
kL_all = np.concatenate([np.full(N_BASELINE, KL_START), kL_true[:-1]])   # per trial, baseline + feedback
kP_all = np.concatenate([np.full(N_BASELINE, KP_START), kP_true[:-1]])
w_all = np.concatenate([np.full(N_BASELINE, W_START), w_true[:-1]])

# 4c. Responses. Every response, baseline and feedback, is drawn from the hidden
# listener with the generative model of section 2. The agent's beliefs play no
# role here.
# lateral response: target + von Mises noise (may exceed +-90 deg rarely; as in notebook)
alpha_r = alpha_t + rng.vonmises(0.0, kL_all)
# polar response: pick the hemifield with probability w, then add von Mises noise.
# The listener's own generative model at perception uses alpha_r; the experimenter's
# likelihood, the update and this simulation use the true cone alpha_t, as in the paper.
correct_hemi = rng.random(N_ALL) < w_all
beta_r = np.where(correct_hemi, beta_t, np.pi - beta_t) + rng.vonmises(0.0, np.maximum(kP_all * np.cos(alpha_t) ** 2, 1e-6))
beta_r = (beta_r + np.pi / 2) % (2 * np.pi) - np.pi / 2          # -> [-90, 270) deg

# Split into the two blocks. From here on we only use what an experimenter sees:
# targets and responses. The hidden schedule is kept only for the plot.
base = slice(0, N_BASELINE)
feed = slice(N_BASELINE, N_ALL)
at, bt, ar, br = alpha_t[feed], beta_t[feed], alpha_r[feed], beta_r[feed]


# =============================================================================
# 5. FIT
# =============================================================================
# 5a. Starting beliefs theta_0 from the baseline block (notebook cells 4, 10):
# plain maximum likelihood, no learning (beliefs constant over the block).
# Parametrised as (log kappa_L, log kappa_P, logit w) so the optimiser is unconstrained.
# The notebook fits lateral and polar separately with a multi-start grid; since the
# likelihood factorises a joint fit from one sensible start gives the same answer here.
def neg_log_lik_baseline(x):
    """Negative log-likelihood of the baseline responses for constant beliefs x."""
    lp_lat, lp_pol = log_lik_trial(alpha_t[base], beta_t[base], alpha_r[base], beta_r[base],
                                   np.exp(x[0]), np.exp(x[1]), expit(x[2]))
    return -np.sum(lp_lat + lp_pol)

res0 = scipy.optimize.minimize(neg_log_lik_baseline, x0=[np.log(5.0), np.log(2.0), logit(0.8)],
                               method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
KL0_FIT, KP0_FIT, W0_FIT = np.exp(res0.x[0]), np.exp(res0.x[1]), expit(res0.x[2])

print('Step 5a. Starting beliefs from the baseline block (%d trials)' % N_BASELINE)
print('            hidden    fitted')
print('  kappa_L  %7.2f   %7.2f' % (KL_START, KL0_FIT))
print('  kappa_P  %7.2f   %7.2f' % (KP_START, KP0_FIT))
print('  w        %7.3f   %7.3f' % (W_START, W0_FIT))

# 5b. Learning rates (notebook cell 14). theta_0 is fixed at the baseline fit.
# For a candidate (eta_L, eta_P, eta_w) we roll the beliefs forward over the
# OBSERVED feedback trials with the update rule, and add up the log-likelihood
# of each response under the beliefs held just before that trial's feedback.
# The fitting cost is the negative of that sum.
#
# NOTEBOOK vs SCRIPT: the notebook minimises with BADS (pybads) on the raw learning
# rates with bounds. Here we use scipy's Nelder-Mead on log learning rates, which
# keeps them positive without bounds. eta_P can still go to ~0 (log eta -> -inf).
def neg_log_lik_learning(log_etas):
    """Negative log-likelihood of the feedback-block responses for learning
    rates exp(log_etas) = (eta_L, eta_P, eta_w), beliefs rolled forward trial by trial."""
    eta_L, eta_P, eta_w = np.exp(log_etas)
    kappa_L, kappa_P, psi = KL0_FIT, KP0_FIT, logit(W0_FIT)
    nll = 0.0
    for i in range(N_TRIALS):
        lp_lat, lp_pol = log_lik_trial(at[i], bt[i], ar[i], br[i], kappa_L, kappa_P, expit(psi))
        nll -= lp_lat + lp_pol                            # predict first ...
        kappa_L, kappa_P, psi = update_beliefs(kappa_L, kappa_P, psi, at[i], bt[i], ar[i], br[i],
                                               eta_L, eta_P, eta_w)   # ... then learn
    return nll

x_start = np.log([0.005, 0.005, 0.005])                   # same start for all three rates
nll_start = neg_log_lik_learning(x_start)
nll_none = neg_log_lik_learning(np.log([1e-12, 1e-12, 1e-12]))   # "no learning" reference

res = scipy.optimize.minimize(neg_log_lik_learning, x_start, method='Nelder-Mead',
                              options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 600})
ETA_L_FIT, ETA_P_FIT, ETA_W_FIT = np.exp(res.x)
# How typical is this fit? Over seeds 1-10 the fitted learning rates had medians
# eta_L 0.019, eta_P 0.012, eta_w 0.019 (ranges 0.014-0.023, 0.004-0.015,
# 0.012-0.026), and learning always beat no learning by 58-174 nats. Seed 9 is the
# seed closest to these medians. One listener and 750 trials carry limited
# information about how fast beliefs move, and theta_0 from 100 baseline trials
# is itself noisy.

print('\nStep 5b. Learning rates by maximum likelihood (%d feedback trials)' % N_TRIALS)
print('  NLL with no learning (eta = 0)           : %9.2f' % nll_none)
print('  NLL at the start values (eta = 0.005)    : %9.2f' % nll_start)
print('  NLL at the optimum (%3d iterations)      : %9.2f' % (res.nit, res.fun))
print('  fitted learning rates (there are no true ones: the hidden listener has a schedule)')
print('  eta_L    %8.4f' % ETA_L_FIT)
print('  eta_P    %8.4f' % ETA_P_FIT)
print('  eta_w    %8.4f' % ETA_W_FIT)

# 5c. The fitted belief trajectory: same roll-forward, with the fitted values.
kL_fit = np.zeros(N_TRIALS + 1)
kP_fit = np.zeros(N_TRIALS + 1)
w_fit = np.zeros(N_TRIALS + 1)
kappa_L, kappa_P, psi = KL0_FIT, KP0_FIT, logit(W0_FIT)
for i in range(N_TRIALS):
    kL_fit[i], kP_fit[i], w_fit[i] = kappa_L, kappa_P, expit(psi)
    kappa_L, kappa_P, psi = update_beliefs(kappa_L, kappa_P, psi, at[i], bt[i], ar[i], br[i],
                                           ETA_L_FIT, ETA_P_FIT, ETA_W_FIT)
kL_fit[-1], kP_fit[-1], w_fit[-1] = kappa_L, kappa_P, expit(psi)


# =============================================================================
# 6. BEHAVIOURAL METRICS IN BLOCKS (notebook cell 6)
# =============================================================================
# 6a. Predicted responses from the fitted model: on each trial, draw a response
# from the fitted beliefs of that trial (rolled forward on the observed data, as
# in the notebook), N_PRED times, with the same sampler as in section 4.
ar_pred = at + rng.vonmises(0.0, kL_fit[:-1], size=(N_PRED, N_TRIALS))
correct = rng.random((N_PRED, N_TRIALS)) < w_fit[:-1]
br_pred = np.where(correct, bt, np.pi - bt) + rng.vonmises(0.0, np.maximum(kP_fit[:-1] * np.cos(at) ** 2, 1e-6))

# 6b. Three metrics per 50-trial block, for the data (1 set) and the model (N_PRED sets):
#   lateral dispersion : SD of the lateral error (deg)
#   polar dispersion   : cos^2(alpha_t)-weighted RMS polar error (deg) after folding
#                        hemifield errors back into the correct hemifield (beta_r -> pi - beta_r)
#   hemifield errors   : cos^2(alpha_t)-weighted % of responses with |polar error| >= 90 deg
# NOTEBOOK vs PAPER: the Fig. 3 caption says hemifield errors are "excluded" from the
# polar dispersion; the notebook (sdPol) folds them back instead. We follow the notebook.
n_blocks = N_TRIALS // BLOCK
block_mid = np.arange(n_blocks) * BLOCK + BLOCK / 2
metrics = {}
for name, a_r, b_r in [('data', ar[None, :], br[None, :]), ('model', ar_pred, br_pred)]:
    shape = (a_r.shape[0], n_blocks, BLOCK)               # (response sets, blocks, trials)
    wgt = (np.cos(at) ** 2)[:n_blocks * BLOCK].reshape(1, n_blocks, BLOCK)
    lat_err = (a_r - at + np.pi) % (2 * np.pi) - np.pi
    pol_err = (b_r - bt + np.pi) % (2 * np.pi) - np.pi
    is_hemi = np.abs(pol_err) >= np.pi / 2
    folded = np.where(is_hemi, np.pi - b_r, b_r)
    pol_err_f = (folded - bt + np.pi) % (2 * np.pi) - np.pi
    lat_err, pol_err_f, is_hemi = (v[:, :n_blocks * BLOCK].reshape(shape) for v in (lat_err, pol_err_f, is_hemi))
    sd_lat = np.rad2deg(np.std(lat_err, axis=2))
    sd_pol = np.rad2deg(np.sqrt(np.sum(wgt * pol_err_f ** 2, axis=2) / np.sum(wgt, axis=2)))
    hemi = 100 * np.sum(wgt * is_hemi, axis=2) / np.sum(wgt, axis=2)
    metrics[name] = [m.mean(axis=0) for m in (sd_lat, sd_pol, hemi)]   # average over response sets


# =============================================================================
# 7. PLOT
# =============================================================================
trials = np.arange(N_TRIALS + 1)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))

# (a) hidden listener's schedule (black) vs. the agent's beliefs rolled forward
# with the fitted learning rates (red dashed) and with no learning, eta = 0 (grey
# dotted: the beliefs stay at theta_0 from the baseline block).
panels_a = [(kL_true, kL_fit, KL0_FIT, r'$\kappa_L$ (lateral precision)'),
            (kP_true, kP_fit, KP0_FIT, r'$\kappa_P$ (polar precision)'),
            (w_true, w_fit, W0_FIT, r'$w$ (hemifield weight)')]
for ax, (true, fit, start, label) in zip(axes[0], panels_a):
    ax.plot(trials, true, color='black', lw=1.5, label='hidden listener (schedule)')
    ax.plot(trials, fit, color='tab:red', ls='--', lw=1.5, label=r'agent, fitted $\eta$')
    ax.plot(trials, np.full(len(trials), start), color='grey', ls=':', lw=1.5, label=r'agent, $\eta$ = 0')
    ax.set_title(label)
    ax.set_xlabel('Feedback trial')
axes[0, 0].set_ylabel('(a) Listener / belief')
axes[0, 2].legend(frameon=False, fontsize=8, loc='upper left')

# (b) behaviour: simulated responses (black) vs. fitted model's predicted responses (dashed)
labels_b = ['Lateral dispersion (deg)', 'Polar dispersion (deg)', 'Hemifield errors (%)']
for k, ax in enumerate(axes[1]):
    ax.plot(block_mid, metrics['data'][k], 'o-', color='black', lw=1.5, ms=4, label='hidden listener (simulated responses)')
    ax.plot(block_mid, metrics['model'][k], '--', color='tab:red', lw=1.5,
            label='fitted model (mean of %d)' % N_PRED)
    ax.set_title(labels_b[k])
    ax.set_xlabel('Feedback trial (block of %d)' % BLOCK)
axes[1, 0].set_ylabel('(b) Behaviour')
axes[1, 0].legend(frameon=False, fontsize=8, loc='best')

fig.suptitle(r'One simulated listener on a prescribed schedule (no true $\eta$).  Agent, fitted $\eta$ = (%.4f, %.4f, %.4f)'
             r'  for  ($\eta_L$, $\eta_P$, $\eta_w$)'
             % (ETA_L_FIT, ETA_P_FIT, ETA_W_FIT), fontsize=11)
fig.tight_layout()
out_png = __file__.replace('.py', '.png')
fig.savefig(out_png, dpi=150)
print('\nSaved %s  (total runtime %.1f s)' % (out_png, time.time() - t_start))
plt.show()
