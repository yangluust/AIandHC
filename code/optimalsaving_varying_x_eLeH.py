import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. Parameters
# -----------------------
beta = 0.95
y = 1.0
sigma = 0.72
xL, xH = 0.5, 2.3  # low- and high-exposure types

# convenience constants from the lognormal income process
mu = np.exp(sigma**2 / 2.0)
K = np.exp(sigma**2) * (np.exp(sigma**2) - 1.0)

def s_star(W, x, lam):
    """
    Approximate optimal saving s^*(W, x; lambda)
    given:
      W   = liquid resources in period 1,
      x   = exposure scale (also appears in risky income),
      lam = lambda, which scales future risky income.
    Formula:
      s^* = [beta*W - (1+lam)*x*mu] / (1+beta)
            + ((1+lam)^2 * x^2 * K) / (beta * (W + (1+lam)*x*mu))
    """
    t = 1.0 + lam
    numer_ce = beta * W - t * x * mu
    term_ce = numer_ce / (1.0 + beta)
    term_prec = (t**2 * x**2 * K) / (beta * (W + t * x * mu))
    return term_ce + term_prec

# -----------------------
# 2. Case A: "fixed W = x + y" for all lambda
# -----------------------
lam_grid = np.linspace(-0.05, 0.30, 200)

sL_fixed = [s_star(xL + y, xL, lam) for lam in lam_grid]
sH_fixed = [s_star(xH + y, xH, lam) for lam in lam_grid]

# -----------------------
# 3. Case B: "shrinking W"
#    At lambda=0, W = x + y
#    At lambda=0.2, W = y
# -----------------------
lam_points = np.array([0.0, 0.2])

# low-x type
sL_var_points = [
    s_star(xL + y, xL, 0.0),  # lambda=0, full resources
    s_star(y,       xL, 0.2)  # lambda=0.2, resources reduced to y
]

# high-x type
sH_var_points = [
    s_star(xH + y, xH, 0.0),  # lambda=0, full resources
    s_star(y,      xH, 0.2)   # lambda=0.2, resources reduced to y
]

# -----------------------
# 3b. Compute values at key lambda points for fixed-W case
# -----------------------
lam_mark = 0.2

# Fixed-W case at lambda=0.2
sL_fixed_0 = s_star(xL + y, xL, 0.0)
sL_fixed_02 = s_star(xL + y, xL, lam_mark)
sH_fixed_0 = s_star(xH + y, xH, 0.0)
sH_fixed_02 = s_star(xH + y, xH, lam_mark)

# -----------------------
# 4. Plot
# -----------------------
# Set up figure with LaTeX-compatible settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.ioff()  # Turn off interactive mode to ensure saving works
plt.figure(figsize=(8, 4.5))

# solid lines: fixed-W case
plt.plot(
    lam_grid, sL_fixed,
    label=r'Low $x_L$, fixed $W=x+y$',
    lw=2, alpha=0.8, color='tab:orange'
)
plt.plot(
    lam_grid, sH_fixed,
    label=r'High $x_H$, fixed $W=x+y$',
    lw=2, alpha=0.8, color='tab:blue'
)

# dashed markers: shrinking-W case
plt.plot(
    lam_points, sL_var_points,
    'o--', lw=1.5, color='tab:orange',
    markersize=6,
    label=r'Low $x_L$, $W=x+y \to y$'
)
plt.plot(
    lam_points, sH_var_points,
    'o--', lw=1.5, color='tab:blue',
    markersize=6,
    label=r'High $x_H$, $W=x+y \to y$'
)

# Add markers at lambda=0.2 for fixed-W case
plt.plot(0.0, sL_fixed_0, 'o', color='tab:orange', markersize=6)
plt.plot(lam_mark, sL_fixed_02, 'o', color='tab:orange', markersize=6)
plt.plot(0.0, sH_fixed_0, 'o', color='tab:blue', markersize=6)
plt.plot(lam_mark, sH_fixed_02, 'o', color='tab:blue', markersize=6)

# reference verticals at lambda=0 and lambda=0.2
plt.axvline(0.0, color='gray', ls=':', lw=0.8)
plt.axvline(0.2, color='gray', ls=':', lw=0.8)

# Add text annotations showing values at lambda=0 and lambda=0.2
# Get y-axis range for positioning
ylim = plt.ylim()
y_range = ylim[1] - ylim[0]

# Annotation settings
fontsize_annot = 8
offset_x = 0.02
offset_y = 0.02 * y_range

# Fixed-W, Low x_L
plt.text(0.0 + offset_x, sL_fixed_0 + offset_y, 
         rf'$\lambda=0$: ${sL_fixed_0:.3f}$',
         fontsize=fontsize_annot, color='tab:orange', 
         va='bottom', ha='left')
plt.text(lam_mark + offset_x, sL_fixed_02 + offset_y,
         rf'$\lambda=0.2$: ${sL_fixed_02:.3f}$',
         fontsize=fontsize_annot, color='tab:orange',
         va='bottom', ha='left')

# Fixed-W, High x_H
plt.text(0.0 + offset_x, sH_fixed_0 + offset_y,
         rf'$\lambda=0$: ${sH_fixed_0:.3f}$',
         fontsize=fontsize_annot, color='tab:blue',
         va='bottom', ha='left')
plt.text(lam_mark + offset_x, sH_fixed_02 + offset_y,
         rf'$\lambda=0.2$: ${sH_fixed_02:.3f}$',
         fontsize=fontsize_annot, color='tab:blue',
         va='bottom', ha='left')

# Shrinking-W, Low x_L
plt.text(0.0 - offset_x, sL_var_points[0] - offset_y,
         rf'$\lambda=0$: ${sL_var_points[0]:.3f}$',
         fontsize=fontsize_annot, color='tab:orange',
         va='top', ha='right')
plt.text(lam_mark - offset_x, sL_var_points[1] - offset_y,
         rf'$\lambda=0.2$: ${sL_var_points[1]:.3f}$',
         fontsize=fontsize_annot, color='tab:orange',
         va='top', ha='right')

# Shrinking-W, High x_H
plt.text(0.0 - offset_x, sH_var_points[0] - offset_y,
         rf'$\lambda=0$: ${sH_var_points[0]:.3f}$',
         fontsize=fontsize_annot, color='tab:blue',
         va='top', ha='right')
plt.text(lam_mark - offset_x, sH_var_points[1] - offset_y,
         rf'$\lambda=0.2$: ${sH_var_points[1]:.3f}$',
         fontsize=fontsize_annot, color='tab:blue',
         va='top', ha='right')

# axes / labels / legend
plt.xlabel(r'$\lambda$ (future-income scale)', fontsize=11)
plt.ylabel(r'Optimal saving $s^\star$', fontsize=11)
plt.title(r'Effect of increasing $\lambda$: fixed vs shrinking current resources', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True, fontsize=9, loc='best')

plt.tight_layout()

# Save as PDF for LaTeX inclusion
plt.savefig('../figure/optimalsaving_varying_x_eLeH.pdf', 
            bbox_inches='tight', 
            pad_inches=0.1,
            dpi=300,
            format='pdf')

# Also save as PNG for preview
plt.savefig('../figure/optimalsaving_varying_x_eLeH.png', 
            bbox_inches='tight', 
            pad_inches=0.1,
            dpi=300,
            format='png')

plt.show()