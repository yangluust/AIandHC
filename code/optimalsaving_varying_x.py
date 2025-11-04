import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.95
y = 1.0
sigma = 0.72

# Helper terms from the model
A = np.exp(sigma**2 / 2.0)                        # A = E[z2] = e^{σ^2/2}
K = np.exp(sigma**2) * (np.exp(sigma**2) - 1.0)   # K = e^{σ^2}(e^{σ^2}-1)

# Second-order approximate saving policy:
# s*(λ; x) = [β(x+y) - (1+λ)x A]/(1+β)
#          + ( (1+λ)^2 x^2 K ) / ( β * [ (x+y) + (1+λ)x A ] )
def s_star(lmbda, x):
    t = 1.0 + lmbda
    term_CE = (beta * (x + y) - t * x * A) / (1.0 + beta)
    term_prec = (t**2 * x**2 * K) / (beta * ((x + y) + t * x * A))
    return term_CE + term_prec

# Grid for lambda
lam_grid = np.linspace(-0.2, 0.4, 300)

# Types
x_L = 0.5
x_H = 2.3

# Compute policy curves
s_L = np.array([s_star(lmbda, x_L) for lmbda in lam_grid])
s_H = np.array([s_star(lmbda, x_H) for lmbda in lam_grid])

# Create figure with matplotlib settings (no LaTeX for now)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.figure(figsize=(8, 4.5))

# Plot low-x curve (orange)
plt.plot(
    lam_grid, s_L,
    label=fr"$x_L = {x_L}$",
    linewidth=2,
    color="tab:orange"
)

# Plot high-x curve (blue)
plt.plot(
    lam_grid, s_H,
    label=fr"$x_H = {x_H}$",
    linewidth=2,
    color="tab:blue"
)

# Axis labels, title, grid, legend
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$s^*(x,y;\lambda)$")
plt.title(r"Optimal Saving vs. $\lambda$: Low-$x$ vs. High-$x$ Types")
plt.grid(True)

plt.legend(loc="lower left", frameon=True)

# Annotate points at λ = 0 and λ = 0.2 for both types
for x_val, color, label in [
    (x_L, "tab:orange", "low $x$"),
    (x_H, "tab:blue",   "high $x$")
]:
    for lam_mark in [0.0, 0.2]:
        sval = s_star(lam_mark, x_val)
        plt.plot(
            [lam_mark], [sval],
            marker="o",
            color=color
        )
        # Place annotation above for low-x, below for high-x
        if x_val == x_H:
            # Place below
            ha = "center"
            va = "top"
            y_offset = -0.03 * (plt.ylim()[1] - plt.ylim()[0])
        else:
            # Place above
            ha = "center"
            va = "bottom"
            y_offset = 0.03 * (plt.ylim()[1] - plt.ylim()[0])
        plt.text(
            lam_mark,
            sval + y_offset,
            f"{label}, λ = {lam_mark:.1f}, {sval:.3f}",
            fontsize=12,
            ha=ha,
            va=va,
            color=color
        )

plt.tight_layout()

# Save as PDF for LaTeX inclusion
plt.savefig('../figure/optimalsaving_varying_x.pdf', 
            bbox_inches='tight', 
            pad_inches=0.1,
            dpi=300,
            format='pdf')

# Also save as PNG for preview
plt.savefig('../figure/optimalsaving_varying_x.png', 
            bbox_inches='tight', 
            pad_inches=0.1,
            dpi=300,
            format='png')

plt.show()
