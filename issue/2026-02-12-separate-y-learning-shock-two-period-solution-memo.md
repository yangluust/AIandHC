# Memo (2026-02-12): Separate learning-ability shock \(y\) from earnings shock \(z\) and re-solve the two-period model

## Purpose
Record the model change and the full **two-period solution procedure** under the new setup where human-capital accumulation depends on a distinct, observed learning shock \(y\) rather than the same shock \(z\) that enters wages. This memo is intended to be read later to implement updates (not now).

## Current baseline (from `2025NovMain.tex`, around Sec. “Model Environment” and “Household Decisions in a Two-Period Model”)
- **Wages / labor income** (unchanged in proposed change):
  \[
  \text{labor income}_t = n_t \cdot w_t z_t x(h_t)
  \]
- **Human capital law of motion** (to be changed):
  \[
  h_{t+1} = z_t e_t + (1-\delta_h)h_t
  \]
- Discrete choices:
  - \(n_t\in\{0,1\}\) (indivisible labor)
  - \(e_t\in\{0,e_L,(1-n_t)e_H\}\) (work prevents choosing \(e_H\))
- Sector productivity step function (unchanged):
  \[
  x(h)=\begin{cases}
  1-\lambda, & h<h_M \\
  1, & h_M\le h<h_H \\
  1+\lambda, & h\ge h_H
  \end{cases}
  \]

## Proposed change: separate learning shock \(y\)
### New human capital accumulation
Replace the baseline law of motion with:
\[
h_{t+1} = y_t e_t + (1-\delta_h)h_t
\]
where \(y_t\) is **not** the same shock as \(z_t\) in wages.

### Stochastic processes
Keep \(z\) as currently specified (AR(1) in logs). Add \(y\) as AR(1) in logs:
\[
\ln y' = \rho_y \ln y + \varepsilon_y,\qquad \varepsilon_y\sim N(0,\sigma_y^2).
\]
Correlation choice (implementation option; must be stated explicitly in text/code later):
- Default/simple: \(\mathrm{corr}(\varepsilon_z,\varepsilon_y)=0\).
- Optional richer version: \((\varepsilon_z,\varepsilon_y)\) bivariate normal with correlation \(\varrho_{zy}\).

### Timing / information (as decided)
- In period \(t\), household **observes \((z_t,y_t)\)** before choosing \((n_t,e_t,a_{t+1})\).
- In the **two-period** model, \(y'\) is irrelevant in the terminal period because there is no further investment.

### State vector
- Full/infinite-horizon state becomes: \((a_t,h_t,z_t,y_t)\).
- Two-period (period 1) state: \((a,h,z,y)\).
- Period 2 state: \((a',h',z')\).

## Two-period model: solution procedure under separate \(y\)
Below is the procedure to mirror the paper’s two-period exposition:

### Period 2 (last period)
Same logic as baseline.
- No investment: \(e_2=0\), and no saving choice.
- Only decision: work \(n_2\in\{0,1\}\).
- Work cutoff in \(z'\) (same functional form as existing Eq. for \(\overline z(h,a)\)):
  \[
  n_2=1 \iff z' \ge \overline z(h',a').
  \]
The sector-dependent scaling still comes from \(x(h')\). The only change is that \(h'\) is now the outcome of period-1 \(y\) and \(e\), not period-1 \(z\).

### Period 1 (deterministic-\(z'\) benchmark, “works in period 2” benchmark)
To parallel the baseline paper structure, one can first:
- assume \(z'\) is known at period 1, and
- (optionally) condition on the event “will work in period 2,” as in the existing text’s intuition-building step.

Given a candidate discrete choice \((n,e)\), human capital next period is:
\[
h'(e)=y e+(1-\delta_h)h.
\]
The intertemporal budget constraint becomes:
\[
c+\frac{c'}{1+r'}=(1+r)a+n(wzx(h))+\frac{w'z'x(h'(e))}{1+r'}.
\]
Log utility delivers the same proportionality as in the baseline derivation:
\[
c'=\beta(1+r')c.
\]
Thus consumption conditional on \((n,e)\) is:
\[
c(n,e)=\frac{1}{1+\beta}\left[(1+r)a+n(wzx(h))+\frac{w'z'x(h'(e))}{1+r'}\right].
\]
The discrete-choice problem is:
\[
\max_{n,e} (1+\beta)\ln c(n,e) - \chi_n n - \chi_e e.
\]

## Critical structural implication: “learner types” become cutoffs in \(y\), not \(z\)
In the baseline text, feasibility regions (non-/slow-/fast learner) are defined by cutoffs in \(z\) because \(h'=z e + (1-\delta)h\).

With \(h'=y e+(1-\delta_h)h\), the analogous feasibility cutoffs are in \(y\).

### Cutoffs to reach middle-sector threshold \(h_M\)
Define:
\[
\underline y_M(h):=\frac{h_M-(1-\delta_h)h}{e_H},\qquad
\overline y_M(h):=\frac{h_M-(1-\delta_h)h}{e_L}.
\]
Interpretation (for current \(h\) such that reaching \(h_M\) is relevant):
- **Non-learners**: \(y<\underline y_M(h)\). Even \(e_H\) cannot reach \(h'\ge h_M\).
- **Slow learners**: \(\underline y_M(h)\le y<\overline y_M(h)\). Only \(e_H\) can reach \(h'\ge h_M\).
- **Fast learners**: \(y\ge \overline y_M(h)\). \(e_L\) is sufficient to reach \(h'\ge h_M\).

### Cutoffs to reach high-sector threshold \(h_H\)
\[
\underline y_H(h):=\frac{h_H-(1-\delta_h)h}{e_H},\qquad
\overline y_H(h):=\frac{h_H-(1-\delta_h)h}{e_L}.
\]

### Diagram implication
Existing “decision-rule diagram” that uses diagonal lines in \((h,z)\) space (baseline \(\underline z_M(h),\overline z_M(h)\)) should be reworked as:
- **Feasibility / learner-type diagram in \((h,y)\)** with diagonals \(\underline y_M,\overline y_M\) (and \(\underline y_H,\overline y_H\) in the relevant \(h\) ranges).
- Work cutoffs remain in **\(z\)**, but now their levels depend on \((h,y,a)\) through the future sector \(x(h'(e))\).

## Deriving period-1 \(z\)-cutoffs conditional on \((h,y,a)\): general reusable formulas
The baseline paper derives multiple \(z\)-cutoffs by comparing discrete options.
This approach still works. The key is to isolate how \(y\) enters via \(h'(e)\Rightarrow x(h'(e))\).

### Notation (recommended for clean derivations)
Define:
- \(h'(e)=y e+(1-\delta_h)h\)
- “non-\(z\)” resources under effort \(e\):
  \[
  A(e;y,h,a):=(1+r)a+\frac{w'z'x(h'(e))}{1+r'}
  \]
- marginal return to \(z\) if working:
  \[
  B(h):=w x(h)
  \]
Total resources:
\[
R(n,e)=A(e;y,h,a)+n\cdot B(h)\cdot z.
\]
Objective index:
\[
U(n,e)=(1+\beta)\ln R(n,e)-\chi_n n-\chi_e e.
\]

### (1) Work vs no-work cutoff holding effort fixed
Compare \((n=1,e)\) to \((n=0,e)\):
\[
z \ge z^{work}(e;y,h,a)
:=\frac{\left(\exp\left(\frac{\chi_n}{1+\beta}\right)-1\right)A(e;y,h,a)}{B(h)}.
\]
Key point: this is still a cutoff in \(z\), but it now **shifts with \(y\)** through \(A(e)\) (because \(x(h'(e))\) changes with \(y\)).

### (2) Work-with-investment vs work-without-investment cutoff (e.g. \(e_L\) vs 0)
Compare \((1,e_L)\) vs \((1,0)\):
\[
(1+\beta)\ln(A(e_L)+Bz)-\chi_e e_L
=(1+\beta)\ln(A(0)+Bz).
\]
Let \(g:=\exp\left(\frac{\chi_e e_L}{1+\beta}\right)\). Then the indifference cutoff is:
\[
z=\frac{A(e_L)-gA(0)}{(g-1)B}.
\]

### (3) Slow-learner tradeoff cutoff (don’t work + high effort vs work + no effort)
Compare \((0,e_H)\) vs \((1,0)\):
\[
(1+\beta)\ln(A(e_H))-\chi_e e_H
=(1+\beta)\ln(A(0)+Bz)-\chi_n,
\]
which implies:
\[
z=\frac{\exp\left(\frac{\chi_n-\chi_e e_H}{1+\beta}\right)A(e_H)-A(0)}{B}.
\]
This is the direct analogue of the baseline “slow learner” cutoff, except:
- the group “slow learners” is defined by **\(y\)**, and
- the cutoff’s level depends on \(y\) through \(A(e_H)\) and potentially \(A(0)\) (via \(x(h')\)).

## Mapping to the baseline text’s structure (what changes, what stays)
### What stays the same
- Period-2 work rule is still a cutoff in \(z\) scaled by sector productivity \(x(h)\).
- Discrete choice set \((n,e)\) and the “work prevents \(e_H\)” constraint.
- The “pairwise comparison \(\Rightarrow\) cutoff” logic.
- Log-utility simplifications that convert the problem to a max over \((1+\beta)\ln(\text{resources})\) minus effort costs.

### What changes mechanically
- Replace every “learning feasibility cutoff in \(z\)” (e.g. baseline \(\underline z_M(h),\overline z_M(h)\)) with the corresponding cutoff in \(y\):
  \[
  \underline z_M,\overline z_M \;\Rightarrow\; \underline y_M,\overline y_M
  \quad\text{and}\quad
  \underline z_H,\overline z_H \;\Rightarrow\; \underline y_H,\overline y_H.
  \]
- Any baseline \(z\)-cutoff that included future income terms will now depend on \(y\) via \(x(h'(e))\) because \(h'(e)=y e+(1-\delta_h)h\).

## Reintroducing uncertainty in the two-period model (implementation-ready outline)
Given “\(y\) observed at choice time” and “two-period,” forward-looking uncertainty primarily concerns \(z'\mid z\).
The general numerical approach:

1) Fix period-1 state \((a,h,z,y)\).
2) Enumerate feasible discrete actions \((n,e)\) given \(e\in\{0,e_L,(1-n)e_H\}\).
3) For each \((n,e)\), compute \(h'=h'(e)=y e+(1-\delta_h)h\) and \(x(h')\).
4) Optimize over saving \(a'\ge 0\) (1D) if saving is included in the two-period exercise; otherwise treat it as chosen as in the existing text’s benchmark.
5) For each candidate \(a'\), compute period-2 optimal work \(n'(z')=\mathbf{1}\{z'\ge \overline z(h',a')\}\).
6) Evaluate expected utility:
   \[
   \ln c + \beta \mathbb{E}_{z'|z}\left[\ln c'(z') - \chi_n n'(z')\right]-\chi_n n-\chi_e e,
   \]
   where:
   \[
   c=(1+r)a+n(wzx(h))-a',
   \quad
   c'(z')=(1+r')a'+n'(z')\cdot w' z' x(h').
   \]
7) Choose best \((n,e,a')\).

Quadrature/discretization note:
- Because \(n'(z')\) is a cutoff rule, \(c'(z')\) is piecewise affine in \(z'\) (0 below cutoff; affine above). Expectations of \(\ln c'(z')\) generally require numerical integration, but are straightforward with a discretized \(z'\) grid.

## Paper edits to make later (checklist)
- **Model environment section**: add \(y\), AR(1) specification for \(y\), and revise the human-capital law of motion to \(h'=y e+(1-\delta_h)h\).
- **Interpretation paragraph**: replace “\(z\) is also learning ability” with “\(y\) is learning ability; \(z\) is earnings productivity.”
- **Two-period section**:
  - Replace feasibility cutoffs in \(z\) (learner types) with feasibility cutoffs in \(y\).
  - Update decision-rule diagram: learner partition in \((h,y)\); keep work cutoffs in \(z\) but note they shift with \((h,y)\) through \(x(h'(e))\).
- **Risk discussion** (two-period): with \(y\) observed, “investment-return risk” is reduced relative to a setup where \(y\) is realized after choosing \(e\); heterogeneity in \(y\) matters, but the intertemporal risk channel is still mainly \(z'\).

## Implementation notes (for later coding/LaTeX edits)
- Add new parameters: \(\rho_y,\sigma_y\) (and optionally \(\varrho_{zy}\)).
- Update any routines/derivations that:
  - use \(z\) to compute next-period \(h'\) (replace with \(y\)),
  - define “learner-type cutoffs” in terms of \(z\) (replace with \(y\)),
  - build figures over \((h,z)\) for learner regions (move to \((h,y)\)).

