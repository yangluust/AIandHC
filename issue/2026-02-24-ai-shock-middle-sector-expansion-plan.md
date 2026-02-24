# Plan (2026-02-24): Expanded middle sector component in anticipated period-2 AI shock

## Goal
Add a new subsubsection at the end of the subsection **“The Effects of an Anticipated Period-2 AI Shock”** in `d:/AIandHC/Main.tex`.
On top of the baseline AI shock in Eq. \eqref{eq:xAI} (productivity changes governed by \(\gamma\)), introduce an additional anticipated period-2 component: **the middle sector expands to the right**, i.e. the high-sector cutoff increases from \(h_H\) to \(h_H'>h_H\).

## Placement in `Main.tex`
- Insert **after** the existing subsubsection “Effects on saving”.
- Insert **before** `\subsection{Limitations of the two-period model}`.

## New period-2 mapping (definition)
Define period-2 sectoral productivity under “AI + expanded middle sector” as:

\[
x^{AI+}(h')=
\begin{cases}
1-\lambda+\gamma\lambda, & h'<h_M,\\
1, & h_M<h'<h_H',\\
1+\lambda+\gamma\lambda, & h'>h_H'.\\
\end{cases}
\]

Timing convention: the cutoff shift is **period 2 only**, so period-1 sector mapping remains as in \eqref{eq:x}. Setting \(h_H'=h_H\) nests back to \eqref{eq:xAI}.

## Mechanism to emphasize (what changes vs \(\gamma\)-only)
### Human capital investment
- The higher cutoff shifts feasibility boundaries for reaching the high sector:
  \[
  \underline{h}_H'(y)=\frac{h_H'-ye_H}{1-\delta},\qquad
  \overline{h}_H'(y)=\frac{h_H'-ye_L}{1-\delta}.
  \]
- **Primary effect is regime reassignment**: some households move from “stay high without extra effort” (under cutoff \(h_H\)) to part-time/full-time learner regimes (under cutoff \(h_H'\)), or become unable to reach the high sector even with \(e_H\).
- **Conditional on a given learner regime**, the relevant \(z\)-cutoff formulas are the **same as in the \(\gamma\)-only case**; \(h_H'\) changes which regime applies (and thus which cutoff comparison is relevant) for a larger/smaller set of \((h,y)\) states.

### Labor supply
Keep the paper’s decomposition:
- **Via income (composition)**: households downgraded from high to middle absent extra effort (\(h'(0)\in(h_H,h_H')\)) have lower anticipated period-2 income than under \(\gamma\)-only, tending to **raise** period-1 labor supply.
- **Via full-time training (composition)**: if more households switch into/intensify full-time training to reach \(h_H'\), period-1 labor supply falls mechanically.
- Net effect is heterogeneous and driven by regime reassignment.
- **Cutoff note**: conditional on a given learner regime (relative to \(h_H'\)), the within-regime \(z\)-cutoff formulas governing work/training choices are unchanged from the \(\gamma\)-only case; \(h_H'\) acts mainly by shifting which regime applies.

### Saving
Benchmark against the existing saving decomposition using \(\Delta(x,a;t)\) and \(\Delta_H(x,a;t)\):
- Downgrading absent extra effort implies lower anticipated period-2 income relative to \(\gamma\)-only, attenuating the saving effects previously attributed to higher future income for high-sector households.
- Households investing to remain high load on the same channels as before:
  - part-time upskilling \(\Rightarrow\) \(\Delta(x,a;t)\),
  - full-time upskilling \(\Rightarrow\) \(\Delta(x,a;t)\) plus \(\Delta_H(x,a;t)\).
- Key difference vs \(\gamma\)-only is again **composition**: more households move from “stay high” into “upskill to remain high,” so the existing \(\Delta\)/\(\Delta_H\) mechanisms apply to more households.
- **Cutoff note**: conditional on a given regime (stay / upskill part-time / upskill full-time), the \(z\)-cutoff formulas underpinning these cases are the same as under \(\gamma\)-only; \(h_H'\) changes saving mainly through regime reassignment (which households fall into each case).

