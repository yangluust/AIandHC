# Memo (2026-02-24): Additional AI component contracts access to high sector

## Goal
Record the implemented revision in `d:/AIandHC/Main.tex` that adds an extra anticipated period-2 AI component on top of the baseline \(\gamma\)-shock in Eq. \eqref{eq:xAI}: **AI contracts access to the high sector** by raising the high-sector cutoff from \(h_H\) to \(h_H'>h_H\) (period 2 only).

## Placement in `Main.tex`
- Insert **after** the existing subsubsection “Effects on saving”.
- Insert **before** `\subsection{Limitations of the two-period model}`.

## New period-2 mapping (definition)
Define period-2 sectoral productivity under “baseline \(\gamma\)-shock + higher high-sector cutoff” as:

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
- Increasing the high-sector cutoff in period 2 shifts the period-1 human-capital thresholds that determine whether reaching the high sector is feasible with effort \(e\in\{e_L,e_H\}\). In the current draft text, the relevant thresholds are stated to increase by \((h_H'-h_H)\frac{1}{1-\delta}\).
- **Churning (regime reassignment) is the main effect**: some households who could remain high under \(e=0\) when the cutoff is \(h_H\) may now be reclassified into the middle sector if \(h'(0)\in(h_H,h_H')\), creating incentives to invest (when feasible) to stay high; others may find \(h_H'\) unattainable even with \(e_H\) and shift toward \(e=0\).
- **Within-type cutoff claim**: conditional on a given learner type, the within-type \(z\)-cutoff formulas governing labor/investment choices are unchanged from the \(\gamma\)-only case; \(h_H'\) shifts which learner type applies.

### Labor supply
Keep the paper’s decomposition:
- **Via income (composition)**: households downgraded from high to middle absent extra effort (\(h'(0)\in(h_H,h_H')\)) have lower anticipated period-2 income than under \(\gamma\)-only, tending to **raise** period-1 labor supply.
- **Via full-time training (composition)**: if more households switch into/intensify full-time training to reach \(h_H'\), period-1 labor supply falls mechanically.
- Net effect is heterogeneous and driven by regime reassignment.
- **Cutoff note**: conditional on a given learner type, the within-type \(z\)-cutoff formulas are unchanged from the \(\gamma\)-only case; \(h_H'\) acts mainly by shifting which learner type applies.

### Saving
The current draft text frames the saving implications of the \(h_H'\) component as a byproduct of churning: changes in learner type induce changes in human-capital investment decisions and therefore change saving outcomes.
- For households who increase human capital investment due to the \(h_H'\) component, saving changes are expressed using \(\Delta(x,a;t)\), \(\Delta_H(x,a;t)\), and \(\Delta_{\text{full-time}}(x,a;t)\) (with \(t>1\)).
- For households who decrease human capital investment, saving changes are expressed using \(-\Delta(x,a;t)\) or \(-\Delta_{\text{full-time}}(x,a;t)\).
- For households whose human capital investment is unchanged, saving responses largely mirror the baseline \(\gamma\)-only shock, except for households reclassified to the middle sector absent investment (those with \(h_H\frac{1}{1-\delta}<h<h_H'\frac{1}{1-\delta}\)), whose saving change is summarized by \(\Delta(x,a;t<1)\).
- **Cutoff note**: conditional on a given learner type/regime, the underlying within-type \(z\)-cutoff formulas are unchanged from the \(\gamma\)-only case; the \(h_H'\) component affects saving mainly through reassignment across regimes.

