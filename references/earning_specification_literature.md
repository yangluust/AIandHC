Yes. Your specification,

$$
y_{ist}=w_t A_{st}\varepsilon_{it},
$$

is closely related to the **efficiency-units** and **Roy-model** traditions. Many papers combine the first two components into a sector-specific price of an efficiency unit,

$$
\widetilde w_{st}\equiv w_tA_{st},
\qquad
y_{ist}=\widetilde w_{st}\varepsilon_{ist}.
$$

Thus, papers may be algebraically equivalent to your formulation even when sectoral productivity is not displayed as a separate multiplier.

### Closest precedents

| Paper                                      | Earnings specification and relation to yours                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Faia, Kudlyak, and Shabalina (2021)**    | Closest modern incomplete-markets analogue. Their multi-sector, multi-occupation Bewley–Aiyagari model has workers with occupation-specific efficiency units $\gamma_j^o$, endogenous occupational wages $w_t^o$, and an exogenous idiosyncratic Markov productivity shock $e_t$. Labor income depends on the product of these terms, along with hours and taxes: $\xi_{j,t}^o=(1-\tau_t)w_t^o\gamma_j^o e_t n_t^o$. This closely matches your multiplicative structure once $w_tA_{st}$ is interpreted as an occupation/sector efficiency wage. ([FRBSF][9]) |
| **Dix-Carneiro (2014)**                    | Explicitly describes the model as an “equilibrium dynamic version of the Roy Model,” closely related to Heckman and Sedlacek (1985) and Lee and Wolpin (2006). Workers supply sector-specific human capital with a deterministic component depending on education, age, and sector-specific experience, plus sector-specific idiosyncratic shocks redrawn each period. Wages are equilibrium human-capital prices, so earnings are the sectoral human-capital price times worker-sector human capital. This is a benchmark structural model for trade-adjustment dynamics. ([Econometrica][10]) |
| **Lagakos and Waugh (2013)**               | The closest direct match. A worker’s income in sector $s$ is $y_i=z_s^i w_s$, where $z_s^i$ is worker-specific sectoral productivity and $w_s$ is the wage per efficiency unit. Their equilibrium sectoral wages satisfy $w_a=p_aA$ and $w_n=A$, so income is explicitly the product of individual productivity, a sectoral price, and aggregate productivity. ([MICHAEL WAUGH][1])                            |
| **Heckman and Sedlacek (1985)**            | In their general-equilibrium Roy model, earnings in sector $s$ are $w_s(i)=\pi_s t_s(i)$, where $\pi_s$ is the market price of a unit of sector-specific productive ability and $t_s(i)$ is the individual’s productivity in that sector. Setting $\pi_s=wA_s$ gives precisely your specification.                                                                                                             |
| **Lee and Wolpin (2006)**                  | Their dynamic multisector model specifies the wage offer as $\hat{w}^j=r_t^j s_{ha}^j$, where $r_t^j$ is the equilibrium rental price of a unit of skill in sector–occupation $j$, and $s_{ha}^j$ is the worker’s sector–occupation-specific skill, including an idiosyncratic shock. Sectoral technology affects the equilibrium skill rental $r_t^j$. Thus, $r_t^j$ corresponds to $w_tA_{jt}$. ([CORE][2])  |
| **Hsieh, Hurst, Jones, and Klenow (2019)** | Individual earnings in occupation $i$ are $(1-\tau^w_{ig})w_i\epsilon_i h(e,s)$: the occupation-specific wage per efficiency unit times idiosyncratic talent and human capital, adjusted for a labor-market wedge. Occupation-specific productivity $A_i$ determines $w_i$ in equilibrium; in a special case, they obtain $w_i=A_i$. ([NBER][3])                                                               |
| **Roy (1951)**                             | The foundational model behind this class of specifications. Workers possess different productivities in different occupations and select the occupation delivering the highest earnings. The paper is more conceptual than your modern multiplicative formulation, but it is the standard foundational citation for sector-specific idiosyncratic productivity and occupational selection. ([OUP Academic][4]) |

### Which papers are most useful to cite?

For a quantitative macroeconomic paper, I would primarily cite:

1. **Faia, Kudlyak, and Shabalina (2021)** for the closest modern incomplete-markets, occupation-specific-skill analogue.
2. **Dix-Carneiro (2014)** for an explicit equilibrium dynamic Roy model with sector-specific human capital prices and trade-adjustment dynamics.
3. **Lagakos and Waugh (2013)** for the clean productivity-times-sector-price decomposition.
4. **Lee and Wolpin (2006)** for a dynamic multisector implementation.
5. **Hsieh et al. (2019)** for a modern general-equilibrium allocation-of-talent model.
6. **Roy (1951)** and **Heckman and Sedlacek (1985)** for the theoretical foundation.

A concise modeling justification could read:

> Following the efficiency-units and Roy-model traditions, labor earnings equal the sector-specific price of an efficiency unit of labor multiplied by worker-specific productivity. We decompose the sector-specific efficiency wage into an economy-wide wage rate and sectoral productivity, so that $y_{ist}=w_tA_{st}\varepsilon_{it}$; see Roy (1951), Heckman and Sedlacek (1985), Lee and Wolpin (2006), Lagakos and Waugh (2013), Dix-Carneiro (2014), Hsieh et al. (2019), and Faia, Kudlyak, and Shabalina (2021).


### References

Dix-Carneiro R (2014) Trade liberalization and labor market dynamics. *Econometrica* 82(3):825–885. ([Econometrica][10])

Faia E, Kudlyak M, Shabalina E (2021) Dynamic labor reallocation with heterogeneous skills and uninsured idiosyncratic risk. *Federal Reserve Bank of San Francisco Working Paper* 2021-16. ([FRBSF][9])

Heckman JJ, Sedlacek G (1985) Heterogeneity, aggregation, and market wage functions: An empirical model of self-selection in the labor market. *Journal of Political Economy* 93(6):1077–1125. ([Chicago Journals][5])

Hsieh CT, Hurst E, Jones CI, Klenow PJ (2019) The allocation of talent and U.S. economic growth. *Econometrica* 87(5):1439–1474. ([Wiley Online Library][6])

Lagakos D, Waugh ME (2013) Selection, agriculture, and cross-country productivity differences. *American Economic Review* 103(2):948–980. ([American Economic Association][7])

Lee D, Wolpin KI (2006) Intersectoral labor mobility and the growth of the service sector. *Econometrica* 74(1):1–46. ([DOI][8])

Roy AD (1951) Some thoughts on the distribution of earnings. *Oxford Economic Papers* 3(2):135–146. ([OUP Academic][4])

[1]: https://www.waugheconomics.com/uploads/2/2/5/6/22563786/selection_lagakos_waugh_final.pdf "Selection, Agriculture, and Cross-Country Productivity Differences"
[2]: https://core.ac.uk/download/pdf/6330451.pdf "C:\1\cohort2\equilib9.wpd"
[3]: https://www.nber.org/system/files/working_papers/w18693/w18693.pdf "The Allocation of Talent and U.S. Economic Growth"
[4]: https://academic.oup.com/oep/article-abstract/3/2/135/2360754?utm_source=chatgpt.com "SOME THOUGHTS ON THE DISTRIBUTION OF EARNINGS 1 | Oxford Economic Papers | Oxford Academic"
[5]: https://www.journals.uchicago.edu/doi/abs/10.1086/261352?utm_source=chatgpt.com "Heterogeneity, Aggregation, and Market Wage Functions: An Empirical Model of Self-Selection in the Labor Market | Journal of Political Economy: Vol 93, No 6"
[6]: https://onlinelibrary.wiley.com/doi/10.3982/ECTA11427?utm_source=chatgpt.com "The Allocation of Talent and U.S. Economic Growth - Hsieh - 2019 - Econometrica - Wiley Online Library"
[7]: https://www.aeaweb.org/articles?id=10.1257%2Faer.103.2.948&utm_source=chatgpt.com "Selection, Agriculture, and Cross-Country Productivity Differences - American Economic Association"
[8]: https://doi.org/10.1111/j.1468-0262.2006.00648.x?utm_source=chatgpt.com "Intersectoral Labor Mobility and the Growth of the Service Sector - Lee - 2006 - Econometrica - Wiley Online Library"
[9]: https://www.frbsf.org/wp-content/uploads/wp2021-16.pdf "Dynamic Labor Reallocation with Heterogeneous Skills and Uninsured Idiosyncratic Risk"
[10]: https://doi.org/10.3982/ECTA10457 "Trade Liberalization and Labor Market Dynamics"
