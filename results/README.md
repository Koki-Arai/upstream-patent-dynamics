# Results Directory

This directory contains the numerical outputs from the main empirical analysis.  
All files are in UTF-8 CSV format (with BOM for Excel compatibility).

---

## File Descriptions

### `table3_pooled_scope.csv`
**Content:** Table 3 — Pooled Evidence and Scope Conditions  
**Produced by:** `pooled_regression_table3.py`

| Column | Description |
|--------|-------------|
| `variable` | Variable name |
| `coef` | Coefficient estimate |
| `se` | Standard error (clustered by IPC subclass) |
| `pval` | p-value |
| `ci_lo`, `ci_hi` | 95% confidence interval |
| `panel` | A (has_reject_reason), B (log_citation_count), C (log_applications) |
| `spec` | Pharma only / Pooled (no interaction) / Pooled + interaction |
| `outcome` | Dependent variable name |

**Key results:**
- `self_cite × pharma` for Panel A: coef = −0.042, p = 0.184 (not significant → scope condition H5 supported)
- `self_cite × pharma` for Panel B: coef = −0.455, p < 0.001 (significant negative → pharma not amplified)
- `patent_stock × pharma` for Panel C: coef = −0.039, p = 0.075 (marginally significant)

---

### `appendix_table_d6_citation_weighted_scrutiny.csv`
**Content:** Appendix Table C8 — Citation-Weighted Scrutiny Robustness  
**Produced by:** `citation_weighted_robustness.py`  
**Dependent variable:** `has_reject_reason`

Panel A: self_cite_ratio replaced by citation-weighted variant (T = 5yr, T = 10yr)  
Panel B: citation weight added as additional control alongside unweighted self_cite_ratio

**Key result:** Coefficient on self_cite_ratio declines by < 2% when citation weights are added as controls (0.513 → 0.504), confirming the scrutiny effect is independent of patent quality.

---

### `appendix_table_d7_citation_weighted_expansion.csv`
**Content:** Appendix Table C9 — Citation-Weighted Patent Stock Robustness  
**Produced by:** `citation_weighted_robustness.py`  
**Dependent variable:** `log_applications`

Citation-weighted patent stocks enter positively and significantly (T = 5yr: 0.153; T = 10yr: 0.110), confirming path dependence is robust to quality weighting.

---

### `appendix_specA_scrutiny.csv`
**Content:** Appendix Table C10 — Macroeconomic Controls: Year FE Replaced  
**Produced by:** `macro_controls_robustness.py`

Scrutiny specification with year FE replaced by five standardized macroeconomic controls.  
**Key result:** self_cite_ratio coefficient = 0.548 (vs. baseline 0.513) — stable.

---

### `appendix_specB_scrutiny.csv`
**Content:** Appendix Table C11 — Scrutiny by Macroeconomic Regime (Split-Sample)  
**Produced by:** `macro_controls_robustness.py`

self_cite_ratio coefficient estimated separately for boom/bust years and high/low inflation years.  
**Key result:** Range 0.487–0.545 across all subsamples (spread = 11% of baseline).

---

### `appendix_specC_expansion.csv`
**Content:** Appendix Table C12 — Portfolio Expansion by Macroeconomic Regime (Split-Sample)  
**Produced by:** `macro_controls_robustness.py`

log_lag_patent_stock coefficient estimated separately by macroeconomic regime.  
**Key result:** Range 0.163–0.169 across all subsamples — highly stable.
