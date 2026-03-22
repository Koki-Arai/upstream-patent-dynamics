# Code Directory

All scripts are designed to run on **Google Colab** (Python 3.12).  
Upload scripts to `/content/` and run with `exec(open('/content/script_name.py').read())`.

---

## Execution Order

```
Step 1:  00_colab_setup.py              → Build pharmaceutical panel (df, df_panel)
Step 2:  patent_robustness_v2_iip.py   → Main analysis: VIF, regression, SMM v2
Step 3a: citation_weighted_robustness.py → Appendix C8–C9
Step 3b: macro_controls_robustness.py  → Appendix C10–C12
Step 3c: pooled_regression_table3.py   → Table 3 (pooled regression)
Step 4:  99_save_session.py            → Save to Google Drive
Step 5:  98a_export_results.py         → Download results locally
```

---

## Script Descriptions

### `00_colab_setup.py`
**Purpose:** Load IIP Patent Database and construct the pharmaceutical application-level panel (`df`) and firm–field–year panel (`df_panel`).

- Reads decade-split `.txt` files from `/content/` in 500,000-row chunks
- Filters to pharmaceutical IPC codes (A61, C07, C12)
- Constructs `self_cite_ratio`, `has_reject_reason`, `log_citation_count`, lagged portfolio stocks, HHI, and density variables
- Saves `df_application.pkl` and `df_panel.pkl` to Google Drive

**RAM requirement:** ~4 GB available recommended.  
**Runtime:** ~20–30 minutes.

---

### `patent_robustness_v2_iip.py`
**Purpose:** Main empirical analysis producing Tables 1, 2, and 4.

- **BLOCK 1:** VIF analysis (Appendix C6)
- **BLOCK 2:** Interaction regression — `self_cite × n_inventors` (Appendix C7)
- **BLOCK 3:** SMM v2 estimation (δ = 0.10) — Table 4
- **BLOCK 4:** Standardized effect size

**Input:** `df`, `df_panel` in memory (from `00_colab_setup.py` or Drive pickle).  
**Runtime:** ~2–5 minutes.

---

### `citation_weighted_robustness.py`
**Purpose:** Citation-weighted robustness checks (Appendix C8–C9).

- Constructs forward citation counts within T = 5 and T = 10 year windows
- Re-estimates scrutiny and expansion specifications with citation-weighted variables
- Outputs: `appendix_table_d6_citation_weighted_scrutiny.csv`, `appendix_table_d7_citation_weighted_expansion.csv`

**Input:** `df`, `df_panel` in memory + IIP `cc_*.txt` files in `/content/`.  
**Runtime:** ~10–20 minutes (cc re-read required).

---

### `citation_weighted_step5_onwards.py`
**Purpose:** Run STEP 5 onwards of citation-weighted analysis independently (use after `!pip install linearmodels` if ModuleNotFoundError occurs).

**Input:** `df_wt`, `df_panel_wt` in memory (produced by `citation_weighted_robustness.py` STEP 1–4).

---

### `macro_controls_robustness.py`
**Purpose:** Macroeconomic controls robustness (Appendix C10–C12).

- **Spec A:** Replace year FE with 5 macroeconomic controls (TOPIX, CPI, USD/JPY, lending rate, CGPI)
- **Spec B/C:** Split-sample by macro regime (boom/bust years, high/low inflation)
- Requires `data/macro_indicators_monthly.csv` uploaded to `/content/マクロ指標.csv`

**Input:** `df`, `df_panel` in memory.  
**Runtime:** ~5–10 minutes.

---

### `pooled_regression_table3.py`
**Purpose:** Pooled regression across all technology fields with pharmaceutical interaction terms (Table 3).

- Loads all technology fields (not just pharma) from IIP raw files
- Constructs `df_all` (~3.3M rows at SAMPLE_FRAC=0.3) and `df_panel_all`
- Estimates 3 panels × 3 specifications = 9 models
- Outputs: `table3_pooled_scope.csv`

**Key setting:** Set `SAMPLE_FRAC = 0.3` (line 96) on free Colab (12 GB RAM). Use 1.0 on Colab Pro+.  
**Input:** IIP `.txt` files in `/content/`.  
**Runtime:** ~30–50 minutes (includes full data reload).

---

### `patent_robustness_v2.py`
**Purpose:** Dummy-data version of `patent_robustness_v2_iip.py` for testing without IIP data.  
Generates synthetic data that mimics the structure of the pharmaceutical panel.

---

### `99_save_session.py`
**Purpose:** Save all in-memory data and results to Google Drive.

- Creates `/content/drive/MyDrive/patent_analysis/` folder structure
- Saves `df_application.pkl`, `df_panel.pkl`, and result CSVs
- Logs session metadata (timestamp, variable sizes, memory usage)

---

### `98a_export_results.py`
**Purpose:** Download estimation results locally (lightweight).  
Exports: VIF table, regression coefficients, SMM results, standardized effects.  
**Output size:** ~50 KB.

---

### `98b_export_pharma_sample.py`
**Purpose:** Download sample data + descriptive statistics.  
Exports: 50,000-row random sample of `df`, full `df_panel`, correlation matrix, data dictionary.  
**Output size:** ~30–80 MB (ZIP).

---

### `98c_export_pharma_full.py`
**Purpose:** Download full pharmaceutical dataset in chunked CSV files.  
Each file: ~150,000 rows (~40 MB). Includes `recombine_data.py` for local reassembly.  
**Output size:** ~200 MB total.

---

### `98d_export_replication.py`
**Purpose:** Create a complete replication package ZIP (data + results + code + README).  
Suitable for Zenodo upload or journal submission.  
**Output size:** ~15–30 MB.
