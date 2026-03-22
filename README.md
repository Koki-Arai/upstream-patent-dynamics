# Upstream Patent Dynamics and Competition: Portfolio Structure, Scrutiny, and Path Dependence

**Koki Arai**  
Faculty of Business Studies, Kyoritsu Women's University  
Email: koki.arai@nifty.ne.jp

JSPS KAKENHI Grant Number 23K01404

---

## Overview

This repository contains the replication code and results for the paper:

> Arai, K. (2025). "Upstream Patent Dynamics and Competition: Portfolio Structure, Scrutiny, and Path Dependence." *Journal of Dynamic Competition* (under review).

The paper examines the upstream patent portfolio dynamics that shape competitive conditions in pharmaceutical markets before any patent linkage mechanism becomes operative. Using micro-level data from the IIP Patent Database (Japanese pharmaceutical patents), we document two structural channels:

1. **Scrutiny channel**: Self-referential accumulation — measured by the share of examiner-generated citations to the applicant's own prior patents — is the dominant empirical predictor of examination intensity.
2. **Path dependence channel**: Firms expand patent portfolios in fields where they already hold larger stocks of applications and granted patents.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── 00_colab_setup.py              # Data loading & pharmaceutical panel construction
│   ├── patent_robustness_v2_iip.py    # Main analysis (VIF, regression, SMM v2)
│   ├── citation_weighted_robustness.py # Citation-weighted robustness (Appendix C8–C9)
│   ├── citation_weighted_step5_onwards.py  # STEP 5+ standalone (after linearmodels install)
│   ├── macro_controls_robustness.py   # Macroeconomic controls (Appendix C10–C12)
│   ├── pooled_regression_table3.py    # Pooled regression / scope conditions (Table 3)
│   ├── patent_robustness_v2.py        # Dummy-data version for testing
│   ├── 98a_export_results.py          # Local export: estimation results
│   ├── 98b_export_pharma_sample.py    # Local export: sample + descriptive stats
│   ├── 98c_export_pharma_full.py      # Local export: full pharma dataset (chunked)
│   ├── 98d_export_replication.py      # Local export: replication package ZIP
│   └── 99_save_session.py             # Save session to Google Drive
├── results/
│   ├── table3_pooled_scope.csv                          # Table 3: Pooled regression results
│   ├── appendix_specA_scrutiny.csv                      # Appendix C10: Macro controls (scrutiny)
│   ├── appendix_specB_scrutiny.csv                      # Appendix C11: Macro split-sample (scrutiny)
│   ├── appendix_specC_expansion.csv                     # Appendix C12: Macro split-sample (expansion)
│   ├── appendix_table_d6_citation_weighted_scrutiny.csv # Appendix C8: Citation-weighted scrutiny
│   └── appendix_table_d7_citation_weighted_expansion.csv # Appendix C9: Citation-weighted expansion
└── data/
    └── macro_indicators_monthly.csv   # Monthly macro indicators (TOPIX, CPI, USD/JPY, etc.)
```

---

## Data Availability

### Included in this repository

- **`data/macro_indicators_monthly.csv`**: Monthly macroeconomic indicators used in Appendix C10–C12.  
  Variables: Nikkei 225, TOPIX (open/high/low/close), CPI (2020=100), USD/JPY exchange rate, lending rate, Corporate Goods Price Index (YoY).  
  Period: 1970–2026.

### Not included — IIP Patent Database

The main analysis uses the **IIP Patent Database** (Institute of Intellectual Property, Tokyo). The following files are required:

| File | Content | Approx. size |
|------|---------|--------------|
| `ap_1990s.txt` – `ap_2020s.txt` | Patent applications | 217–242 MB each |
| `cc_1990s.txt` – `cc_2020s.txt` | Citation links | 74–966 MB each |
| `applicant_1990s.txt` – `applicant_2020s.txt` | Applicant information | — |
| `inventor_1990s.txt` – `inventor_2020s.txt` | Inventor information | — |
| `hr_1990s.txt` – `hr_2020s.txt` | Rights-holder (granted patents) | — |

**The IIP Patent Database is available from the Institute of Intellectual Property (IIP), Japan.**  
Access: https://www.iip.or.jp/patentdb/  
The database requires a data use agreement with IIP. Researchers wishing to replicate this study should apply directly to IIP.

> **For data access inquiries**, please contact the author at koki.arai@nifty.ne.jp.  
> We are unable to share the raw IIP data files due to the data use agreement, but we are happy to assist researchers who have obtained their own IIP license.

---

## Replication Instructions

### Environment

The analysis is designed to run on **Google Colab** (free or Pro tier).

- Python 3.12
- Google Drive for persistent storage

### Step-by-step

**Step 1: Obtain IIP data**  
Apply for access at https://www.iip.or.jp/patentdb/ and download the decade-split `.txt` files.

**Step 2: Upload files to Google Colab**  
Upload all IIP `.txt` files to `/content/` in your Colab session.  
Optionally, save them to Google Drive to avoid re-uploading after session resets:
```python
import shutil
for f in ['ap_1990s.txt', 'cc_1990s.txt', ...]:
    shutil.copy(f'/content/{f}', f'/content/drive/MyDrive/IIP_raw/{f}')
```

**Step 3: Install packages**
```python
import subprocess, sys
for pkg in ['numpy==1.26.4', 'scipy==1.13.1',
            'statsmodels==0.14.2', 'pandas==2.2.2', 'linearmodels']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pkg])
```

**Step 4: Build the pharmaceutical panel**
```python
exec(open('/content/00_colab_setup.py').read())
# Outputs: df (905,408 rows), df_panel (391,551 rows)
# Saves to: /content/drive/MyDrive/patent_analysis_cache/
```

**Step 5: Run main analysis**
```python
exec(open('/content/patent_robustness_v2_iip.py').read())
# Outputs: VIF table, regression coefficients, SMM v2 results
```

**Step 6: Run additional robustness checks**
```python
# Citation-weighted robustness (Appendix C8–C9)
exec(open('/content/citation_weighted_robustness.py').read())

# Macroeconomic controls (Appendix C10–C12)
# First upload data/macro_indicators_monthly.csv to /content/マクロ指標.csv
exec(open('/content/macro_controls_robustness.py').read())

# Pooled regression — Table 3 (requires SAMPLE_FRAC=0.3 on free Colab)
exec(open('/content/pooled_regression_table3.py').read())
```

**Step 7: Export results**
```python
exec(open('/content/98a_export_results.py').read())   # Lightweight: coefficients only
exec(open('/content/98d_export_replication.py').read()) # Full replication ZIP
```

### Session recovery after Colab disconnect

```python
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
SAVE_DIR = '/content/drive/MyDrive/patent_analysis_cache'
df       = pd.read_pickle(f'{SAVE_DIR}/df_application.pkl')
df_panel = pd.read_pickle(f'{SAVE_DIR}/df_panel.pkl')
print(f'df: {len(df):,}  df_panel: {len(df_panel):,}')
```

---

## Key Variables

| Variable | Definition | Source |
|----------|-----------|--------|
| `self_cite_ratio` | Share of examiner citations directed at applicant's own prior patents | IIP `cc.txt` (examiner codes 19, 22, 31, 75, 89, 93) |
| `has_reject_reason` | =1 if any rejection-related citation recorded (codes 19, 22, 89) | IIP `cc.txt` |
| `log_citation_count` | log(1 + total examiner citations) | IIP `cc.txt` |
| `log_lag_patent_stock` | log(1 + cumulative prior applications by firm in field, lagged 1 yr) | IIP `ap.txt` |
| `log_lag_grant_stock` | log(1 + cumulative prior granted patents by firm in field, lagged 1 yr) | IIP `hr.txt` |
| `lag_applicant_hhi` | Lagged Herfindahl–Hirschman Index of applicant filing shares in field | IIP `ap.txt` |
| `log_lag_density` | log(1 + total applications in field, lagged 1 yr) | IIP `ap.txt` |

**Pharmaceutical classification**: IPC codes A61K, A61P, C07D, C07K, C12N, C12Q, G01N.

---

## Main Results

| Finding | Key coefficient | Table |
|---------|----------------|-------|
| Self-citation → scrutiny (rejection) | 0.976 (SE 0.002) | Table 1 |
| Self-citation → scrutiny (citation intensity) | 0.747 (SE 0.003) | Table 1 |
| Patent stock → expansion | 0.111 (SE 0.003) | Table 2 |
| Grant stock → expansion | 0.109 (SE 0.003) | Table 2 |
| Self-citation → expansion (not robust) | −0.016 → 0.000 (excl. top 1%) | Table 2 / Appendix C5 |
| SMM v2: scrutiny parameter (φ̂) | 0.963 | Table 4 |
| SMM v2: patent stock parameter (π̂_k) | 0.118 | Table 4 |

---

## Software Requirements

See `requirements.txt` for the full list. Core dependencies:

```
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
statsmodels==0.14.2
linearmodels>=0.60
```

---

## Citation

```bibtex
@unpublished{arai2025upstream,
  author  = {Arai, Koki},
  title   = {Upstream Patent Dynamics and Competition:
             Portfolio Structure, Scrutiny, and Path Dependence},
  year    = {2025},
  note    = {Under review, Journal of Dynamic Competition},
  institution = {Kyoritsu Women's University}
}
```

---

## Funding

This research was supported by JSPS KAKENHI Grant Number **23K01404**.

---

## License

Code: MIT License  
Data (`macro_indicators_monthly.csv`): Compiled from publicly available sources (JPX, Statistics Bureau of Japan, Bank of Japan). Free to use with attribution.  
IIP Patent Database: Subject to IIP data use agreement — not redistributable.
