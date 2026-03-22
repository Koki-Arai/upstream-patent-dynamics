# Data Access

## IIP Patent Database

The main empirical analysis uses the **IIP Patent Database**, provided by the Institute of Intellectual Property (IIP), Tokyo, Japan.

### How to obtain access

1. Visit the IIP Patent Database page:  
   https://www.iip.or.jp/patentdb/

2. Submit a data use application to IIP. The database is available to academic researchers at no cost, subject to a data use agreement.

3. Download the decade-split text files. The following files are required for full replication:

| Decade | Files needed |
|--------|-------------|
| 1990s | `ap_1990s.txt`, `cc_1990s.txt`, `applicant_1990s.txt`, `inventor_1990s.txt`, `hr_1990s.txt` |
| 2000s | `ap_2000s.txt`, `cc_2000s.txt`, `applicant_2000s.txt`, `inventor_2000s.txt`, `hr_2000s.txt` |
| 2010s | `ap_2010s.txt`, `cc_2010s.txt`, `applicant_2010s.txt`, `inventor_2010s.txt`, `hr_2010s.txt` |
| 2020s | `ap_2020s.txt`, `cc_2020s.txt`, `applicant_2020s.txt`, `inventor_2020s.txt`, `hr_2020s.txt` |

### Column structure

| Table | Key columns used |
|-------|-----------------|
| `ap_*.txt` | `ida` (app ID), `adate` (app date), `class1` (IPC), `claim1` (claim count) |
| `cc_*.txt` | `citing`, `cited`, `reason`, `reason_date` |
| `applicant_*.txt` | `ida`, `seq`, `idname` (applicant code) |
| `inventor_*.txt` | `ida`, `ida_seq` |
| `hr_*.txt` | `ida`, `seq` (granted patents) |

### Contact

If you have questions about accessing the IIP data or reproducing specific parts of the analysis, please contact:

**Koki Arai**  
Faculty of Business Studies, Kyoritsu Women's University  
Email: koki.arai@nifty.ne.jp

> We cannot share the raw IIP data files due to the data use agreement, but we are happy to assist researchers who have obtained their own IIP license.

---

## Macroeconomic Indicators

The file `data/macro_indicators_monthly.csv` is included in this repository.

**Variables:**
- Nikkei 225: open, high, low, close (monthly)
- TOPIX: open, high, low, close (monthly)
- CPI: index (2020 = 100)
- USD/JPY exchange rate (month-end Tokyo market)
- Lending rate: new loans, domestic banks (Bank of Japan)
- Corporate Goods Price Index: YoY change

**Sources:** JPX (Nikkei/TOPIX), Statistics Bureau of Japan (CPI), Bank of Japan (lending rate, CGPI).  
**Period:** January 1970 – January 2026.

These data are compiled from publicly available sources and may be freely used with attribution.
