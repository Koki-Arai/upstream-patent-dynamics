# ================================================================
# 98d_export_replication.py
# ローカル保存 D — 論文再現パッケージ（Replication Package）
# ================================================================
# 【内容】論文の Table 1〜4 を再現するための最小限セット
#   replication_package_YYYYMMDD.zip
#   ├── data/
#   │   ├── df_pharma_sample50k.csv     分析用サンプル（5万行）
#   │   ├── df_panel_full.csv           expansion パネル全件
#   │   ├── descriptive_stats.csv       記述統計
#   │   └── data_dictionary.csv         変数定義
#   ├── results/
#   │   ├── vif.csv                     VIF（Table A3）
#   │   ├── regression_coefs.csv        Table 1 / Table 1B
#   │   ├── smm_comparison.csv          Table 4B
#   │   └── standardized_effects.json   効果量サマリー
#   ├── code/
#   │   ├── patent_robustness_v2_iip.py 分析本体
#   │   ├── 00_colab_setup.py           前処理
#   │   └── 99_save_session.py          保存スクリプト
#   └── README.txt                      再現手順
#
# 【容量】目安 15〜30 MB（圧縮後）
# 【用途】ジャーナル投稿時のデータ・コード公開（Zenodo等）
#         共同研究者との共有
# ================================================================

import os, json, datetime, zipfile, textwrap
import pandas as pd
import numpy as np
from google.colab import files

TS      = datetime.datetime.now().strftime('%Y%m%d_%H%M')
TSD     = datetime.datetime.now().strftime('%Y-%m-%d')
WORK    = f'/content/replication_work_{TS}'
ZIP_OUT = f'/content/replication_package_{TS}.zip'

for sub in ['data', 'results', 'code']:
    os.makedirs(f'{WORK}/{sub}', exist_ok=True)

print(f"Building replication package: {os.path.basename(ZIP_OUT)}")
print(f"Timestamp: {TS}\n")

added = []   # (zip_arcname, local_path)

def add(local_path, arc_path):
    added.append((arc_path, local_path))
    mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"  + {arc_path:60s} {mb*1024:6.0f} KB")


# ================================================================
# DATA
# ================================================================
print("[DATA]")

df_obj    = globals().get('df')
panel_obj = globals().get('df_panel')

DF_COLS = [
    'app_id', 'year_id', 'ipc', 'field_id', 'claim1',
    'applicant_id', 'has_reject_reason', 'citation_count',
    'log_citation_count', 'self_cite_ratio', 'n_inventors',
    'lag_applicant_hhi', 'log_lag_density',
]
PANEL_COLS = [
    'applicant_id', 'field_id', 'year_id', 'n_filed',
    'log_lag_patent_stock', 'log_lag_grant_stock',
    'log_applications', 'avg_self_cite_ratio',
    'lag_applicant_hhi', 'log_lag_density',
]

# df sample 50k
if df_obj is not None:
    avail = [c for c in DF_COLS if c in df_obj.columns]
    sample = df_obj[avail].sample(min(50_000, len(df_obj)),
                                  random_state=42).reset_index(drop=True)
    p = f'{WORK}/data/df_pharma_sample50k.csv'
    sample.to_csv(p, index=False, encoding='utf-8-sig')
    add(p, 'data/df_pharma_sample50k.csv')

# df_panel full
if panel_obj is not None:
    avail_p = [c for c in PANEL_COLS if c in panel_obj.columns]
    p = f'{WORK}/data/df_panel_full.csv'
    panel_obj[avail_p].to_csv(p, index=False, encoding='utf-8-sig')
    add(p, 'data/df_panel_full.csv')

# descriptive stats
if df_obj is not None:
    stat_vars = [c for c in DF_COLS if c in df_obj.columns
                 and df_obj[c].dtype in ['float64','int64','int32']]
    desc = df_obj[stat_vars].describe(percentiles=[.25,.50,.75]).T.reset_index()
    desc = desc.rename(columns={'index':'variable'})
    p = f'{WORK}/data/descriptive_stats.csv'
    desc.to_csv(p, index=False, encoding='utf-8-sig')
    add(p, 'data/descriptive_stats.csv')

# data dictionary
dd = [
    {'table':'df',      'variable':'app_id',            'description':'出願番号 (IIP: ida)'},
    {'table':'df',      'variable':'year_id',            'description':'出願年'},
    {'table':'df',      'variable':'field_id',           'description':'IPC先頭4文字カテゴリ'},
    {'table':'df',      'variable':'has_reject_reason',  'description':'拒絶引用あり (0/1)'},
    {'table':'df',      'variable':'log_citation_count', 'description':'log(1+引用件数)'},
    {'table':'df',      'variable':'self_cite_ratio',    'description':'自己引用比率 [0,1]'},
    {'table':'df',      'variable':'n_inventors',        'description':'発明者数'},
    {'table':'df',      'variable':'lag_applicant_hhi',  'description':'前年出願人HHI'},
    {'table':'df',      'variable':'log_lag_density',    'description':'log(1+前年出願件数)'},
    {'table':'df',      'variable':'claim1',             'description':'請求項数'},
    {'table':'df_panel','variable':'log_lag_patent_stock','description':'log(1+累積出願ストック)'},
    {'table':'df_panel','variable':'log_lag_grant_stock', 'description':'log(1+累積登録ストック)'},
    {'table':'df_panel','variable':'log_applications',   'description':'log(1+当年出願件数)'},
    {'table':'df_panel','variable':'avg_self_cite_ratio','description':'フィールド平均自己引用比率'},
]
p = f'{WORK}/data/data_dictionary.csv'
pd.DataFrame(dd).to_csv(p, index=False, encoding='utf-8-sig')
add(p, 'data/data_dictionary.csv')


# ================================================================
# RESULTS
# ================================================================
print("\n[RESULTS]")

# VIF
vif_obj = globals().get('vif_df') or (
    globals().get('results', {}).get('vif')
    if isinstance(globals().get('results'), dict) else None
)
if vif_obj is not None:
    p = f'{WORK}/results/vif.csv'
    vif_obj.to_csv(p, index=False, encoding='utf-8-sig')
    add(p, 'results/vif.csv')

# Regression coefficients
reg_obj = globals().get('reg_results') or (
    globals().get('results', {}).get('reg_results')
    if isinstance(globals().get('results'), dict) else None
)
if reg_obj is not None and isinstance(reg_obj, dict):
    rows = []
    rename = {'self_x_inv': 'self_cite_ratio:n_inventors'}
    for key, fit in reg_obj.items():
        outcome, model = key.rsplit('_', 1)
        try:
            params = fit.params
            bse    = fit.std_errors if hasattr(fit,'std_errors') else fit.bse
            pvals  = fit.pvalues
            for a in [params, bse, pvals]:
                if hasattr(a,'iloc') and a.ndim>1: a = a.iloc[:,0]
            for v in params.index:
                rows.append({
                    'outcome': outcome, 'model': model,
                    'variable': rename.get(v, v),
                    'coef': round(float(params[v]),6),
                    'se':   round(float(bse[v]),6)   if v in bse.index   else None,
                    'pval': round(float(pvals[v]),6) if v in pvals.index else None,
                })
        except Exception: pass
    if rows:
        p = f'{WORK}/results/regression_coefs.csv'
        pd.DataFrame(rows).to_csv(p, index=False, encoding='utf-8-sig')
        add(p, 'results/regression_coefs.csv')

# SMM comparison table
best_obj = globals().get('best')
msim_obj = globals().get('m_sim')
smm_tbl  = globals().get('results', {}).get('smm_table') \
           if isinstance(globals().get('results'), dict) else None

smm_rows = [
    {'Mechanism':'phi_s (scrutiny)',      'RF':0.976, 'SMMv1':1.837,
     'SMMv2': round(float(best_obj.x[0]),4) if best_obj else None,
     'm_sim': round(float(msim_obj[0]),4) if msim_obj is not None else None,
     'sign_v1':'OK','sign_v2':'OK'},
    {'Mechanism':'pi_k (patent stock)',   'RF':0.111, 'SMMv1':-0.049,
     'SMMv2': round(float(best_obj.x[1]),4) if best_obj else None,
     'm_sim': round(float(msim_obj[1]),4) if msim_obj is not None else None,
     'sign_v1':'REVERSAL','sign_v2':'OK'},
    {'Mechanism':'pi_g (grant stock)',    'RF':0.109, 'SMMv1':0.040,
     'SMMv2': round(float(best_obj.x[2]),4) if best_obj else None,
     'm_sim': round(float(msim_obj[2]),4) if msim_obj is not None else None,
     'sign_v1':'OK','sign_v2':'OK'},
]
p = f'{WORK}/results/smm_comparison.csv'
pd.DataFrame(smm_rows).to_csv(p, index=False, encoding='utf-8-sig')
add(p, 'results/smm_comparison.csv')

# standardized effects JSON
if df_obj is not None and 'self_cite_ratio' in df_obj.columns:
    sd = float(df_obj['self_cite_ratio'].std())
    eff = {
        'n':              len(df_obj),
        'sd_self_cite':   round(sd, 4),
        'mean_self_cite': round(float(df_obj['self_cite_ratio'].mean()), 4),
        'coef_reject':    0.976,
        'delta_reject_1sd':   round(0.976*sd, 4),
        'mean_reject_rate':   0.425,
        'pct_increase_reject': round(0.976*sd/0.425*100, 1),
        'vif_max': float(vif_obj[vif_obj['feature']!='const']['VIF'].max())
                   if vif_obj is not None else None,
    }
    p = f'{WORK}/results/standardized_effects.json'
    with open(p,'w',encoding='utf-8') as f:
        json.dump(eff, f, ensure_ascii=False, indent=2)
    add(p, 'results/standardized_effects.json')


# ================================================================
# CODE — analysis scripts from /content
# ================================================================
print("\n[CODE]")

code_files = [
    '/content/patent_robustness_v2_iip.py',
    '/content/00_colab_setup.py',
    '/content/99_save_session.py',
    '/content/98a_export_results.py',
    '/content/98b_export_pharma_sample.py',
    '/content/98c_export_pharma_full.py',
    '/content/98d_export_replication.py',
]
for src in code_files:
    if os.path.exists(src):
        fname = os.path.basename(src)
        dst   = f'{WORK}/code/{fname}'
        import shutil; shutil.copy(src, dst)
        add(dst, f'code/{fname}')
    else:
        print(f"  [skip] {os.path.basename(src)} — not found in /content")


# ================================================================
# README
# ================================================================
print("\n[README]")

n_df    = len(df_obj)    if df_obj    is not None else 0
n_panel = len(panel_obj) if panel_obj is not None else 0

readme_text = textwrap.dedent(f"""\
    ================================================================
    Replication Package
    "Upstream Patent Dynamics and Competition:
     Portfolio Structure, Scrutiny, and Path Dependence"
    ================================================================
    Generated : {TSD}
    Data      : IIP Patent Database (pharmaceutical sector)
    Sample    : {n_df:,} applications (pharma, 1988-2023)
    Panel     : {n_panel:,} firm-field-year observations

    ----------------------------------------------------------------
    FOLDER STRUCTURE
    ----------------------------------------------------------------
    data/
      df_pharma_sample50k.csv   Random sample for analysis/verification
      df_panel_full.csv         Full firm-field-year expansion panel
      descriptive_stats.csv     Table A1 source data
      data_dictionary.csv       Variable definitions

    results/
      vif.csv                   Table A3: VIF results
      regression_coefs.csv      Table 1 / Table 1B: scrutiny regressions
      smm_comparison.csv        Table 4B: SMM v1 vs. v2
      standardized_effects.json Standardized effect size summary

    code/
      00_colab_setup.py         IIP data loading and preprocessing
      patent_robustness_v2_iip.py  Main analysis (VIF, regression, SMM)
      99_save_session.py        Session save to Google Drive
      98a–d_export_*.py         Local download scripts

    ----------------------------------------------------------------
    REPLICATION STEPS (Google Colab)
    ----------------------------------------------------------------
    1. Upload IIP files to /content:
         ap_1990s.txt ... ap_2020s.txt
         cc_1990s.txt ... cc_2020s.txt
         applicant_*.txt, inventor_*.txt, hr_*.txt

    2. Run 00_colab_setup.py (CELL 0 through CELL 7)
         → Creates df (application level) and df_panel

    3. Run patent_robustness_v2_iip.py
         → Produces VIF, regression, and SMM results

    4. Run 99_save_session.py to save to Google Drive

    ----------------------------------------------------------------
    KEY RESULTS SUMMARY
    ----------------------------------------------------------------
    BLOCK 1 — VIF (Reviewer 1: multicollinearity)
      Max VIF = 1.168 (self_cite_ratio: 1.008)
      → No multicollinearity concern

    BLOCK 2 — Interaction regression
      has_reject_reason:   self_cite x n_inv = -0.005 (p=0.131) NS
      log_citation_count:  self_cite x n_inv = -0.043 (p<0.001) **
      → Scrutiny channel operates independently of team size (rejection);
        partial attenuation for larger teams (citation intensity)

    BLOCK 3 — SMM (Reviewer 3: sign reversal)
      SMM v1 (delta=0):    pi_k = -0.049  [SIGN REVERSAL]
      SMM v2 (delta=0.10): pi_k = +0.118  [RESOLVED]
      → Depreciation assumption resolves the inconsistency

    ----------------------------------------------------------------
    SOFTWARE
    ----------------------------------------------------------------
    Python 3.12, numpy 1.26.4, pandas 2.2.2,
    statsmodels 0.14.2, scipy 1.13.1, linearmodels (latest)

    ----------------------------------------------------------------
    DATA SOURCE
    ----------------------------------------------------------------
    IIP Patent Database (Institute of Intellectual Property, 2024)
    https://www.iip.or.jp/patentdb/
    JSPS KAKENHI Grant Number 23K01404
    ================================================================
""")

p = f'{WORK}/README.txt'
with open(p, 'w', encoding='utf-8') as f:
    f.write(readme_text)
add(p, 'README.txt')


# ================================================================
# ZIP & DOWNLOAD
# ================================================================
print(f"\n{'='*55}")
print(f"Packing {len(added)} files into ZIP ...")

with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for arc_name, local_path in added:
        zf.write(local_path, arc_name)

zip_mb = os.path.getsize(ZIP_OUT) / 1024 / 1024
print(f"\n  📦 {os.path.basename(ZIP_OUT):55s} {zip_mb:.1f} MB")
print(f"\nDownloading ...")
files.download(ZIP_OUT)
print(f"  ↓ {os.path.basename(ZIP_OUT)}")

print("\n✓ Export D (Replication Package) complete.")
print(f"  Total files: {len(added)}")
print(f"  ZIP size   : {zip_mb:.1f} MB")
