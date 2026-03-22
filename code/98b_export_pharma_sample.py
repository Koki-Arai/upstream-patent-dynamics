# ================================================================
# 98b_export_pharma_sample.py
# ローカル保存 B — 医薬品データのサンプル＋記述統計
# ================================================================
# 【内容】
#   - df の無作為サンプル（デフォルト 50,000行）
#   - df_panel の全件（391,551行、比較的軽量）
#   - 記述統計（変数別 mean/sd/min/max/分位点）
#   - 変数の相関行列
# 【容量】目安 30〜80 MB
# 【用途】
#   - ローカルでの追加分析・可視化
#   - 論文 Table A1（記述統計表）の素材
#   - 共同研究者へのデータ共有
#
# 【実行条件】00_colab_setup.py 完了後（run_all不要）
# ================================================================

import os, json, datetime, zipfile
import pandas as pd
import numpy as np
from google.colab import files

TS        = datetime.datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR   = f'/content/export_pharma_sample_{TS}'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 設定 ─────────────────────────────────────────────────────
SAMPLE_N   = 50_000    # df のサンプル行数（変更可）
RANDOM_SEED = 42
ZIP_OUTPUT = True      # True にすると最後に ZIP にまとめてダウンロード

print(f"Output dir : {OUT_DIR}")
print(f"Sample size: {SAMPLE_N:,} rows from df")
print(f"Timestamp  : {TS}\n")

saved = []

def note(path):
    mb = os.path.getsize(path) / 1024 / 1024
    saved.append(path)
    print(f"  ✓ {os.path.basename(path):50s} {mb:6.1f} MB")


# ----------------------------------------------------------------
# 1. df サンプル
# ----------------------------------------------------------------
print("[1/4] df sample")

df_obj = globals().get('df')
if df_obj is not None:
    n_sample = min(SAMPLE_N, len(df_obj))
    df_sample = df_obj.sample(n_sample, random_state=RANDOM_SEED).reset_index(drop=True)
    p = f'{OUT_DIR}/df_pharma_sample_{n_sample}.csv'
    df_sample.to_csv(p, index=False, encoding='utf-8-sig')
    note(p)
    print(f"     ({n_sample:,} / {len(df_obj):,} rows, {len(df_obj.columns)} cols)")
else:
    print("  [skip] df not found")
    df_sample = None


# ----------------------------------------------------------------
# 2. df_panel 全件
# ----------------------------------------------------------------
print("\n[2/4] df_panel (full)")

panel_obj = globals().get('df_panel')
if panel_obj is not None:
    p = f'{OUT_DIR}/df_panel_full.csv'
    panel_obj.to_csv(p, index=False, encoding='utf-8-sig')
    note(p)
    print(f"     ({len(panel_obj):,} rows, {len(panel_obj.columns)} cols)")
else:
    print("  [skip] df_panel not found")


# ----------------------------------------------------------------
# 3. 記述統計
# ----------------------------------------------------------------
print("\n[3/4] Descriptive statistics")

# 分析に使う主要変数
ANALYSIS_VARS = [
    'has_reject_reason', 'log_citation_count', 'citation_count',
    'self_cite_ratio', 'lag_applicant_hhi', 'log_lag_density',
    'claim1', 'n_inventors', 'year_id',
]

if df_obj is not None:
    avail = [v for v in ANALYSIS_VARS if v in df_obj.columns]
    desc  = df_obj[avail].describe(percentiles=[.10, .25, .50, .75, .90]).T
    desc.index.name = 'variable'
    desc = desc.reset_index()

    # 追加統計
    extras = []
    for v in avail:
        col = df_obj[v].dropna()
        extras.append({
            'variable': v,
            'skewness': round(float(col.skew()), 4),
            'kurtosis': round(float(col.kurt()), 4),
            'n_nonzero': int((col != 0).sum()),
            'pct_zero':  round(float((col == 0).mean() * 100), 2),
            'n_missing': int(df_obj[v].isna().sum()),
        })
    extras_df = pd.DataFrame(extras)
    desc_full = desc.merge(extras_df, on='variable', how='left')

    p = f'{OUT_DIR}/descriptive_stats.csv'
    desc_full.to_csv(p, index=False, encoding='utf-8-sig')
    note(p)

    # df_panel の記述統計
    PANEL_VARS = ['log_applications', 'log_lag_patent_stock',
                  'log_lag_grant_stock', 'avg_self_cite_ratio',
                  'lag_applicant_hhi', 'log_lag_density']
    if panel_obj is not None:
        p_avail = [v for v in PANEL_VARS if v in panel_obj.columns]
        panel_desc = panel_obj[p_avail].describe(
            percentiles=[.10, .25, .50, .75, .90]).T.reset_index()
        panel_desc.columns.name = None
        panel_desc = panel_desc.rename(columns={'index': 'variable'})
        p2 = f'{OUT_DIR}/descriptive_stats_panel.csv'
        panel_desc.to_csv(p2, index=False, encoding='utf-8-sig')
        note(p2)

    # 相関行列
    corr_vars = [v for v in avail if df_obj[v].dtype in [float, int, 'float64', 'int64']]
    corr_mat  = df_obj[corr_vars].corr().round(4)
    p3 = f'{OUT_DIR}/correlation_matrix.csv'
    corr_mat.to_csv(p3, encoding='utf-8-sig')
    note(p3)

    # 年別・フィールド別の集計
    if 'year_id' in df_obj.columns:
        yearly = df_obj.groupby('year_id').agg(
            n_apps            = ('app_id' if 'app_id' in df_obj.columns else 'self_cite_ratio', 'count'),
            mean_self_cite    = ('self_cite_ratio', 'mean'),
            mean_reject       = ('has_reject_reason', 'mean'),
            mean_log_cite     = ('log_citation_count', 'mean'),
        ).reset_index()
        p4 = f'{OUT_DIR}/yearly_aggregates.csv'
        yearly.to_csv(p4, index=False, encoding='utf-8-sig')
        note(p4)
else:
    print("  [skip] df not found")


# ----------------------------------------------------------------
# 4. データ辞書（変数説明）
# ----------------------------------------------------------------
print("\n[4/4] Data dictionary")

data_dict = [
    # application level
    {'table':'df','variable':'app_id',            'type':'str',   'description':'出願番号 (IIP: ida)'},
    {'table':'df','variable':'year_id',           'type':'int',   'description':'出願年 (adateの先頭4文字)'},
    {'table':'df','variable':'ipc',               'type':'str',   'description':'IPCクラス (class1)'},
    {'table':'df','variable':'field_id',          'type':'int',   'description':'IPC先頭4文字のカテゴリコード'},
    {'table':'df','variable':'claim1',            'type':'float', 'description':'請求項数'},
    {'table':'df','variable':'applicant_id',      'type':'str',   'description':'出願人ID (idname; NaN時はapp_idで代替)'},
    {'table':'df','variable':'is_pharma',         'type':'int',   'description':'医薬品ダミー (A61/C07/C12)'},
    {'table':'df','variable':'has_reject_reason', 'type':'int',   'description':'拒絶関連引用あり (reason 19/22/89)'},
    {'table':'df','variable':'citation_count',    'type':'float', 'description':'examiner引用件数合計'},
    {'table':'df','variable':'log_citation_count','type':'float', 'description':'log(1 + citation_count)'},
    {'table':'df','variable':'self_cite_ratio',   'type':'float', 'description':'自己引用比率 = 同一出願人引用 / 総引用'},
    {'table':'df','variable':'n_inventors',       'type':'float', 'description':'発明者数 (inventor.txtのida_seqのunique数)'},
    {'table':'df','variable':'lag_applicant_hhi', 'type':'float', 'description':'前年フィールド内出願人HHI'},
    {'table':'df','variable':'log_lag_density',   'type':'float', 'description':'log(1 + 前年フィールド内出願件数)'},
    # panel level
    {'table':'df_panel','variable':'applicant_id',          'type':'str',   'description':'出願人ID'},
    {'table':'df_panel','variable':'field_id',              'type':'int',   'description':'フィールドID'},
    {'table':'df_panel','variable':'year_id',               'type':'int',   'description':'年'},
    {'table':'df_panel','variable':'n_filed',               'type':'int',   'description':'当該年の出願件数'},
    {'table':'df_panel','variable':'patent_stock',          'type':'float', 'description':'累積出願ストック (δ=0, t-1まで)'},
    {'table':'df_panel','variable':'grant_stock',           'type':'float', 'description':'累積登録ストック (δ=0, t-1まで)'},
    {'table':'df_panel','variable':'log_lag_patent_stock',  'type':'float', 'description':'log(1 + patent_stock)'},
    {'table':'df_panel','variable':'log_lag_grant_stock',   'type':'float', 'description':'log(1 + grant_stock)'},
    {'table':'df_panel','variable':'log_applications',      'type':'float', 'description':'log(1 + n_filed)'},
    {'table':'df_panel','variable':'avg_self_cite_ratio',   'type':'float', 'description':'field-year平均自己引用比率'},
    {'table':'df_panel','variable':'lag_applicant_hhi',     'type':'float', 'description':'前年フィールド内出願人HHI'},
    {'table':'df_panel','variable':'log_lag_density',       'type':'float', 'description':'log(1 + 前年フィールド内出願件数)'},
]
p = f'{OUT_DIR}/data_dictionary.csv'
pd.DataFrame(data_dict).to_csv(p, index=False, encoding='utf-8-sig')
note(p)


# ----------------------------------------------------------------
# ZIP にまとめてダウンロード
# ----------------------------------------------------------------
if ZIP_OUTPUT:
    zip_path = f'/content/export_pharma_sample_{TS}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in saved:
            zf.write(path, os.path.basename(path))
    zip_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\n  📦 ZIP: {os.path.basename(zip_path)}  ({zip_mb:.1f} MB)")
    files.download(zip_path)
    print(f"  ↓ {os.path.basename(zip_path)}")
else:
    print(f"\nDownloading {len(saved)} files ...")
    for path in saved:
        files.download(path)
        print(f"  ↓ {os.path.basename(path)}")

print("\n✓ Export B complete.")
