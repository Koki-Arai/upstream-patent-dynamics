# ================================================================
# 98c_export_pharma_full.py
# ローカル保存 C — 医薬品データ全件（分割ダウンロード）
# ================================================================
# 【内容】df（905,408行）を複数ファイルに分割してダウンロード
# 【容量】1ファイルあたり約 30〜50 MB（合計 200MB前後）
# 【用途】完全なデータをローカルに保存して再分析・別ツールで利用
#
# 【分割の仕組み】
#   - CHUNK_SIZE 行ごとに CSV で出力
#   - ローカルで結合: pd.concat([pd.read_csv(f) for f in files])
#
# 【注意】全件ダウンロードは時間がかかります。
#         通常は Export B (サンプル) で十分です。
#         必要な場合のみ実行してください。
# ================================================================

import os, datetime, zipfile
import pandas as pd
import numpy as np
from google.colab import files

TS       = datetime.datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR  = f'/content/export_pharma_full_{TS}'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 設定 ─────────────────────────────────────────────────────
CHUNK_SIZE    = 150_000    # 1ファイルの行数（目安 40MB/file）
COMPRESS_ZIP  = True       # True: ZIP圧縮してダウンロード
                           # False: 分割CSVを個別にダウンロード

# 保存する列（不要な列を除外してサイズを削減）
DF_COLS = [
    'app_id', 'year_id', 'ipc', 'field_id', 'claim1',
    'applicant_id', 'is_pharma',
    'has_reject_reason', 'citation_count', 'log_citation_count',
    'self_cite_ratio', 'n_inventors',
    'lag_applicant_hhi', 'log_lag_density',
]
PANEL_COLS = [
    'applicant_id', 'field_id', 'year_id', 'n_filed',
    'patent_stock', 'grant_stock',
    'log_lag_patent_stock', 'log_lag_grant_stock',
    'log_applications', 'avg_self_cite_ratio',
    'lag_applicant_hhi', 'log_lag_density',
]

print(f"Output dir : {OUT_DIR}")
print(f"Chunk size : {CHUNK_SIZE:,} rows per file")
print(f"Timestamp  : {TS}\n")

saved_zips = []


# ----------------------------------------------------------------
# ヘルパー: DataFrame を分割して CSV/ZIP 保存
# ----------------------------------------------------------------
def export_chunked(df, prefix, col_list, chunk_size, out_dir, compress):
    """DataFrame を chunk_size 行ずつ分割して保存 → ダウンロード。"""
    avail_cols = [c for c in col_list if c in df.columns]
    df_export  = df[avail_cols].reset_index(drop=True)
    n_chunks   = (len(df_export) + chunk_size - 1) // chunk_size
    chunk_paths = []

    print(f"  {len(df_export):,} rows × {len(avail_cols)} cols → {n_chunks} files")

    for i in range(n_chunks):
        start = i * chunk_size
        end   = min((i + 1) * chunk_size, len(df_export))
        chunk = df_export.iloc[start:end]
        fname = f'{prefix}_part{i+1:02d}of{n_chunks:02d}.csv'
        path  = f'{out_dir}/{fname}'
        chunk.to_csv(path, index=False, encoding='utf-8-sig')
        mb = os.path.getsize(path) / 1024 / 1024
        chunk_paths.append(path)
        print(f"    ✓ {fname:55s} {mb:5.1f} MB  (rows {start:,}–{end:,})")

    if compress:
        zip_path = f'/content/{prefix}_{TS}.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in chunk_paths:
                zf.write(p, os.path.basename(p))
            # 結合用README
            readme = (
                f"# {prefix} — Full pharmaceutical dataset\n"
                f"# Generated: {TS}\n"
                f"# Total rows: {len(df_export):,}\n"
                f"# Files: {n_chunks} parts\n\n"
                f"# To recombine in Python:\n"
                f"# import pandas as pd, glob\n"
                f"# df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('{prefix}_part*.csv'))],\n"
                f"#                ignore_index=True)\n"
            )
            zf.writestr(f'{prefix}_README.txt', readme)
        zip_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(f"\n  📦 {os.path.basename(zip_path):55s} {zip_mb:5.1f} MB")
        saved_zips.append(zip_path)
        return zip_path
    else:
        saved_zips.extend(chunk_paths)
        return chunk_paths


# ----------------------------------------------------------------
# 1. df 全件（application level）
# ----------------------------------------------------------------
print("[1/2] df — application level (full)")
df_obj = globals().get('df')
if df_obj is not None:
    export_chunked(df_obj, 'df_pharma_full', DF_COLS,
                   CHUNK_SIZE, OUT_DIR, COMPRESS_ZIP)
else:
    print("  [skip] df not found")


# ----------------------------------------------------------------
# 2. df_panel 全件（firm-field-year）
# ----------------------------------------------------------------
print("\n[2/2] df_panel — firm-field-year (full)")
panel_obj = globals().get('df_panel')
if panel_obj is not None:
    # panel は比較的小さいので1ファイル
    export_chunked(panel_obj, 'df_panel_full', PANEL_COLS,
                   len(panel_obj), OUT_DIR, COMPRESS_ZIP)
else:
    print("  [skip] df_panel not found")


# ----------------------------------------------------------------
# 結合用スクリプトを同梱
# ----------------------------------------------------------------
recombine_script = f'''\
# recombine_data.py
# 分割ファイルを結合して元の DataFrame を復元するスクリプト
# Generated: {TS}

import pandas as pd
import glob

# df (application level)
df_files = sorted(glob.glob('df_pharma_full_part*.csv'))
if df_files:
    df = pd.concat([pd.read_csv(f) for f in df_files], ignore_index=True)
    print(f"df: {{len(df):,}} rows x {{len(df.columns)}} cols")
    # df.to_pickle('df_pharma_full.pkl')  # pkl保存する場合

# df_panel (firm-field-year)
panel_files = sorted(glob.glob('df_panel_full_part*.csv'))
if panel_files:
    df_panel = pd.concat([pd.read_csv(f) for f in panel_files], ignore_index=True)
    print(f"df_panel: {{len(df_panel):,}} rows x {{len(df_panel.columns)}} cols")
    # df_panel.to_pickle('df_panel_full.pkl')
'''
script_path = f'{OUT_DIR}/recombine_data.py'
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(recombine_script)
print(f"\n  ✓ {'recombine_data.py':55s}  (結合スクリプト)")
if COMPRESS_ZIP:
    # 既存ZIPに追記はできないので単独でダウンロード
    saved_zips.append(script_path)


# ----------------------------------------------------------------
# ダウンロード
# ----------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Downloading {len(saved_zips)} file(s) ...")
print('='*55)
for path in saved_zips:
    files.download(path)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ↓ {os.path.basename(path):50s} {mb:6.1f} MB")

print("\n✓ Export C complete.")
print("  ローカルで結合するには recombine_data.py を参照してください。")
