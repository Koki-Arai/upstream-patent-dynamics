# ================================================================
# 00_colab_setup.py  (v5 — 医薬品絞り込み・メモリ節約版)
# Patent Portfolio Dynamics — Colab 事前セットアップ
# ================================================================
# 【RAMクラッシュ対策】
#   全1085万件のapをメモリに乗せず、各ファイルを chunk 単位で
#   読み込みながら医薬品(A61/C07/C12)のみ抽出する。
#   ccも医薬品出願のIDセットで絞り込む（約4800万→数百万行に削減）。
# ================================================================


# ----------------------------------------------------------------
# CELL 0: ファイル確認（前回と同じ。確認済みならスキップ可）
# ----------------------------------------------------------------
import os, pandas as pd

IIP_DIR = '/content'
SEP, ENCODING = '\t', 'utf-8'

print("=== Column names (ap / cc / applicant / inventor / hr) ===")
for prefix in ['ap', 'cc', 'applicant', 'inventor', 'hr']:
    path = f'{IIP_DIR}/{prefix}_1990s.txt'
    if not os.path.exists(path):
        continue
    s = pd.read_csv(path, sep=SEP, encoding=ENCODING, nrows=1, dtype=str)
    print(f"  {prefix:12s}: {list(s.columns)}")


# ----------------------------------------------------------------
# CELL 1: Google Drive マウント（保存用）
# ----------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/patent_analysis_cache'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save dir: {SAVE_DIR}")


# ----------------------------------------------------------------
# CELL 2: パッケージ（バージョン固定）
# ----------------------------------------------------------------
import subprocess, sys
for pkg in ['numpy==1.26.4','scipy==1.13.1',
            'statsmodels==0.14.2','pandas==2.2.2']:
    subprocess.check_call(
        [sys.executable,'-m','pip','install','--quiet',pkg])
    print(f'  ✓ {pkg}')
print('All packages ready. Restart runtime if first install.')


# ----------------------------------------------------------------
# CELL 3: バージョン確認
# ----------------------------------------------------------------
import numpy as np, pandas as pd, statsmodels, scipy
print(f'numpy {np.__version__}  pandas {pd.__version__}  '
      f'statsmodels {statsmodels.__version__}  scipy {scipy.__version__}')
from packaging import version
assert version.parse(statsmodels.__version__) >= version.parse('0.14')
assert version.parse(scipy.__version__)       >= version.parse('1.11')
print('✓ Version requirements met.')


# ----------------------------------------------------------------
# CELL 4: IIP データ読み込み（医薬品絞り込み・chunk処理）
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import os, gc, warnings
warnings.filterwarnings('ignore')

IIP_DIR  = '/content'
SAVE_DIR = '/content/drive/MyDrive/patent_analysis_cache'
DECADES  = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000   # 1チャンクあたり50万行

# 医薬品IPCフィルタ（論文 Section 4.1 に対応）
PHARMA_IPC     = ['A61', 'C07', 'C12']
VALID_REASONS  = [19, 22, 31, 75, 89, 93]
REJECT_REASONS = [19, 22, 89]


# ======== [1/5] ap — chunk読み込みで医薬品のみ抽出 ========
print("[1/5] Loading ap (pharma only, chunked) ...")

ap_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/ap_{dec}.txt'
    if not os.path.exists(path):
        continue
    n_pharma = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','adate','class1','claim1'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        mask = chunk['class1'].str[:3].isin(PHARMA_IPC)
        pharma = chunk[mask].copy()
        n_pharma += len(pharma)
        ap_frames.append(pharma)
    print(f"  ✓ ap_{dec}.txt  → {n_pharma:,} pharma rows")

ap = pd.concat(ap_frames, ignore_index=True)
del ap_frames; gc.collect()

# 列整形
ap = ap.rename(columns={'ida':'app_id','adate':'app_date',
                         'class1':'ipc','claim1':'claim1'})
ap['year_id'] = pd.to_numeric(ap['app_date'].str[:4], errors='coerce')
ap = ap.dropna(subset=['year_id'])
ap['year_id'] = ap['year_id'].astype(int)
ap['claim1']  = pd.to_numeric(ap['claim1'], errors='coerce').fillna(0)
ap['field_id'] = ap['ipc'].str[:4].astype('category').cat.codes
ap['is_pharma'] = 1   # すでに医薬品のみ

print(f"  ap pharma total: {len(ap):,} rows")
print(f"  year range: {ap['year_id'].min()} – {ap['year_id'].max()}")

# 医薬品出願IDセット（cc絞り込み用）
pharma_ids = set(ap['app_id'].unique())
print(f"  pharma app_id set: {len(pharma_ids):,} IDs")


# ======== [2/5] applicant — 医薬品出願のみ ========
print("\n[2/5] Loading applicant (pharma only, chunked) ...")

appl_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/applicant_{dec}.txt'
    if not os.path.exists(path):
        continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','seq','idname'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        chunk['seq'] = pd.to_numeric(chunk['seq'], errors='coerce')
        # 筆頭出願人(seq=1)かつ医薬品出願のみ
        mask = (chunk['seq'] == 1) & (chunk['ida'].isin(pharma_ids))
        filtered = chunk[mask][['ida','idname']].copy()
        n += len(filtered)
        appl_frames.append(filtered)
    print(f"  ✓ applicant_{dec}.txt  → {n:,} rows")

appl = pd.concat(appl_frames, ignore_index=True)
del appl_frames; gc.collect()
appl = appl.rename(columns={'ida':'app_id','idname':'applicant_id'})
appl = appl.drop_duplicates('app_id', keep='first')

# applicant_id が NaN の場合は app_id 自体を代替IDとして使用
appl['applicant_id'] = appl['applicant_id'].fillna(appl['app_id'])

ap = ap.merge(appl, on='app_id', how='left')
# マッチしなかった出願はapp_idで代替
ap['applicant_id'] = ap['applicant_id'].fillna(ap['app_id'])
print(f"  applicant matched: {(ap['applicant_id'] != ap['app_id']).mean():.1%} (idname)")


# ======== [3/5] cc — 医薬品出願が citing または cited に含まれる行のみ ========
print("\n[3/5] Loading cc (pharma-related only, chunked) ...")

cc_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/cc_{dec}.txt'
    if not os.path.exists(path):
        continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['citing','cited','reason'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        chunk['reason'] = pd.to_numeric(chunk['reason'], errors='coerce')
        # valid reason かつ citing が医薬品出願
        mask = (chunk['reason'].isin(VALID_REASONS) &
                chunk['citing'].isin(pharma_ids))
        filtered = chunk[mask].copy()
        n += len(filtered)
        cc_frames.append(filtered)
    print(f"  ✓ cc_{dec}.txt  → {n:,} pharma-citing rows")

cc = pd.concat(cc_frames, ignore_index=True)
del cc_frames; gc.collect()
cc['is_reject'] = cc['reason'].isin(REJECT_REASONS).astype(int)
print(f"  cc total: {len(cc):,} rows")

# 出願ごとの引用集計
ci_agg = (cc.groupby('citing')
            .agg(citation_count   =('cited',    'count'),
                 has_reject_reason=('is_reject', 'max'))
            .reset_index()
            .rename(columns={'citing':'app_id'}))

# 自己引用比率
ap_id_appl = ap[['app_id','applicant_id']].drop_duplicates('app_id')
cc_self = (cc
    .merge(ap_id_appl.rename(columns={'app_id':'cited',
                                       'applicant_id':'cited_appl'}),
           on='cited', how='left')
    .merge(ap_id_appl.rename(columns={'app_id':'citing',
                                       'applicant_id':'citing_appl'}),
           on='citing', how='left')
)
cc_self['is_self'] = (
    cc_self['cited_appl'] == cc_self['citing_appl']).astype(int)
self_cite_agg = (cc_self.groupby('citing')
                        .agg(self_cite_count=('is_self','sum'),
                             total_cite     =('is_self','count'))
                        .reset_index()
                        .rename(columns={'citing':'app_id'}))
self_cite_agg['self_cite_ratio'] = (
    self_cite_agg['self_cite_count']
    / self_cite_agg['total_cite'].clip(lower=1))
del cc, cc_self; gc.collect()


# ======== [4/5] inventor — 医薬品出願のみ ========
print("\n[4/5] Loading inventor (pharma only, chunked) ...")

inv_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/inventor_{dec}.txt'
    if not os.path.exists(path):
        continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','ida_seq'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        mask = chunk['ida'].isin(pharma_ids)
        filtered = chunk[mask].copy()
        n += len(filtered)
        inv_frames.append(filtered)
    print(f"  ✓ inventor_{dec}.txt  → {n:,} rows")

inv = pd.concat(inv_frames, ignore_index=True)
del inv_frames; gc.collect()
inv = inv.rename(columns={'ida':'app_id','ida_seq':'inv_uid'})
n_inv = (inv.groupby('app_id')['inv_uid']
            .nunique().reset_index()
            .rename(columns={'inv_uid':'n_inventors'}))
del inv; gc.collect()


# ======== [5/5] hr — 医薬品出願のみ ========
print("\n[5/5] Loading hr (pharma only, chunked) ...")

hr_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/hr_{dec}.txt'
    if not os.path.exists(path):
        continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','seq'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        chunk['seq'] = pd.to_numeric(chunk['seq'], errors='coerce')
        mask = (chunk['seq'] == 1) & (chunk['ida'].isin(pharma_ids))
        filtered = chunk[mask][['ida']].copy()
        n += len(filtered)
        hr_frames.append(filtered)
    print(f"  ✓ hr_{dec}.txt  → {n:,} rows")

hr = pd.concat(hr_frames, ignore_index=True)
del hr_frames; gc.collect()
hr = hr.rename(columns={'ida':'app_id'}).drop_duplicates()
hr = hr.merge(ap[['app_id','year_id']], on='app_id', how='left')
hr = hr.rename(columns={'year_id':'grant_year'})
print(f"  hr (granted): {len(hr):,} rows")


# ======== ラグ変数・ストックパネル ========
print("\nBuilding lag variables & stock panel ...")

field_year_apps = (ap.groupby(['field_id','year_id'])
                     .agg(n_apps=('app_id','count')).reset_index())

app_share = (ap.groupby(['field_id','year_id','applicant_id'])
               .size().reset_index(name='cnt')
               .merge(field_year_apps, on=['field_id','year_id']))
app_share['share_sq'] = (app_share['cnt'] / app_share['n_apps']) ** 2
hhi = (app_share.groupby(['field_id','year_id'])['share_sq']
                .sum().reset_index()
                .rename(columns={'share_sq':'applicant_hhi'}))
hhi_lag = hhi.copy()
hhi_lag['year_id'] = hhi_lag['year_id'] + 1
hhi_lag = hhi_lag.rename(columns={'applicant_hhi':'lag_applicant_hhi'})

log_density = field_year_apps.copy()
log_density['log_lag_density'] = np.log1p(log_density['n_apps'])
log_density['year_id'] = log_density['year_id'] + 1
log_density = log_density[['field_id','year_id','log_lag_density']]

firm_field_apps = (
    ap.groupby(['applicant_id','field_id','year_id'])
      .size().reset_index(name='n_filed')
      .sort_values(['applicant_id','field_id','year_id'])
)
firm_field_apps['patent_stock'] = (
    firm_field_apps.groupby(['applicant_id','field_id'])['n_filed']
                   .cumsum().shift(1))

ap_hr = ap[['app_id','applicant_id','field_id']].merge(
    hr[['app_id','grant_year']], on='app_id', how='left')
firm_field_grant = (
    ap_hr.dropna(subset=['grant_year'])
         .assign(grant_year=lambda x: x['grant_year'].astype(int))
         .groupby(['applicant_id','field_id','grant_year'])
         .size().reset_index(name='n_granted')
         .sort_values(['applicant_id','field_id','grant_year'])
)
firm_field_grant['grant_stock'] = (
    firm_field_grant.groupby(['applicant_id','field_id'])['n_granted']
                    .cumsum().shift(1))
firm_field_grant = firm_field_grant.rename(columns={'grant_year':'year_id'})


# ======== データセット結合 ========
print("Building df & df_panel ...")

df = (ap
      .merge(ci_agg,  on='app_id', how='left')
      .merge(self_cite_agg[['app_id','self_cite_ratio']], on='app_id', how='left')
      .merge(n_inv,   on='app_id', how='left')
      .merge(hhi_lag, on=['field_id','year_id'], how='left')
      .merge(log_density, on=['field_id','year_id'], how='left')
)
df['self_cite_ratio']   = df['self_cite_ratio'].fillna(0)
df['citation_count']    = df['citation_count'].fillna(0)
df['has_reject_reason'] = df['has_reject_reason'].fillna(0).astype(int)
df['n_inventors']       = df['n_inventors'].fillna(1)
df['log_citation_count'] = np.log1p(df['citation_count'])

df_panel = (firm_field_apps
    .merge(firm_field_grant[['applicant_id','field_id','year_id','grant_stock']],
           on=['applicant_id','field_id','year_id'], how='left')
    .merge(hhi_lag,     on=['field_id','year_id'], how='left')
    .merge(log_density, on=['field_id','year_id'], how='left')
)
df_panel['log_lag_patent_stock'] = np.log1p(df_panel['patent_stock'].fillna(0))
df_panel['log_lag_grant_stock']  = np.log1p(df_panel['grant_stock'].fillna(0))
df_panel['log_applications']     = np.log1p(df_panel['n_filed'])

sc_fld = (df.groupby(['field_id','year_id'])['self_cite_ratio']
            .mean().reset_index()
            .rename(columns={'self_cite_ratio':'avg_self_cite_ratio'}))
df_panel = df_panel.merge(sc_fld, on=['field_id','year_id'], how='left')

gc.collect()

# ======== 確認 ========
print(f'\n=== Dataset summary ===')
print(f'df  (application level)   : {len(df):,} rows')
print(f'df_panel (firm-field-year): {len(df_panel):,} rows')
print(f'year range                : {df["year_id"].min()} – {df["year_id"].max()}')

required = ['has_reject_reason','log_citation_count','self_cite_ratio',
            'lag_applicant_hhi','log_lag_density','claim1','n_inventors',
            'field_id','year_id']
print('\nColumn check (df):')
for c in required:
    print(f"  {'✓' if c in df.columns else '✗ MISSING'}  {c}")


# ----------------------------------------------------------------
# CELL 5: pickle 保存（Drive）
# ----------------------------------------------------------------
df.to_pickle(f'{SAVE_DIR}/df_application.pkl')
df_panel.to_pickle(f'{SAVE_DIR}/df_panel.pkl')
print(f'\nSaved → {SAVE_DIR}')
print('→ 次は patent_robustness_v2_iip.py を実行してください。')


# ----------------------------------------------------------------
# CELL 6: ランタイム再接続後の高速復帰（必要時にコメント解除）
# ----------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')
# import pandas as pd
# SAVE_DIR = '/content/drive/MyDrive/patent_analysis_cache'
# df       = pd.read_pickle(f'{SAVE_DIR}/df_application.pkl')
# df_panel = pd.read_pickle(f'{SAVE_DIR}/df_panel.pkl')
# print(f'Loaded: df={len(df):,}  df_panel={len(df_panel):,}')
