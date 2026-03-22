"""
pooled_regression_table3.py
============================
Table 3. Pooled Evidence and Scope Conditions
==============================================

【目的】
  Section 4.5・5.4 が説明している pooled regression を推計する。
  医薬品 + 非医薬品の全技術分野を対象に、pharma ダミーとの交差項を入れ、
  scrutiny channel と path dependence が医薬品に固有か一般的かを検証する。

【モデル（Section 4.5 準拠）】

  Scrutiny 方程式（出願レベル）:
    Y_igt = β₁ · self_cite_ratio_igt
          + β₂ · (self_cite_ratio_igt × pharma_g)
          + γ · X_igt                          ← controls
          + α_g + δ_t + ε_igt

  Expansion 方程式（firm-field-year パネル）:
    log_apps_igt = π₁ · log_lag_patent_stock_igt
                 + π₂ · (log_lag_patent_stock_igt × pharma_g)
                 + η · X_igt
                 + α_g + δ_t + ε_igt

  Y = has_reject_reason, log_citation_count
  交差項の係数 β₂ / π₂ が検定の核心:
    有意正  → pharma で効果が強い  → sector-specific mechanism
    非有意  → 汎用メカニズム  → H5 (scope condition) を支持

【出力】
  Table 3 本体（本文挿入用）と Appendix 詳細版を CSV で保存。

【実行条件】
  1. 全技術分野の df_all および df_panel_all が必要。
     → 本スクリプトの STEP 1 で /content/*.txt から新規に構築する。
     → 医薬品 df・df_panel はすでにメモリにあれば再利用する（STEP 1スキップ可）。
  2. linearmodels がインストール済みであること。
     !pip install linearmodels

【所要時間】
  STEP 1（全件読み込み）: 約 20〜40 分（1085万件）
  STEP 2以降（回帰）: 約 5〜10 分
"""

import os, gc, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

IIP_DIR   = '/content'
DECADES   = ['1990s', '2000s', '2010s', '2020s']
SEP, ENCODING = '\t', 'utf-8'
CHUNKSIZE = 500_000
SAVE_DIR  = '/content/drive/MyDrive/patent_analysis_cache'

PHARMA_IPC     = ['A61', 'C07', 'C12']
VALID_REASONS  = [19, 22, 31, 75, 89, 93]
REJECT_REASONS = [19, 22, 89]

# ================================================================
# STEP 0: 既存データの確認
# ================================================================
print("="*60)
print("STEP 0: Checking available data in memory")
print("="*60)

df_pharma    = globals().get('df')          # 医薬品 application-level
panel_pharma = globals().get('df_panel')    # 医薬品 firm-field-year

_have_pharma  = df_pharma is not None
_have_panel   = panel_pharma is not None

print(f"  df (pharma)      : {'✓ available' if _have_pharma else '✗ not found'}"
      + (f"  ({len(df_pharma):,} rows)" if _have_pharma else ""))
print(f"  df_panel (pharma): {'✓ available' if _have_panel else '✗ not found'}"
      + (f"  ({len(panel_pharma):,} rows)" if _have_panel else ""))

# ================================================================
# STEP 1: 全技術分野データの構築
# ================================================================
# 【設計】
#   - 全件（約1085万件）を chunk で読む
#   - pharma ダミーを付与して df_pharma とスタック
#   - cc・applicant・inventor・hr は全件が必要なため再読み込み
#   - メモリ節約のため application-level の必要最小限の列のみ保持
# ================================================================
print("\n" + "="*60)
print("STEP 1: Building full-sample dataset (all technology fields)")
print("="*60)
print("  [Note] This step loads ~10 million+ records. Estimated time: 20-40 min.")
print("  If RAM is tight, reduce CHUNKSIZE or use SAMPLE_FRAC below.\n")

# サンプリング設定（RAM が足りない場合は 0.1〜0.3 に下げる）
# 本番は 1.0 推奨。0.3 でも問題の趣旨は再現できる。
SAMPLE_FRAC = 1.0   # ← 必要なら 0.3 に変更

# ---- [1/4] ap 全技術分野 ----
print("[1/4] Loading ap (all fields, chunked) ...")

ap_all_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/ap_{dec}.txt'
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','adate','class1','claim1'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        if SAMPLE_FRAC < 1.0:
            chunk = chunk.sample(frac=SAMPLE_FRAC, random_state=42)
        ap_all_frames.append(chunk)
        n += len(chunk)
    print(f"  ✓ ap_{dec}.txt → {n:,} rows")

ap_all = pd.concat(ap_all_frames, ignore_index=True)
del ap_all_frames; gc.collect()

ap_all = ap_all.rename(columns={'ida':'app_id','adate':'app_date',
                                 'class1':'ipc','claim1':'claim1'})
ap_all['year_id'] = pd.to_numeric(ap_all['app_date'].str[:4], errors='coerce')
ap_all = ap_all.dropna(subset=['year_id'])
ap_all['year_id'] = ap_all['year_id'].astype(int)
ap_all['claim1']  = pd.to_numeric(ap_all['claim1'], errors='coerce').fillna(0)
ap_all['field_id'] = ap_all['ipc'].str[:4].astype('category').cat.codes
ap_all['is_pharma'] = ap_all['ipc'].str[:3].isin(PHARMA_IPC).astype(int)

pharma_ids_all = set(ap_all[ap_all['is_pharma']==1]['app_id'].unique())
all_ids        = set(ap_all['app_id'].unique())
print(f"\n  ap total    : {len(ap_all):,} rows")
print(f"  pharma share: {ap_all['is_pharma'].mean():.1%}  ({ap_all['is_pharma'].sum():,})")

# ---- [2/4] applicant ----
print("\n[2/4] Loading applicant (all, chunked) ...")

appl_all_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/applicant_{dec}.txt'
    if not os.path.exists(path): continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','seq','idname'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        chunk['seq'] = pd.to_numeric(chunk['seq'], errors='coerce')
        mask = (chunk['seq'] == 1) & (chunk['ida'].isin(all_ids))
        filtered = chunk[mask][['ida','idname']].copy()
        n += len(filtered)
        appl_all_frames.append(filtered)
    print(f"  ✓ applicant_{dec}.txt → {n:,} rows")

appl_all = pd.concat(appl_all_frames, ignore_index=True)
del appl_all_frames; gc.collect()
appl_all = appl_all.rename(columns={'ida':'app_id','idname':'applicant_id'})
appl_all = appl_all.drop_duplicates('app_id', keep='first')
appl_all['applicant_id'] = appl_all['applicant_id'].fillna(appl_all['app_id'])
ap_all = ap_all.merge(appl_all, on='app_id', how='left')
ap_all['applicant_id'] = ap_all['applicant_id'].fillna(ap_all['app_id'])
del appl_all; gc.collect()

# ---- [3/4] cc 全技術分野（citing が全件対象）----
print("\n[3/4] Loading cc (all citing, chunked) ...")

cc_all_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/cc_{dec}.txt'
    if not os.path.exists(path): continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['citing','cited','reason'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        chunk['reason'] = pd.to_numeric(chunk['reason'], errors='coerce')
        mask = (chunk['reason'].isin(VALID_REASONS) &
                chunk['citing'].isin(all_ids))
        filtered = chunk[mask].copy()
        n += len(filtered)
        cc_all_frames.append(filtered)
    print(f"  ✓ cc_{dec}.txt → {n:,} valid rows")

cc_all = pd.concat(cc_all_frames, ignore_index=True)
del cc_all_frames; gc.collect()
cc_all['is_reject'] = cc_all['reason'].isin(REJECT_REASONS).astype(int)
print(f"  cc total: {len(cc_all):,} rows")

# citation 集計
ci_all = (cc_all.groupby('citing')
                .agg(citation_count   =('cited',     'count'),
                     has_reject_reason=('is_reject',  'max'))
                .reset_index()
                .rename(columns={'citing':'app_id'}))

# 自己引用比率
ap_id_appl_all = ap_all[['app_id','applicant_id']].drop_duplicates('app_id')
cc_self_all = (cc_all
    .merge(ap_id_appl_all.rename(columns={'app_id':'cited',
                                           'applicant_id':'cited_appl'}),
           on='cited', how='left')
    .merge(ap_id_appl_all.rename(columns={'app_id':'citing',
                                           'applicant_id':'citing_appl'}),
           on='citing', how='left'))
cc_self_all['is_self'] = (cc_self_all['cited_appl'] == cc_self_all['citing_appl']).astype(int)
self_all_agg = (cc_self_all.groupby('citing')
                            .agg(self_cite_count=('is_self','sum'),
                                 total_cite     =('is_self','count'))
                            .reset_index()
                            .rename(columns={'citing':'app_id'}))
self_all_agg['self_cite_ratio'] = (
    self_all_agg['self_cite_count'] / self_all_agg['total_cite'].clip(lower=1))
del cc_all, cc_self_all; gc.collect()

# ---- [4/4] inventor ----
print("\n[4/4] Loading inventor (all, chunked) ...")

inv_all_frames = []
for dec in DECADES:
    path = f'{IIP_DIR}/inventor_{dec}.txt'
    if not os.path.exists(path): continue
    n = 0
    for chunk in pd.read_csv(path, sep=SEP, encoding=ENCODING,
                              usecols=['ida','ida_seq'],
                              dtype=str, chunksize=CHUNKSIZE,
                              low_memory=False):
        mask = chunk['ida'].isin(all_ids)
        filtered = chunk[mask].copy()
        n += len(filtered)
        inv_all_frames.append(filtered)
    print(f"  ✓ inventor_{dec}.txt → {n:,} rows")

inv_all = pd.concat(inv_all_frames, ignore_index=True)
del inv_all_frames; gc.collect()
inv_all = inv_all.rename(columns={'ida':'app_id','ida_seq':'inv_uid'})
n_inv_all = (inv_all.groupby('app_id')['inv_uid']
                    .nunique().reset_index()
                    .rename(columns={'inv_uid':'n_inventors'}))
del inv_all; gc.collect()

# ---- ラグ変数 ----
print("\nBuilding lag variables ...")

fy_apps_all = (ap_all.groupby(['field_id','year_id'])
                     .agg(n_apps=('app_id','count')).reset_index())
app_share_all = (ap_all.groupby(['field_id','year_id','applicant_id'])
                        .size().reset_index(name='cnt')
                        .merge(fy_apps_all, on=['field_id','year_id']))
app_share_all['share_sq'] = (app_share_all['cnt'] / app_share_all['n_apps']) ** 2
hhi_all = (app_share_all.groupby(['field_id','year_id'])['share_sq']
                        .sum().reset_index()
                        .rename(columns={'share_sq':'applicant_hhi'}))
hhi_lag_all = hhi_all.copy()
hhi_lag_all['year_id'] = hhi_lag_all['year_id'] + 1
hhi_lag_all = hhi_lag_all.rename(columns={'applicant_hhi':'lag_applicant_hhi'})

log_density_all = fy_apps_all.copy()
log_density_all['log_lag_density'] = np.log1p(log_density_all['n_apps'])
log_density_all['year_id'] = log_density_all['year_id'] + 1
log_density_all = log_density_all[['field_id','year_id','log_lag_density']]

# df_all（全技術分野 application-level）
df_all = (ap_all
          .merge(ci_all,  on='app_id', how='left')
          .merge(self_all_agg[['app_id','self_cite_ratio']], on='app_id', how='left')
          .merge(n_inv_all, on='app_id', how='left')
          .merge(hhi_lag_all,   on=['field_id','year_id'], how='left')
          .merge(log_density_all, on=['field_id','year_id'], how='left'))
df_all['self_cite_ratio']   = df_all['self_cite_ratio'].fillna(0)
df_all['citation_count']    = df_all['citation_count'].fillna(0)
df_all['has_reject_reason'] = df_all['has_reject_reason'].fillna(0).astype(int)
df_all['n_inventors']       = df_all['n_inventors'].fillna(1)
df_all['log_citation_count'] = np.log1p(df_all['citation_count'])

# df_panel_all（全技術分野 firm-field-year）
firm_field_apps_all = (
    ap_all.groupby(['applicant_id','field_id','year_id'])
          .size().reset_index(name='n_filed')
          .sort_values(['applicant_id','field_id','year_id']))
firm_field_apps_all['patent_stock'] = (
    firm_field_apps_all.groupby(['applicant_id','field_id'])['n_filed']
                       .cumsum().shift(1))
df_panel_all = (firm_field_apps_all
    .merge(hhi_lag_all,    on=['field_id','year_id'], how='left')
    .merge(log_density_all, on=['field_id','year_id'], how='left'))
df_panel_all['log_lag_patent_stock'] = np.log1p(df_panel_all['patent_stock'].fillna(0))
df_panel_all['log_applications']     = np.log1p(df_panel_all['n_filed'])
sc_fld_all = (df_all.groupby(['field_id','year_id'])['self_cite_ratio']
                    .mean().reset_index()
                    .rename(columns={'self_cite_ratio':'avg_self_cite_ratio'}))
df_panel_all = df_panel_all.merge(sc_fld_all, on=['field_id','year_id'], how='left')
df_panel_all['is_pharma'] = df_panel_all['field_id'].isin(
    df_all[df_all['is_pharma']==1]['field_id'].unique()).astype(int)

gc.collect()
print(f"\n  df_all      : {len(df_all):,} rows")
print(f"  df_panel_all: {len(df_panel_all):,} rows")
print(f"  pharma share (df_all): {df_all['is_pharma'].mean():.1%}")

# ================================================================
# STEP 1.5: 回帰用サンプルの作成（RAM節約）
# ================================================================
# AbsorbingLS は大規模データで大量RAMを消費するため、
# pharma 全件 ＋ 非pharma ランダム50万件 に絞った回帰サンプルを作成する。
# 交差項の識別には十分なサンプルサイズ。
# ================================================================
REG_NONPHARMA_N = 500_000   # 非pharmaのサンプル数（変更可）

df_pharma_reg    = df_all[df_all['is_pharma']==1].copy()
df_nonpharma_all = df_all[df_all['is_pharma']==0]
df_nonpharma_reg = df_nonpharma_all.sample(
    min(REG_NONPHARMA_N, len(df_nonpharma_all)), random_state=42)
df_reg = pd.concat([df_pharma_reg, df_nonpharma_reg], ignore_index=True)
del df_nonpharma_all; gc.collect()

print(f"\n  Regression sample:")
print(f"    pharma    : {len(df_pharma_reg):,} rows (all)")
print(f"    non-pharma: {len(df_nonpharma_reg):,} rows (random sample)")
print(f"    total     : {len(df_reg):,} rows")

# panel も同様に絞る
panel_pharma_reg    = df_panel_all[df_panel_all['is_pharma']==1].copy()
panel_nonpharma_all = df_panel_all[df_panel_all['is_pharma']==0]
panel_nonpharma_reg = panel_nonpharma_all.sample(
    min(REG_NONPHARMA_N, len(panel_nonpharma_all)), random_state=42)
df_panel_reg = pd.concat([panel_pharma_reg, panel_nonpharma_reg], ignore_index=True)
del panel_nonpharma_all; gc.collect()

print(f"  Panel regression sample: {len(df_panel_reg):,} rows")

# 交差項の構築（df_reg / df_panel_reg ベース）
df_reg['sc_x_pharma']      = df_reg['self_cite_ratio'] * df_reg['is_pharma']
df_panel_reg['ps_x_pharma'] = df_panel_reg['log_lag_patent_stock'] * df_panel_reg['is_pharma']

# ================================================================
# STEP 2: Pooled 回帰
# ================================================================
print("\n" + "="*60)
print("STEP 2: Pooled regressions with pharmaceutical interaction terms")
print("="*60)

from linearmodels.iv.absorbing import AbsorbingLS

def run_absorbing_pooled(df_est, outcome, xvars, cluster_col='field_id'):
    """AbsorbingLS: field_id + year_id の FE を吸収。"""
    absorb_cols = [cluster_col, 'year_id']
    reg_vars = list(set([outcome] + xvars + absorb_cols))
    d = df_est[reg_vars].dropna().copy()
    d[cluster_col] = d[cluster_col].astype('category')
    d['year_id']   = d['year_id'].astype('category')
    mod = AbsorbingLS(dependent=d[outcome],
                      exog=d[xvars],
                      absorb=d[absorb_cols])
    return mod.fit(cov_type='clustered',
                   clusters=d[cluster_col].cat.codes.values)

def extract_coef(fit, varname, rename=None):
    vn = rename or varname
    try:
        p  = fit.params
        se = fit.std_errors
        pv = fit.pvalues
        ci = fit.conf_int()
        for a in [p, se, pv]:
            if hasattr(a,'iloc') and a.ndim>1: a = a.iloc[:,0]
        if hasattr(ci,'iloc') and ci.ndim>1:
            ci_lo, ci_hi = ci.iloc[:,0], ci.iloc[:,1]
        else:
            ci_lo = ci_hi = pd.Series(dtype=float)
        return {
            'variable': vn,
            'coef':  round(float(p[varname]), 4),
            'se':    round(float(se[varname]), 4),
            'pval':  round(float(pv[varname]), 4),
            'ci_lo': round(float(ci_lo[varname]), 4) if varname in ci_lo.index else None,
            'ci_hi': round(float(ci_hi[varname]), 4) if varname in ci_hi.index else None,
        }
    except Exception as e:
        return {'variable': vn, 'coef': None, 'se': None, 'pval': None, 'note': str(e)}

# 交差項は STEP 1.5 で構築済み

BASE_X_SCR = ['lag_applicant_hhi','log_lag_density','claim1','n_inventors']
BASE_X_EXP = ['lag_applicant_hhi','log_lag_density']

results_pooled = []

# ── Panel A: Scrutiny（has_reject_reason）────────────────────────
print("\n--- Panel A: Scrutiny (has_reject_reason) ---")

# A1: pharma only（ベースライン再現）
print("  [A1] Pharma only (benchmark) ...")
fit_A1 = run_absorbing_pooled(
    df_pharma_reg,
    'has_reject_reason',
    ['self_cite_ratio'] + BASE_X_SCR)
r = extract_coef(fit_A1, 'self_cite_ratio', 'self_cite_ratio (pharma only)')
r.update({'panel':'A','spec':'Pharma only','outcome':'has_reject_reason'})
results_pooled.append(r)
print(f"    coef={r['coef']:.4f} SE={r['se']:.4f}")

# A2: pooled, self_cite_ratio のみ（pharma ダミーは field FE に吸収されるため除外）
# is_pharma は field_id FE と完全共線 → xvars に含めない
print("  [A2] Pooled, self_cite_ratio only ...")
fit_A2 = run_absorbing_pooled(
    df_reg, 'has_reject_reason',
    ['self_cite_ratio'] + BASE_X_SCR)
r = extract_coef(fit_A2, 'self_cite_ratio', 'self_cite_ratio (pooled, no interact)')
r.update({'panel':'A','spec':'Pooled (no interaction)','outcome':'has_reject_reason'})
results_pooled.append(r)
print(f"    coef={r['coef']:.4f} SE={r['se']:.4f}")

# A3: pooled + self_cite × pharma 交差項（核心仕様）
print("  [A3] Pooled + self_cite × pharma interaction ...")
# is_pharma は field FE に吸収 → 除外。交差項 sc_x_pharma のみ追加。
fit_A3 = run_absorbing_pooled(
    df_reg, 'has_reject_reason',
    ['self_cite_ratio','sc_x_pharma'] + BASE_X_SCR)
r_main  = extract_coef(fit_A3, 'self_cite_ratio',  'self_cite_ratio (main)')
r_inter = extract_coef(fit_A3, 'sc_x_pharma',      'self_cite × pharma (interact)')
r_main.update( {'panel':'A','spec':'Pooled + interaction','outcome':'has_reject_reason'})
r_inter.update({'panel':'A','spec':'Pooled + interaction','outcome':'has_reject_reason'})
results_pooled.extend([r_main, r_inter])
sig = ('***' if r_inter['pval'] and r_inter['pval']<0.001 else
       '**'  if r_inter['pval'] and r_inter['pval']<0.01  else
       '*'   if r_inter['pval'] and r_inter['pval']<0.05  else 'ns')
print(f"    main={r_main['coef']:.4f}  interact={r_inter['coef']:.4f} ({sig})")

# ── Panel B: Scrutiny（log_citation_count）───────────────────────
print("\n--- Panel B: Scrutiny (log_citation_count) ---")

print("  [B1] Pharma only (benchmark) ...")
fit_B1 = run_absorbing_pooled(
    df_pharma_reg,
    'log_citation_count',
    ['self_cite_ratio'] + BASE_X_SCR)
r = extract_coef(fit_B1, 'self_cite_ratio', 'self_cite_ratio (pharma only)')
r.update({'panel':'B','spec':'Pharma only','outcome':'log_citation_count'})
results_pooled.append(r)
print(f"    coef={r['coef']:.4f} SE={r['se']:.4f}")

print("  [B2] Pooled, no interact ...")
fit_B2 = run_absorbing_pooled(
    df_reg, 'log_citation_count',
    ['self_cite_ratio'] + BASE_X_SCR)
r = extract_coef(fit_B2, 'self_cite_ratio', 'self_cite_ratio (pooled, no interact)')
r.update({'panel':'B','spec':'Pooled (no interaction)','outcome':'log_citation_count'})
results_pooled.append(r)
print(f"    coef={r['coef']:.4f} SE={r['se']:.4f}")

print("  [B3] Pooled + self_cite × pharma ...")
fit_B3 = run_absorbing_pooled(
    df_reg, 'log_citation_count',
    ['self_cite_ratio','sc_x_pharma'] + BASE_X_SCR)
r_main  = extract_coef(fit_B3, 'self_cite_ratio', 'self_cite_ratio (main)')
r_inter = extract_coef(fit_B3, 'sc_x_pharma',     'self_cite × pharma (interact)')
r_main.update( {'panel':'B','spec':'Pooled + interaction','outcome':'log_citation_count'})
r_inter.update({'panel':'B','spec':'Pooled + interaction','outcome':'log_citation_count'})
results_pooled.extend([r_main, r_inter])
sig = ('***' if r_inter['pval'] and r_inter['pval']<0.001 else
       '**'  if r_inter['pval'] and r_inter['pval']<0.01  else
       '*'   if r_inter['pval'] and r_inter['pval']<0.05  else 'ns')
print(f"    main={r_main['coef']:.4f}  interact={r_inter['coef']:.4f} ({sig})")

# ── Panel C: Expansion（log_applications）────────────────────────
print("\n--- Panel C: Expansion (log_applications) ---")

print("  [C1] Pharma only (benchmark) ...")
fit_C1 = run_absorbing_pooled(
    panel_pharma_reg,
    'log_applications',
    ['log_lag_patent_stock'] + BASE_X_EXP,
    cluster_col='field_id')
r = extract_coef(fit_C1, 'log_lag_patent_stock', 'log_lag_patent_stock (pharma only)')
r.update({'panel':'C','spec':'Pharma only','outcome':'log_applications'})
results_pooled.append(r)
print(f"    coef={r['coef']:.4f} SE={r['se']:.4f}")

print("  [C2] Pooled, no interact ...")
fit_C2 = run_absorbing_pooled(
    df_panel_reg, 'log_applications',
    ['log_lag_patent_stock'] + BASE_X_EXP,
    cluster_col='field_id')
r = extract_coef(fit_C2, 'log_lag_patent_stock', 'log_lag_patent_stock (pooled, no interact)')
r.update({'panel':'C','spec':'Pooled (no interaction)','outcome':'log_applications'})
results_pooled.append(r)
print(f"    coef={r['coef']:.4f} SE={r['se']:.4f}")

print("  [C3] Pooled + patent_stock × pharma ...")
fit_C3 = run_absorbing_pooled(
    df_panel_reg, 'log_applications',
    ['log_lag_patent_stock','ps_x_pharma'] + BASE_X_EXP,
    cluster_col='field_id')
r_main  = extract_coef(fit_C3, 'log_lag_patent_stock', 'log_lag_patent_stock (main)')
r_inter = extract_coef(fit_C3, 'ps_x_pharma',          'patent_stock × pharma (interact)')
r_main.update( {'panel':'C','spec':'Pooled + interaction','outcome':'log_applications'})
r_inter.update({'panel':'C','spec':'Pooled + interaction','outcome':'log_applications'})
results_pooled.extend([r_main, r_inter])
sig = ('***' if r_inter['pval'] and r_inter['pval']<0.001 else
       '**'  if r_inter['pval'] and r_inter['pval']<0.01  else
       '*'   if r_inter['pval'] and r_inter['pval']<0.05  else 'ns')
print(f"    main={r_main['coef']:.4f}  interact={r_inter['coef']:.4f} ({sig})")

# ================================================================
# STEP 3: Table 3 本体の整形・表示（論文掲載フォーマット）
# ================================================================
print("\n" + "="*60)
print("STEP 3: Table 3 — Pooled Evidence and Scope Conditions")
print("="*60)

def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "—"

def stars(p):
    if p is None: return ''
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

# --- Panel A/B: Scrutiny 表（本文掲載用）---
print("""
Table 3.  Pooled Evidence and Scope Conditions

Dependent variable: (A) has_reject_reason  (B) log_citation_count  (C) log_applications
Fixed effects: IPC-subclass, year.  SE clustered by field.
""")

header = f"{'Variable':<42} {'(A) RR':>10} {'(A) RR':>12} {'(B) CI':>10} {'(B) CI':>12} {'(C) Exp':>10} {'(C) Exp':>12}"
subhdr = f"{'':42} {'Pharma':>10} {'Pooled':>12} {'Pharma':>10} {'Pooled':>12} {'Pharma':>10} {'Pooled':>12}"
print("-"*100)
print(header)
print(subhdr)
print("-"*100)

# 行を抽出して表示
def get_coef(panel, spec, outcome, var_keyword):
    rows = [r for r in results_pooled
            if r['panel']==panel and r['spec']==spec
            and r['outcome']==outcome and var_keyword in r['variable']]
    return rows[0] if rows else None

rows_display = [
    ('self_cite_ratio (main/only)',   'self_cite_ratio'),
    ('  self_cite × pharma',          'sc_x_pharma'),
    ('log_lag_patent_stock (main/only)', 'log_lag_patent_stock'),
    ('  patent_stock × pharma',         'ps_x_pharma'),
]

def show_row(label, var_kw):
    # Pharma only: panel A,B,C / spec=Pharma only
    rA_p = get_coef('A','Pharma only','has_reject_reason', var_kw)
    rA_x = get_coef('A','Pooled + interaction','has_reject_reason', var_kw)
    rB_p = get_coef('B','Pharma only','log_citation_count', var_kw)
    rB_x = get_coef('B','Pooled + interaction','log_citation_count', var_kw)
    rC_p = get_coef('C','Pharma only','log_applications', var_kw)
    rC_x = get_coef('C','Pooled + interaction','log_applications', var_kw)

    def _c(r):
        if r is None or r['coef'] is None: return '—'
        return fmt(r['coef']) + stars(r['pval'])
    def _s(r):
        if r is None or r['se'] is None: return ''
        return '(' + fmt(r['se']) + ')'

    # coef行
    line_coef = (f"{label:<42}"
        + f"{_c(rA_p):>10}" + f"{_c(rA_x):>12}"
        + f"{_c(rB_p):>10}" + f"{_c(rB_x):>12}"
        + f"{_c(rC_p):>10}" + f"{_c(rC_x):>12}")
    print(line_coef)
    # se行
    line_se = (f"{'':42}"
        + f"{_s(rA_p):>10}" + f"{_s(rA_x):>12}"
        + f"{_s(rB_p):>10}" + f"{_s(rB_x):>12}"
        + f"{_s(rC_p):>10}" + f"{_s(rC_x):>12}")
    print(line_se)

for label, var_kw in rows_display:
    show_row(label, var_kw)
    print()

print("-"*100)
print("*** p<0.001  ** p<0.01  * p<0.05")
print("Fixed effects: IPC subclass (field_id) and year, absorbed via AbsorbingLS.")
print("SE clustered at the IPC subclass level.")
print("(A) Pharma: n = pharma applications only.")
print("(A) Pooled: n = full sample including non-pharma.")
print("Interaction = variable × is_pharma (1 = pharmaceutical field).")

# ================================================================
# STEP 4: 結果の解釈ガイド
# ================================================================
print("\n" + "="*60)
print("STEP 4: Interpretation for Section 5.4 / Table 3")
print("="*60)

# 自動解釈
r_sc_inter_A = get_coef('A','Pooled + interaction','has_reject_reason','sc_x_pharma')
r_sc_inter_B = get_coef('B','Pooled + interaction','log_citation_count','sc_x_pharma')
r_ps_inter_C = get_coef('C','Pooled + interaction','log_applications','ps_x_pharma')

print("\n【H5（Scope Conditions）の検証】")
print("\n[Scrutiny channel: self_cite × pharma 交差項]")
for r, label in [(r_sc_inter_A,'has_reject_reason'),
                  (r_sc_inter_B,'log_citation_count')]:
    if r and r['coef'] is not None:
        sig = stars(r['pval'])
        direction = "positive" if r['coef'] > 0 else "negative/zero"
        if r['pval'] and r['pval'] > 0.05:
            interpretation = (
                "→ 交差項非有意: self_cite の scrutiny 効果は pharma に特有でなく、"
                "汎用メカニズム。H5（scope condition）を支持。"
                "\n  論文への記載: 'the pooled interaction term is not significant, "
                "indicating that self-referential accumulation raises scrutiny "
                "broadly across technologies, not exclusively in pharmaceuticals.'")
        else:
            interpretation = (
                f"→ 交差項有意（{sig}）: self_cite の scrutiny 効果は pharma で"
                f"{'さらに強い' if r['coef']>0 else '弱い'}。"
                f"\n  論文への記載: 'the interaction is significant, "
                f"suggesting that the scrutiny channel is {'amplified' if r['coef']>0 else 'attenuated'} "
                f"in pharmaceutical fields.'")
        print(f"  {label}: coef={r['coef']:.4f} (SE={r['se']:.4f}, p={r['pval']:.3f})")
        print(f"  {interpretation}")

print("\n[Expansion channel: patent_stock × pharma 交差項]")
if r_ps_inter_C and r_ps_inter_C['coef'] is not None:
    r = r_ps_inter_C
    sig = stars(r['pval'])
    if r['pval'] and r['pval'] > 0.05:
        interpretation = (
            "→ 交差項非有意: path dependence は pharma に特有でなく汎用。"
            "\n  論文への記載: 'path dependence is not significantly stronger "
            "in pharmaceuticals, indicating that portfolio reinforcement is "
            "a general feature of cumulative innovation.'")
    else:
        interpretation = (
            f"→ 交差項有意（{sig}）: path dependence は pharma で"
            f"{'特に強い' if r['coef']>0 else '弱い'}。")
    print(f"  log_applications: coef={r['coef']:.4f} (SE={r['se']:.4f}, p={r['pval']:.3f})")
    print(f"  {interpretation}")

print("""
【Section 5.4 向け本文テンプレート（結果確認後に数値を入れてください）】

  Table 3 reports the pooled regression results. The coefficient on
  self_cite_ratio in the pooled specification is [COEF_POOLED] ([SE]),
  closely matching the pharmaceutical estimate of [COEF_PHARMA].
  The interaction term self_cite_ratio × pharma is [COEF_INTERACT]
  ([SE_INTERACT]) and [is/is not] statistically significant (p = [P_VALUE]).
  This [supports/qualifies] the scope interpretation in H5: the scrutiny
  channel [operates broadly / is amplified] in pharmaceutical technologies
  [relative to other fields].

  For portfolio expansion (Panel C), the path dependence coefficient
  in the pooled specification is [COEF_PS_POOLED], and the interaction
  patent_stock × pharma is [COEF_PS_INTERACT] ([SE]) with p = [P],
  indicating that portfolio reinforcement is [general / sector-specific].
""")

# ================================================================
# STEP 5: CSV 保存
# ================================================================
print("\n" + "="*60)
print("STEP 5: Saving results")
print("="*60)

import os
os.makedirs(SAVE_DIR, exist_ok=True)

df_results = pd.DataFrame(results_pooled)
p = f'{SAVE_DIR}/table3_pooled_scope.csv'
df_results.to_csv(p, index=False, encoding='utf-8-sig')
kb = os.path.getsize(p) / 1024
print(f"  ✓ table3_pooled_scope.csv  ({kb:.1f} KB)")

# df_all も保存（再実行を避けるため）
df_all.to_pickle(f'{SAVE_DIR}/df_all_fields.pkl')
df_panel_all.to_pickle(f'{SAVE_DIR}/df_panel_all_fields.pkl')
print(f"  ✓ df_all_fields.pkl")
print(f"  ✓ df_panel_all_fields.pkl")

try:
    from google.colab import files as _colab_files
    _colab_files.download(p)
    print(f"  ↓ Downloaded: {os.path.basename(p)}")
except Exception:
    print("  (download skipped)")

print("\n✓ Table 3 pooled regression complete.")
print("  → 数値を Section 5.4 の [ ] 内に入れて本文テンプレートを完成させてください。")
