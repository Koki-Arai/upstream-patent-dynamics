"""
macro_controls_robustness.py
============================
マクロ指標をコントロール変数として加えた頑健性確認
=====================================================

【分析設計の方針】

(1) year 固定効果との関係
    現行推計は field_id + year_id の固定効果を含む。
    year FE は「全企業・全分野に共通する年次変動」をすべて吸収するため、
    TOPIX・CPI 等のマクロ変数（年ダミーと完全共線）を year FE と
    同時に入れることはできない。

    → 対応: year FE を「外して」マクロ変数で代替する仕様と、
             year FE を維持しつつ「self_cite × マクロ変数」の
             交差項を入れる仕様の両方を推計する。

(2) 3種類の仕様
    Spec A: year FE を外してマクロ変数に置換
            → マクロ変動の「水準効果」を直接推定
    Spec B: year FE 維持 + self_cite_ratio × マクロ変数 の交差項
            → マクロ環境が scrutiny channel を変調するかを検証
            → year FE と共線にならない（交差項は within-field variation）
    Spec C: Spec B の expansion 方程式版
            → マクロ環境が path dependence の強さを変調するかを検証

(3) マクロ変数の選択と変換
    - log_topix      : 資本市場の期待収益（水準）
    - topix_ret      : 年次リターン（変化率 = 景気サイクルの代理）
    - cpi_inflation  : CPI 前年比（実質 R&D コストの代理）
    - usdjpy_mean    : 円安進行（輸出競争力・海外売上への影響）
    - lending_rate   : 貸出金利（資金調達コスト）
    - cgpi_yoy       : 企業物価前年比（川上コスト圧力）

    すべて year-level 変数として df に year_id でマージする。

【実行条件】
    df および df_panel が定義済みであること。
    マクロ CSV は '/content/マクロ指標.csv' に配置されていること
    （または MACRO_CSV_PATH を変更）。

【所要時間】約 5〜10 分
"""

import os, gc, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

MACRO_CSV_PATH = '/content/マクロ指標.csv'
SAVE_DIR       = '/content/drive/MyDrive/patent_analysis_cache'

# ================================================================
# STEP 1: マクロデータの読み込みと年次集計
# ================================================================
print("="*60)
print("STEP 1: Loading and aggregating macro indicators")
print("="*60)

df_raw = pd.read_csv(MACRO_CSV_PATH, header=None,
                     encoding='utf-8-sig', dtype=str)

data_rows = df_raw.iloc[2:].copy()
data_rows.columns = range(df_raw.shape[1])

def parse_date(s):
    if pd.isna(s) or str(s).strip() == '': return None, None
    try:
        s = str(s).strip()
        yr = int(s[:4])
        mo_str = s[5:].replace('年','').replace('月','').strip()
        mo = int(mo_str)
        return yr, mo
    except:
        return None, None

data_rows['year'], data_rows['month'] = zip(
    *data_rows[0].apply(parse_date))
data_rows = data_rows.dropna(subset=['year'])
data_rows['year']  = data_rows['year'].astype(int)
data_rows['month'] = data_rows['month'].astype(int)

def clean_num(s):
    if pd.isna(s): return np.nan
    return pd.to_numeric(str(s).replace(',','').strip(), errors='coerce')

data_rows['topix_close']  = data_rows[8].apply(clean_num)
data_rows['cpi']          = data_rows[9].apply(clean_num)
data_rows['usdjpy']       = data_rows[10].apply(clean_num)
data_rows['lending_rate'] = data_rows[11].apply(clean_num)
data_rows['cgpi_yoy']     = data_rows[12].apply(clean_num)

# IIP データとのオーバーラップ期間（1988-2023）に限定
valid = data_rows[
    (data_rows['year'] >= 1988) & (data_rows['year'] <= 2023)
].copy()
print(f"  Monthly rows in 1988-2023: {len(valid)}")

# 年次集計（年平均を基本とし、景気サイクルは変化率で捉える）
annual = valid.groupby('year').agg(
    topix_mean     = ('topix_close',  'mean'),
    cpi_mean       = ('cpi',          'mean'),
    usdjpy_mean    = ('usdjpy',       'mean'),
    lending_rate   = ('lending_rate', 'mean'),
    cgpi_yoy       = ('cgpi_yoy',     'mean'),
    n_months       = ('topix_close',  'count'),
).reset_index()

# 派生変数
annual['log_topix']     = np.log(annual['topix_mean'])
annual['topix_ret']     = annual['topix_mean'].pct_change()    # 年次リターン
annual['cpi_inflation'] = annual['cpi_mean'].pct_change() * 100

# 標準化（回帰係数の比較可能性のため）
for col in ['log_topix','topix_ret','cpi_inflation',
            'usdjpy_mean','lending_rate','cgpi_yoy']:
    mu, sd = annual[col].mean(), annual[col].std()
    if sd > 0:
        annual[f'{col}_std'] = (annual[col] - mu) / sd
    else:
        annual[f'{col}_std'] = 0.0

annual = annual.rename(columns={'year': 'year_id'})
print(f"  Annual macro data: {len(annual)} years "
      f"({annual['year_id'].min()}–{annual['year_id'].max()})")
print(f"  Missing TOPIX: {annual['topix_mean'].isna().sum()} years "
      f"({annual[annual['topix_mean'].isna()]['year_id'].tolist()})")

print("\n  Summary of annual macro variables:")
disp_cols = ['year_id','log_topix','topix_ret','cpi_inflation',
             'usdjpy_mean','lending_rate','cgpi_yoy','n_months']
print(annual[disp_cols].to_string(index=False))

# ================================================================
# STEP 2: macro を df / df_panel にマージ
# ================================================================
print("\n" + "="*60)
print("STEP 2: Merging macro variables into df and df_panel")
print("="*60)

df_obj    = globals().get('df')
panel_obj = globals().get('df_panel')

if df_obj is None or panel_obj is None:
    raise RuntimeError(
        "df または df_panel が見つかりません。\n"
        "00_colab_setup.py を実行後に再実行してください。"
    )

macro_cols = ['year_id',
              'log_topix','log_topix_std',
              'topix_ret','topix_ret_std',
              'cpi_inflation','cpi_inflation_std',
              'usdjpy_mean','usdjpy_mean_std',
              'lending_rate','lending_rate_std',
              'cgpi_yoy','cgpi_yoy_std']

df_m     = df_obj.merge(annual[macro_cols], on='year_id', how='left')
panel_m  = panel_obj.merge(annual[macro_cols], on='year_id', how='left')

match_df    = df_m['log_topix'].notna().mean()
match_panel = panel_m['log_topix'].notna().mean()
print(f"  df      macro match rate: {match_df:.1%}  "
      f"({df_m['log_topix'].notna().sum():,}/{len(df_m):,})")
print(f"  df_panel macro match rate: {match_panel:.1%}  "
      f"({panel_m['log_topix'].notna().sum():,}/{len(panel_m):,})")

# 交差項の事前構築（Spec B / C 用）
interact_vars_scr = ['log_topix','topix_ret','cpi_inflation',
                     'usdjpy_mean','lending_rate']
for v in interact_vars_scr:
    std_col = f'{v}_std'
    df_m[f'sc_x_{v}'] = (
        df_m['self_cite_ratio'] * df_m[std_col].fillna(0))

interact_vars_exp = ['log_topix','topix_ret','cpi_inflation']
for v in interact_vars_exp:
    std_col = f'{v}_std'
    panel_m[f'ps_x_{v}'] = (
        panel_m['log_lag_patent_stock'] * panel_m[std_col].fillna(0))

gc.collect()

# ================================================================
# STEP 3: 回帰推計
# ================================================================
print("\n" + "="*60)
print("STEP 3: Regressions")
print("="*60)

from linearmodels.iv.absorbing import AbsorbingLS
import statsmodels.formula.api as smf

def run_absorbing_fe(df_est, outcome, xvars,
                     cluster_col='field_id',
                     absorb_cols=None):
    """
    linearmodels.AbsorbingLS で固定効果を吸収。
    absorb_cols: 固定効果として吸収するカラムのリスト
                 デフォルト = [cluster_col, 'year_id']
    """
    if absorb_cols is None:
        absorb_cols = [cluster_col, 'year_id']
    reg_vars = [outcome] + xvars + absorb_cols
    d = df_est[list(set(reg_vars))].dropna().copy()
    for c in absorb_cols:
        d[c] = d[c].astype('category')
    mod = AbsorbingLS(
        dependent = d[outcome],
        exog      = d[xvars],
        absorb    = d[absorb_cols],
    )
    cluster_codes = d[cluster_col].cat.codes.values
    return mod.fit(cov_type='clustered', clusters=cluster_codes), d


def run_ols_cluster(df_est, outcome, xvars,
                    fe_cols=None, cluster_col='field_id'):
    """
    smf.ols で FE を C() として展開（Spec A 用: year FE なし）。
    year FE を外してマクロ変数に置換する場合に使用。
    """
    fe_str = ' + '.join([f'C({c})' for c in (fe_cols or [])])
    x_str  = ' + '.join(xvars)
    fml    = f'{outcome} ~ {x_str}' + (f' + {fe_str}' if fe_str else '')
    reg_vars = [outcome] + xvars + (fe_cols or [])
    d = df_est[list(set(reg_vars))].dropna().copy()
    m = smf.ols(fml, data=d).fit(
        cov_type='cluster',
        cov_kwds={'groups': d[cluster_col].values}
    )
    return m, d


def extract_key(fit, varname, rename=None, is_absorbing=True):
    vn = rename or varname
    try:
        params = fit.params
        bse    = fit.std_errors if is_absorbing else fit.bse
        pvals  = fit.pvalues
        for a in [params, bse, pvals]:
            if hasattr(a,'iloc') and a.ndim > 1: a = a.iloc[:,0]
        return {
            'variable': vn,
            'coef': round(float(params[varname]),6),
            'se':   round(float(bse[varname]),   6),
            'pval': round(float(pvals[varname]),  6),
        }
    except Exception as e:
        return {'variable': vn, 'coef': None, 'se': None,
                'pval': None, 'note': str(e)}


BASE_X_SCR = ['lag_applicant_hhi','log_lag_density','claim1','n_inventors']
BASE_X_EXP = ['lag_applicant_hhi','log_lag_density']

results_all = {}

# ── Spec A: year FE を外してマクロ変数で置換 ─────────────────
print("\n--- Spec A: year FE replaced by macro variables (scrutiny) ---")
specA_rows = []

# ベースライン再現（field FE + year FE）
fit, _ = run_absorbing_fe(df_m, 'has_reject_reason',
                          ['self_cite_ratio'] + BASE_X_SCR,
                          absorb_cols=['field_id','year_id'])
row = extract_key(fit, 'self_cite_ratio')
row['spec'] = 'Baseline (field FE + year FE)'
specA_rows.append(row)
print(f"  Baseline: self_cite={row['coef']:.4f} (SE={row['se']:.4f})")

# field FE のみ + マクロ変数
macro_set_A = ['log_topix_std','topix_ret_std','cpi_inflation_std',
               'usdjpy_mean_std','lending_rate_std']
xvars_A = ['self_cite_ratio'] + BASE_X_SCR + macro_set_A

fit, d = run_absorbing_fe(df_m, 'has_reject_reason', xvars_A,
                          absorb_cols=['field_id'])  # year FE 除外
row = extract_key(fit, 'self_cite_ratio')
row['spec'] = 'field FE only + macro controls'
specA_rows.append(row)
print(f"  Spec A  : self_cite={row['coef']:.4f} (SE={row['se']:.4f})")

# マクロ変数の係数も取得
for mv in macro_set_A:
    r = extract_key(fit, mv)
    r['spec'] = 'field FE only + macro controls'
    specA_rows.append(r)
    sig = '***' if r['pval'] and r['pval']<0.001 else \
          '**'  if r['pval'] and r['pval']<0.01  else \
          '*'   if r['pval'] and r['pval']<0.05  else ''
    if r['coef'] is not None:
        print(f"          {mv:30s}: {r['coef']:>8.4f}{sig} (SE={r['se']:.4f})")

results_all['specA_scrutiny'] = specA_rows


# ── Spec B: マクロ局面別サブサンプル比較（scrutiny）─────────
# 【設計変更の理由】
#   year FE と純粋な年次マクロ変数（TOPIX等）の交差項
#   sc_x_v = self_cite × v_year は、year FE による within-year
#   変換後に self_cite の定数倍となり完全共線になる。
#   (demeaned sc_x_v = v_year × demeaned sc ∝ demeaned sc)
#   → 交差項アプローチは原理的に識別不可。
#
#   代替: マクロ局面（好況/不況）でサブサンプルを分割し、
#   self_cite_ratio の係数を比較する。
#   係数が安定 → scrutiny channel はマクロ環境に依存しない普遍的メカニズム。
#   係数が大きく異なる → マクロ局面による変調が存在。

import statsmodels.formula.api as smf

print("\n--- Spec B: Split-sample by macro regime (scrutiny) ---")
specB_rows = []

# 実際の TOPIX リターンデータから boom/bust を分類
annual_ret = annual[['year_id','topix_ret']].dropna()
boom_years  = set(annual_ret[annual_ret['topix_ret'] > 0]['year_id'].tolist())
bust_years  = set(annual_ret[annual_ret['topix_ret'] <= 0]['year_id'].tolist())
hi_cpi_yrs  = set(annual_ret.merge(annual[['year_id','cpi_inflation']],on='year_id',how='left')
                             .query('cpi_inflation > 1.0')['year_id'].tolist())
lo_cpi_yrs  = set(annual_ret.merge(annual[['year_id','cpi_inflation']],on='year_id',how='left')
                             .query('cpi_inflation <= 1.0')['year_id'].tolist())

df_m['is_boom']   = df_m['year_id'].isin(boom_years).astype(int)
df_m['is_hi_cpi'] = df_m['year_id'].isin(hi_cpi_yrs).astype(int)

fml_scr = ('has_reject_reason ~ self_cite_ratio + lag_applicant_hhi '
           '+ log_lag_density + claim1 + n_inventors + C(field_id) + C(year_id)')

regimes = [
    ('Full sample',                    df_m),
    ('Boom years (TOPIX ret > 0)',     df_m[df_m['is_boom']==1]),
    ('Bust years (TOPIX ret ≤ 0)',     df_m[df_m['is_boom']==0]),
    ('High inflation (CPI infl > 1%)', df_m[df_m['is_hi_cpi']==1]),
    ('Low inflation (CPI infl ≤ 1%)',  df_m[df_m['is_hi_cpi']==0]),
]

for label, sub in regimes:
    sub = sub.dropna(subset=['has_reject_reason','self_cite_ratio',
                             'lag_applicant_hhi','log_lag_density',
                             'claim1','n_inventors','field_id','year_id'])
    if len(sub) < 500:
        print(f"  [skip] {label}: n={len(sub):,} (too small)")
        continue
    try:
        m = smf.ols(fml_scr, data=sub).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub['field_id'].values})
        c  = m.params['self_cite_ratio']
        se = m.bse['self_cite_ratio']
        p  = m.pvalues['self_cite_ratio']
        sig = ('***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
        print(f"  {label:45s} n={len(sub):>7,}  "
              f"self_cite={c:.4f}{sig} ({se:.4f}) p={p:.3f}")
        specB_rows.append({
            'spec': label, 'variable': 'self_cite_ratio',
            'coef': round(c,6), 'se': round(se,6), 'pval': round(p,6),
            'n': len(sub),
        })
    except Exception as e:
        print(f"  ERROR {label}: {e}")

results_all['specB_scrutiny'] = specB_rows


# ── Spec C: マクロ局面別サブサンプル比較（expansion）────────
# 同じ理由で交差項は識別不可。サブサンプル比較に変更。

print("\n--- Spec C: Split-sample by macro regime (expansion) ---")
specC_rows = []

panel_m['is_boom']   = panel_m['year_id'].isin(boom_years).astype(int)
panel_m['is_hi_cpi'] = panel_m['year_id'].isin(hi_cpi_yrs).astype(int)

fml_exp = ('log_applications ~ log_lag_patent_stock + log_lag_grant_stock '
           '+ lag_applicant_hhi + log_lag_density + C(field_id) + C(year_id)')

regimes_exp = [
    ('Full sample',                    panel_m),
    ('Boom years (TOPIX ret > 0)',     panel_m[panel_m['is_boom']==1]),
    ('Bust years (TOPIX ret ≤ 0)',     panel_m[panel_m['is_boom']==0]),
    ('High inflation (CPI infl > 1%)', panel_m[panel_m['is_hi_cpi']==1]),
    ('Low inflation (CPI infl ≤ 1%)',  panel_m[panel_m['is_hi_cpi']==0]),
]

for label, sub in regimes_exp:
    sub = sub.dropna(subset=['log_applications','log_lag_patent_stock',
                             'log_lag_grant_stock','lag_applicant_hhi',
                             'log_lag_density','field_id','year_id'])
    if len(sub) < 200:
        print(f"  [skip] {label}: n={len(sub):,}")
        continue
    try:
        m = smf.ols(fml_exp, data=sub).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub['field_id'].values})
        c_ps  = m.params['log_lag_patent_stock']
        se_ps = m.bse['log_lag_patent_stock']
        p_ps  = m.pvalues['log_lag_patent_stock']
        sig = ('***' if p_ps<0.001 else '**' if p_ps<0.01 else '*' if p_ps<0.05 else '')
        print(f"  {label:45s} n={len(sub):>7,}  "
              f"patent_stock={c_ps:.4f}{sig} ({se_ps:.4f}) p={p_ps:.3f}")
        specC_rows.append({
            'spec': label, 'variable': 'log_lag_patent_stock',
            'coef': round(c_ps,6), 'se': round(se_ps,6), 'pval': round(p_ps,6),
            'n': len(sub),
        })
    except Exception as e:
        print(f"  ERROR {label}: {e}")

results_all['specC_expansion'] = specC_rows

# ================================================================
# STEP 4: 結果表示
# ================================================================
print("\n" + "="*60)
print("STEP 4: Results Tables")
print("="*60)

def print_result_table(title, rows, key='coef'):
    print(f"\n[{title}]")
    print("-"*75)
    print(f"  {'Spec':<30} {'Variable':<35} {'Coef':>8} {'SE':>9} {'p':>7}")
    print("-"*75)
    for r in rows:
        if r.get('coef') is None: continue
        sig = ('***' if r['pval']<0.001 else
               '**'  if r['pval']<0.01  else
               '*'   if r['pval']<0.05  else '')
        print(f"  {r.get('spec',''):<30} {r['variable']:<35} "
              f"{r['coef']:>8.4f}{sig:<3} ({r['se']:>6.4f}) {r['pval']:>7.3f}")
    print("-"*75)
    print("  *** p<0.001  ** p<0.01  * p<0.05  |  SE clustered by field.")

print_result_table(
    "Spec A — Scrutiny: year FE replaced by macro variables",
    results_all['specA_scrutiny'])

print_result_table(
    "Spec B — Scrutiny: self_cite × macro interaction (year FE maintained)",
    results_all['specB_scrutiny'])

print_result_table(
    "Spec C — Expansion: patent_stock × macro interaction (year FE maintained)",
    results_all['specC_expansion'])

# ================================================================
# STEP 5: 解釈ガイド
# ================================================================
print("\n" + "="*60)
print("STEP 5: Interpretation guide")
print("="*60)

print("""
【Spec A の解釈（year FE → マクロ変数置換）】

  year FE を外してマクロ変数で置換しても self_cite_ratio の係数が
  安定していれば、scrutiny channel は「年代効果」の代替変数に
  よって駆動されておらず、構造的なものであることを示す。

  各マクロ変数の係数の符号と意味:
    log_topix   (+): 株高年 = 出願件数多く審査強度も高い（好況効果）
    topix_ret   (-): 株価上昇年 = 審査が緩まる（余裕効果）?
    cpi_infl    (+): インフレ年 = コスト圧力で戦略的出願増 → scrutiny↑
    usdjpy      (+): 円安年 = 輸出企業の収益↑ → 出願活性化
    lending_rate(-): 金利高 = R&D 資金調達コスト↑ → 出願減 → scrutiny?

  ※ 符号は理論的に両方向ありうる。結果を見て解釈する。

【Spec B の解釈（self_cite × マクロ変数の交差項）】

  交差項が有意であれば、scrutiny channel の強さがマクロ環境によって
  変調されることを意味する。例えば:

    self_cite × topix_ret が負:
      株価上昇（好況）局面では self_cite の審査強化効果が弱まる。
      → 審査官が忙しく、自己引用による先行技術複雑性を見落とす?
      → あるいは好況期には出願の質が高く拒絶理由が少ない?

    self_cite × cpi_inflation が正:
      インフレ局面では self_cite の審査強化効果が強まる。
      → インフレ期には戦略的（防衛的）出願が増え、
        自己引用密度が高い出願が特に厳しく審査される?

  交差項が非有意であれば:
    「scrutiny channel はマクロ環境に依存しない普遍的メカニズム」
    という解釈を支持する（robustness として記載）。

【Spec C の解釈（patent_stock × マクロ変数の交差項）】

  交差項が有意であれば、path dependence の強さが景気循環によって
  変動することを意味する。例えば:

    patent_stock × topix_ret が正:
      好況年ほど既存ストックから次の出願への正の連鎖が強まる。
      → cumulative advantage が好況期に増幅される。

    patent_stock × cpi_inflation が負:
      インフレ年は R&D コスト上昇でストック効果が弱まる。

【論文への位置づけ】

  これらの推計はいずれも Appendix の robustness check として位置づける。
  主結果（Section 5）への影響が小さければ:
    "Appendix Table D8 shows that the main findings are robust to
     controlling for aggregate macroeconomic conditions and to
     allowing macro-level moderation of the key mechanisms."
  と一段落で処理可能。

  交差項が有意であれば:
    Section 7.3（Limitations and Future Directions）で
    「マクロ変動との連動」を将来課題として論じることができる。
""")

# ================================================================
# STEP 6: CSV 保存
# ================================================================
print("\n" + "="*60)
print("STEP 6: Saving results")
print("="*60)

os.makedirs(SAVE_DIR, exist_ok=True)

for name, rows in results_all.items():
    p = f'{SAVE_DIR}/appendix_{name}.csv'
    pd.DataFrame(rows).to_csv(p, index=False, encoding='utf-8-sig')
    kb = os.path.getsize(p) / 1024
    print(f"  ✓ {os.path.basename(p):55s} {kb:.1f} KB")

# macro annual データも保存
macro_save = annual.copy()
macro_save = macro_save.rename(columns={'year_id':'year'})
p_macro = f'{SAVE_DIR}/macro_annual_1988_2023.csv'
macro_save.to_csv(p_macro, index=False, encoding='utf-8-sig')
print(f"  ✓ {os.path.basename(p_macro):55s}")

try:
    from google.colab import files as _colab_files
    _paths = [f'{SAVE_DIR}/appendix_{name}.csv' for name in results_all]
    for _p in _paths:
        _colab_files.download(_p)
    print("\n  ↓ Results downloaded.")
except Exception:
    print("  (download skipped — not in Colab or files unavailable)")

print("\n✓ Macro controls robustness check complete.")
