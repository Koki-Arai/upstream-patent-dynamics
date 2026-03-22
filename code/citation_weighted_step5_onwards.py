# ================================================================
# citation_weighted_step5_onwards.py
# STEP 5 以降のみ再実行するセル
# ================================================================
# 【使用場面】
#   citation_weighted_robustness.py を実行中に
#   linearmodels の ModuleNotFoundError が出た場合、
#   !pip install linearmodels の後にこのセルだけ実行する。
#   df_wt / df_panel_wt はすでに定義済みのはず。
#
# 【前提】
#   df_wt      : STEP 3 完了後の application-level DataFrame
#   df_panel_wt: STEP 4 完了後の firm-field-year DataFrame
# ================================================================

import os, gc, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

SAVE_DIR    = '/content/drive/MyDrive/patent_analysis_cache'
CITE_WINDOWS = [5, 10]

# ── 変数の存在確認 ────────────────────────────────────────────
df_wt       = globals().get('df_wt')
df_panel_wt = globals().get('df_panel_wt')

if df_wt is None or df_panel_wt is None:
    raise RuntimeError(
        "df_wt または df_panel_wt が見つかりません。\n"
        "citation_weighted_robustness.py を最初から実行してください。"
    )

print(f"df_wt      : {len(df_wt):,} rows × {len(df_wt.columns)} cols")
print(f"df_panel_wt: {len(df_panel_wt):,} rows × {len(df_panel_wt.columns)} cols")

# 必要な列が存在するか確認
needed_wt = [f'wt_self_cite_{T}yr' for T in CITE_WINDOWS] + \
            [f'wt_{T}yr' for T in CITE_WINDOWS]
needed_panel = [f'log_lag_wt_patent_stock_{T}yr' for T in CITE_WINDOWS]

missing = [c for c in needed_wt    if c not in df_wt.columns] + \
          [c for c in needed_panel  if c not in df_panel_wt.columns]
if missing:
    raise RuntimeError(f"以下の列が不足しています: {missing}\n"
                       "STEP 3〜4 から再実行してください。")
print("✓ All required columns present\n")


# ================================================================
# STEP 5: 回帰（再実行）
# ================================================================
print("="*60)
print("STEP 5: Citation-weighted robustness regressions")
print("="*60)

from linearmodels.iv.absorbing import AbsorbingLS

def run_absorbing(df_est, outcome, xvars, cluster_col='field_id'):
    """linearmodels 7.0 対応の AbsorbingLS ラッパー。"""
    reg_vars = [outcome] + xvars + [cluster_col, 'year_id']
    d = df_est[reg_vars].dropna().copy()
    d[cluster_col] = d[cluster_col].astype('category')
    d['year_id']   = d['year_id'].astype('category')
    mod = AbsorbingLS(
        dependent = d[outcome],
        exog      = d[xvars],
        absorb    = d[[cluster_col, 'year_id']],
    )
    cluster_codes = d[cluster_col].cat.codes.values
    return mod.fit(cov_type='clustered', clusters=cluster_codes)


def extract_key(fit, varname, rename=None):
    """fit から指定変数の coef / se / pval を抽出。"""
    vn = rename or varname
    try:
        params = fit.params
        bse    = fit.std_errors
        pvals  = fit.pvalues
        for a in [params, bse, pvals]:
            if hasattr(a, 'iloc') and a.ndim > 1:
                a = a.iloc[:, 0]
        return {
            'variable': vn,
            'coef': round(float(params[varname]), 6),
            'se':   round(float(bse[varname]),    6),
            'pval': round(float(pvals[varname]),   6),
        }
    except Exception as e:
        return {'variable': vn, 'coef': None, 'se': None, 'pval': None,
                'note': str(e)}


BASE_X_SCR = ['lag_applicant_hhi', 'log_lag_density', 'claim1', 'n_inventors']
BASE_X_EXP = ['lag_applicant_hhi', 'log_lag_density']

# ── Scrutiny 方程式 ───────────────────────────────────────────
print("\n--- Scrutiny equation (has_reject_reason) ---")
scrutiny_rows = []

# ① ベースライン（論文 Table 1 再現）
print("  [1/5] Baseline (unweighted) ...")
fit = run_absorbing(df_wt, 'has_reject_reason',
                    ['self_cite_ratio'] + BASE_X_SCR)
row = extract_key(fit, 'self_cite_ratio')
row['model'] = 'Baseline (unweighted)'
scrutiny_rows.append(row)
print(f"        self_cite_ratio: {row['coef']:.4f} (SE={row['se']:.4f}, p={row['pval']:.3f})")

# ② 重み付き自己引用比率に置換
for T in CITE_WINDOWS:
    wt_var = f'wt_self_cite_{T}yr'
    print(f"  [{'2' if T==5 else '3'}/5] Citation-weighted self_cite (T={T}yr) ...")
    fit = run_absorbing(df_wt, 'has_reject_reason',
                        [wt_var] + BASE_X_SCR)
    row = extract_key(fit, wt_var,
                      rename=f'wt_self_cite_ratio (T={T}yr)')
    row['model'] = f'Citation-weighted self_cite (T={T}yr)'
    scrutiny_rows.append(row)
    print(f"        {row['variable']}: {row['coef']:.4f} (SE={row['se']:.4f}, p={row['pval']:.3f})")

# ③ 重みをコントロールとして追加（主変数は維持）
for T in CITE_WINDOWS:
    wt_var = f'wt_{T}yr'
    print(f"  [{'4' if T==5 else '5'}/5] self_cite + citation weight control (T={T}yr) ...")
    fit = run_absorbing(df_wt, 'has_reject_reason',
                        ['self_cite_ratio', wt_var] + BASE_X_SCR)
    row = extract_key(fit, 'self_cite_ratio',
                      rename=f'self_cite_ratio (+wt_control T={T}yr)')
    row['model'] = f'self_cite + citation weight control (T={T}yr)'
    scrutiny_rows.append(row)
    print(f"        self_cite_ratio: {row['coef']:.4f} (SE={row['se']:.4f}, p={row['pval']:.3f})")

# ── Expansion 方程式 ──────────────────────────────────────────
print("\n--- Expansion equation (log_applications) ---")
expansion_rows = []

# ① ベースライン
print("  [1/3] Baseline (unweighted) ...")
fit = run_absorbing(df_panel_wt, 'log_applications',
                    ['log_lag_patent_stock', 'log_lag_grant_stock'] + BASE_X_EXP,
                    cluster_col='field_id')
for var in ['log_lag_patent_stock', 'log_lag_grant_stock']:
    row = extract_key(fit, var)
    row['model'] = 'Baseline (unweighted)'
    expansion_rows.append(row)
    print(f"        {var}: {row['coef']:.4f} (SE={row['se']:.4f}, p={row['pval']:.3f})")

# ② 重み付きストック
for i, T in enumerate(CITE_WINDOWS, 2):
    wt_stock = f'log_lag_wt_patent_stock_{T}yr'
    print(f"  [{i+1}/3] Citation-weighted patent stock (T={T}yr) ...")
    fit = run_absorbing(df_panel_wt, 'log_applications',
                        [wt_stock, 'log_lag_grant_stock'] + BASE_X_EXP,
                        cluster_col='field_id')
    row = extract_key(fit, wt_stock,
                      rename=f'log_lag_wt_patent_stock (T={T}yr)')
    row['model'] = f'Citation-weighted patent stock (T={T}yr)'
    expansion_rows.append(row)
    print(f"        {row['variable']}: {row['coef']:.4f} (SE={row['se']:.4f}, p={row['pval']:.3f})")


# ================================================================
# STEP 6: Appendix Table の表示
# ================================================================
print("\n" + "="*60)
print("STEP 6: Appendix Table D6 & D7")
print("="*60)

def print_table(title, outcome, rows, key_var_label='Key variable'):
    print(f"\n[{title}]")
    print(f"Outcome: {outcome}")
    print("-"*72)
    print(f"  {'Model':<45} {'Coef':>8} {'SE':>10} {'p-val':>8}")
    print("-"*72)
    for r in rows:
        if r['coef'] is None:
            print(f"  {r['model']:<45} {'N/A':>8}")
            continue
        sig = ('***' if r['pval'] < 0.001 else
               '**'  if r['pval'] < 0.01  else
               '*'   if r['pval'] < 0.05  else '')
        print(f"  {r['model']:<45} "
              f"{r['coef']:>8.4f}{sig:<3} "
              f"({r['se']:>7.4f}) "
              f"{r['pval']:>8.3f}")
    print("-"*72)
    print("  *** p<0.001  ** p<0.01  * p<0.05")
    print("  Fixed effects: field_id, year_id (absorbed). SE clustered by field.")

print_table("Table D6: Citation-Weighted Scrutiny Robustness",
            "has_reject_reason", scrutiny_rows)
print_table("Table D7: Citation-Weighted Patent Stock Robustness",
            "log_applications", expansion_rows)


# ================================================================
# STEP 7: CSV 保存
# ================================================================
print("\n" + "="*60)
print("STEP 7: Saving results")
print("="*60)

import os
os.makedirs(SAVE_DIR, exist_ok=True)

scr_df = pd.DataFrame(scrutiny_rows)
exp_df = pd.DataFrame(expansion_rows)

scr_path = f'{SAVE_DIR}/appendix_table_d6_citation_weighted_scrutiny.csv'
exp_path = f'{SAVE_DIR}/appendix_table_d7_citation_weighted_expansion.csv'
scr_df.to_csv(scr_path, index=False, encoding='utf-8-sig')
exp_df.to_csv(exp_path, index=False, encoding='utf-8-sig')

for p in [scr_path, exp_path]:
    kb = os.path.getsize(p) / 1024
    print(f"  ✓ {os.path.basename(p):60s} {kb:.1f} KB")

# df_wt / df_panel_wt の保存
df_wt.to_pickle(f'{SAVE_DIR}/df_application_wt.pkl')
df_panel_wt.to_pickle(f'{SAVE_DIR}/df_panel_wt.pkl')
print(f"  ✓ df_application_wt.pkl")
print(f"  ✓ df_panel_wt.pkl")

# ローカルダウンロード
try:
    from google.colab import files
    files.download(scr_path)
    files.download(exp_path)
    print("\n  ↓ Table D6 and D7 downloaded.")
except Exception:
    pass

# ================================================================
# STEP 8: 解釈ガイド
# ================================================================
print("\n" + "="*60)
print("STEP 8: Interpretation guide")
print("="*60)

# 自動解釈
scr_baseline = next((r for r in scrutiny_rows if r['model']=='Baseline (unweighted)'), None)
scr_wt5      = next((r for r in scrutiny_rows if 'T=5yr' in r['model'] and 'Citation-weighted' in r['model']), None)
exp_baseline = next((r for r in expansion_rows if r['model']=='Baseline (unweighted)'
                     and 'patent_stock' in r['variable']), None)
exp_wt5      = next((r for r in expansion_rows if 'T=5yr' in r['model']), None)

print("\n【Table D6: Scrutiny の解釈】")
if scr_baseline and scr_wt5 and scr_baseline['coef'] and scr_wt5['coef']:
    sign_consistent = (scr_baseline['coef'] > 0) == (scr_wt5['coef'] > 0)
    sig_maintained  = scr_wt5['pval'] < 0.05
    print(f"  Baseline self_cite_ratio: {scr_baseline['coef']:.4f} (p={scr_baseline['pval']:.3f})")
    print(f"  Citation-weighted (T=5yr): {scr_wt5['coef']:.4f} (p={scr_wt5['pval']:.3f})")
    if sign_consistent and sig_maintained:
        print("  → ✓ 符号一致・有意性維持")
        print("    「質の高い先行特許への自己引用ほど審査強度を高める」を支持。")
        print("    論文への記載: robustness として Table D6 を引用し、")
        print("    'The scrutiny effect holds for citation-weighted')と記述可。")
    elif sign_consistent and not sig_maintained:
        print("  → △ 符号一致・有意性低下")
        print("    効果の方向は維持されるが統計的有意性は弱まる。")
        print("    「重み付きでも正の関係が確認されるが精度は低下する」と注記。")
    else:
        print("  → ✗ 符号反転：追加検討が必要。")

print("\n【Table D7: Expansion の解釈】")
if exp_baseline and exp_wt5 and exp_baseline['coef'] and exp_wt5['coef']:
    sign_consistent = (exp_baseline['coef'] > 0) == (exp_wt5['coef'] > 0)
    sig_maintained  = exp_wt5['pval'] < 0.05 if exp_wt5['pval'] else False
    print(f"  Baseline log_lag_patent_stock: {exp_baseline['coef']:.4f} (p={exp_baseline['pval']:.3f})")
    print(f"  Citation-weighted (T=5yr):     {exp_wt5['coef']:.4f} (p={exp_wt5['pval']:.3f})")
    if sign_consistent and sig_maintained:
        print("  → ✓ path dependence は重み付きでも維持。")
        print("    「質の高い特許の蓄積がより強く次の出願を促進する」を示唆。")
    elif sign_consistent:
        print("  → △ 方向は同じだが有意性低下。注記として言及可。")
    else:
        print("  → ✗ 符号反転：weighted stock の構築方法を再検討。")

print("\n【論文への挿入箇所】")
print("  Section 6（Robustness Checks）末尾に新設:")
print("  '6.6 Citation-Weighted Specifications'")
print("  Appendix Table D6（scrutiny）、Table D7（expansion）を参照。")

print("\n✓ Citation-weighted robustness check complete.")
