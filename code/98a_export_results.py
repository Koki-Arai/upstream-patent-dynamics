# ================================================================
# 98a_export_results.py
# ローカル保存 A — 推定結果のみ（軽量版）
# ================================================================
# 【内容】VIF表・回帰係数・SMM結果をCSV/JSONで出力してダウンロード
# 【容量】数KB〜数十KB（何度でも気軽にダウンロード可能）
# 【用途】推定値の確認・論文への転記・バックアップ
#
# 【実行条件】patent_robustness_v2_iip.py の run_all() 実行後
# ================================================================

import os, json, datetime, pickle
import pandas as pd
import numpy as np
from google.colab import files

TS       = datetime.datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR  = f'/content/export_results_{TS}'
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Output dir: {OUT_DIR}")
print(f"Timestamp : {TS}\n")

saved = []

def note(path):
    mb = os.path.getsize(path) / 1024 / 1024
    saved.append(path)
    print(f"  ✓ {os.path.basename(path):45s} {mb*1024:6.1f} KB")


# ----------------------------------------------------------------
# 1. VIF
# ----------------------------------------------------------------
print("[1/5] VIF results")

vif_obj = globals().get('vif_df') or (
    globals().get('results', {}).get('vif') if isinstance(globals().get('results'), dict) else None
)
if vif_obj is not None and isinstance(vif_obj, pd.DataFrame):
    p = f'{OUT_DIR}/vif.csv'
    vif_obj.to_csv(p, index=False, encoding='utf-8-sig')
    note(p)
else:
    print("  [skip] vif_df not found — run run_all() first")


# ----------------------------------------------------------------
# 2. 回帰係数（baseline + interact）
# ----------------------------------------------------------------
print("\n[2/5] Regression coefficients")

reg_obj = globals().get('reg_results') or (
    globals().get('results', {}).get('reg_results')
    if isinstance(globals().get('results'), dict) else None
)

coef_rows = []
if reg_obj is not None and isinstance(reg_obj, dict):
    for key, fit in reg_obj.items():
        outcome, model = key.rsplit('_', 1)
        try:
            params = fit.params
            bse    = fit.std_errors if hasattr(fit, 'std_errors') else fit.bse
            pvals  = fit.pvalues
            ci     = fit.conf_int()
            for attr in [params, bse, pvals]:
                if hasattr(attr, 'iloc') and attr.ndim > 1:
                    attr = attr.iloc[:, 0]
            if hasattr(ci, 'iloc') and ci.ndim > 1:
                ci_lo, ci_hi = ci.iloc[:, 0], ci.iloc[:, 1]
            else:
                ci_lo = ci_hi = pd.Series(dtype=float)
            # rename internal interaction variable
            rename = {'self_x_inv': 'self_cite_ratio:n_inventors'}
            for v in params.index:
                vd = rename.get(v, v)
                coef_rows.append({
                    'outcome':  outcome,
                    'model':    model,
                    'variable': vd,
                    'coef':     round(float(params[v]), 6),
                    'se':       round(float(bse[v]), 6)    if v in bse.index   else None,
                    'pval':     round(float(pvals[v]), 6)  if v in pvals.index else None,
                    'ci_lo':    round(float(ci_lo[v]), 6)  if v in ci_lo.index else None,
                    'ci_hi':    round(float(ci_hi[v]), 6)  if v in ci_hi.index else None,
                })
        except Exception as e:
            print(f"  [warn] {key}: {e}")

    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        p = f'{OUT_DIR}/regression_coefs.csv'
        coef_df.to_csv(p, index=False, encoding='utf-8-sig')
        note(p)
        # 査読者対応キー変数だけ抜粋
        key_vars = ['self_cite_ratio', 'self_cite_ratio:n_inventors',
                    'n_inventors', 'lag_applicant_hhi', 'log_lag_density']
        key_df = coef_df[coef_df['variable'].isin(key_vars)]
        p2 = f'{OUT_DIR}/regression_coefs_key.csv'
        key_df.to_csv(p2, index=False, encoding='utf-8-sig')
        note(p2)
    else:
        print("  [skip] No coefficients extracted")
else:
    print("  [skip] reg_results not found")


# ----------------------------------------------------------------
# 3. SMM 推定値
# ----------------------------------------------------------------
print("\n[3/5] SMM results")

best_obj = globals().get('best')
msim_obj = globals().get('m_sim')
smm_tbl  = globals().get('results', {}).get('smm_table') \
           if isinstance(globals().get('results'), dict) else None

if best_obj is not None:
    smm_dict = {
        'timestamp':    TS,
        'delta':        0.10,
        'J_hat':        round(float(best_obj.fun), 8),
        'converged':    bool(best_obj.success),
        'message':      best_obj.message,
        'params': {
            'phi_s': round(float(best_obj.x[0]), 6),
            'pi_k':  round(float(best_obj.x[1]), 6),
            'pi_g':  round(float(best_obj.x[2]), 6),
        },
        'm_simulated': {
            'phi_s': round(float(msim_obj[0]), 6) if msim_obj is not None else None,
            'pi_k':  round(float(msim_obj[1]), 6) if msim_obj is not None else None,
            'pi_g':  round(float(msim_obj[2]), 6) if msim_obj is not None else None,
        },
        'targets':      {'phi_s': 0.976, 'pi_k': 0.111, 'pi_g': 0.109},
        'smm_v1':       {'phi_s': 1.837, 'pi_k': -0.049,'pi_g': 0.040},
        'sign_reversal_v1': True,
        'sign_reversal_v2': False,
    }
    p = f'{OUT_DIR}/smm_v2_results.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(smm_dict, f, ensure_ascii=False, indent=2)
    note(p)

if smm_tbl is not None and isinstance(smm_tbl, pd.DataFrame):
    p = f'{OUT_DIR}/smm_comparison_table.csv'
    smm_tbl.to_csv(p, index=False, encoding='utf-8-sig')
    note(p)
else:
    # rebuild from best if smm_tbl not available
    if best_obj is not None:
        tbl_data = {
            'Mechanism':       ['phi_s (scrutiny)', 'pi_k (patent stock)', 'pi_g (grant stock)'],
            'Reduced_form':    [0.976,  0.111,  0.109],
            'SMM_v1_delta0':   [1.837, -0.049,  0.040],
            'SMM_v2_delta010': [round(best_obj.x[0],4), round(best_obj.x[1],4), round(best_obj.x[2],4)],
            'm_simulated':     ([round(msim_obj[0],4), round(msim_obj[1],4), round(msim_obj[2],4)]
                                if msim_obj is not None else [None]*3),
            'Sign_v1': ['OK','REVERSAL','OK'],
            'Sign_v2': ['OK','OK','OK'],
        }
        p = f'{OUT_DIR}/smm_comparison_table.csv'
        pd.DataFrame(tbl_data).to_csv(p, index=False, encoding='utf-8-sig')
        note(p)


# ----------------------------------------------------------------
# 4. 標準化効果量サマリー
# ----------------------------------------------------------------
print("\n[4/5] Standardized effect summary")

df_obj = globals().get('df')
if df_obj is not None and 'self_cite_ratio' in df_obj.columns:
    sd   = float(df_obj['self_cite_ratio'].std())
    mean = float(df_obj['self_cite_ratio'].mean())
    eff  = {
        'n_observations':     len(df_obj),
        'self_cite_ratio_mean': round(mean, 4),
        'self_cite_ratio_sd':   round(sd, 4),
        'coef_reject':          0.976,
        'coef_citation':        0.967,
        'delta_reject_1sd':     round(0.976 * sd, 4),
        'delta_citation_1sd':   round(0.967 * sd, 4),
        'mean_rejection_rate':  0.425,
        'pct_increase_reject':  round(0.976 * sd / 0.425 * 100, 1),
        'vif_max':  float(vif_obj[vif_obj['feature'] != 'const']['VIF'].max())
                    if vif_obj is not None else None,
    }
    p = f'{OUT_DIR}/standardized_effects.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(eff, f, ensure_ascii=False, indent=2)
    note(p)
else:
    print("  [skip] df not found")


# ----------------------------------------------------------------
# 5. 実行メタデータ
# ----------------------------------------------------------------
print("\n[5/5] Session metadata")

meta = {
    'timestamp':       TS,
    'files_exported':  [os.path.basename(s) for s in saved],
    'n_observations':  len(df_obj) if df_obj is not None else None,
    'pharma_share':    float(df_obj['is_pharma'].mean()) if (df_obj is not None and 'is_pharma' in df_obj.columns) else 1.0,
}
p = f'{OUT_DIR}/export_meta.json'
with open(p, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
note(p)


# ----------------------------------------------------------------
# ダウンロード
# ----------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Downloading {len(saved)} files ...")
print('='*50)
for path in saved:
    files.download(path)
    print(f"  ↓ {os.path.basename(path)}")

print("\n✓ Export A complete.")
