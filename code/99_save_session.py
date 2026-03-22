# ================================================================
# 99_save_session.py
# Patent Portfolio Dynamics — セッション保存スクリプト
# ================================================================
# 【実行タイミング】
#   - 分析が一段落したとき
#   - ランタイム切れが心配になったとき
#   - いつでも上書き保存として実行可能
#
# 【保存先フォルダ構成】
#   MyDrive/patent_analysis/
#   ├── 01_raw_combined/        IIPファイルを decade 結合した中間データ
#   ├── 02_pharma/              医薬品に絞ったメインデータ
#   ├── 03_results/             回帰・VIF・SMM の推定結果
#   ├── 04_figures/             図（将来の可視化用プレースホルダ）
#   └── 99_session_log/         実行ログ・メモリ使用量・変数一覧
# ================================================================

import os, gc, sys, pickle, json, datetime, traceback
import pandas as pd
import numpy as np

# ── 保存先ルート ──────────────────────────────────────────────
DRIVE_ROOT  = '/content/drive/MyDrive'
SAVE_ROOT   = f'{DRIVE_ROOT}/patent_analysis'

DIRS = {
    'raw':     f'{SAVE_ROOT}/01_raw_combined',
    'pharma':  f'{SAVE_ROOT}/02_pharma',
    'results': f'{SAVE_ROOT}/03_results',
    'figures': f'{SAVE_ROOT}/04_figures',
    'log':     f'{SAVE_ROOT}/99_session_log',
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

print("=== Save directories ===")
for k, d in DIRS.items():
    print(f"  {k:10s} → {d}")


# ── タイムスタンプ ────────────────────────────────────────────
TS = datetime.datetime.now().strftime('%Y%m%d_%H%M')
print(f"\nTimestamp: {TS}")


# ── ヘルパー ─────────────────────────────────────────────────
def save_df(df, folder, filename, formats=('pkl', 'csv')):
    """DataFrame を pickle と CSV で保存し、ファイルサイズを報告する。"""
    results = {}
    for fmt in formats:
        path = f'{folder}/{filename}.{fmt}'
        try:
            if fmt == 'pkl':
                df.to_pickle(path)
            elif fmt == 'csv':
                df.to_csv(path, index=False, encoding='utf-8-sig')
            size_mb = os.path.getsize(path) / 1024 / 1024
            results[fmt] = f'{size_mb:.1f} MB'
            print(f"  ✓ {os.path.basename(path):50s} {size_mb:6.1f} MB")
        except Exception as e:
            results[fmt] = f'ERROR: {e}'
            print(f"  ✗ {filename}.{fmt}: {e}")
    return results


def save_json(obj, path):
    """dict / list を JSON で保存。numpy 型は自動変換。"""
    def convert(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        if isinstance(o, pd.DataFrame):   return o.to_dict(orient='records')
        if isinstance(o, pd.Series):      return o.to_dict()
        return str(o)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ {os.path.basename(path):50s} {size_kb:6.1f} KB")


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ {os.path.basename(path):50s} {size_mb:6.1f} MB")


def check_var(name):
    """変数が定義済みかチェック。"""
    return name in dir(__builtins__) or name in globals()


def get_var(name, default=None):
    """グローバル変数を安全に取得。"""
    return globals().get(name, default)


# ================================================================
# BLOCK A: 01_raw_combined — decade結合ファイル
# ================================================================
print("\n" + "="*55)
print("BLOCK A: 01_raw_combined (decade結合中間データ)")
print("="*55)

# このブロックは 00_colab_setup.py で ap/cc/applicant 等を
# 結合した直後に実行する想定。変数が存在する場合のみ保存する。

raw_targets = {
    # 変数名            保存ファイル名          保存形式
    'ap':              ('ap_combined',          ('pkl',)),
    'appl':            ('applicant_combined',   ('pkl',)),
    'cc':              ('cc_combined',          ('pkl',)),
    'inv':             ('inventor_combined',    ('pkl',)),
    'hr':              ('hr_combined',          ('pkl',)),
}

saved_raw = []
for varname, (filename, fmts) in raw_targets.items():
    obj = get_var(varname)
    if obj is not None and isinstance(obj, pd.DataFrame):
        print(f"\n  Saving {varname} ({len(obj):,} rows) ...")
        save_df(obj, DIRS['raw'], filename, formats=fmts)
        saved_raw.append(varname)
    else:
        print(f"  [skip] {varname} — not found in session")

if not saved_raw:
    print("  (No raw combined DataFrames found — run 00_colab_setup.py first)")


# ================================================================
# BLOCK B: 02_pharma — 医薬品絞り込みデータ
# ================================================================
print("\n" + "="*55)
print("BLOCK B: 02_pharma (医薬品メインデータ)")
print("="*55)

pharma_targets = {
    'df':       ('df_application',  ('pkl', 'csv')),
    'df_panel': ('df_panel',        ('pkl', 'csv')),
}

saved_pharma = []
for varname, (filename, fmts) in pharma_targets.items():
    obj = get_var(varname)
    if obj is not None and isinstance(obj, pd.DataFrame):
        print(f"\n  Saving {varname} ({len(obj):,} rows × {len(obj.columns)} cols) ...")
        save_df(obj, DIRS['pharma'], filename, formats=fmts)
        saved_pharma.append(varname)
    else:
        print(f"  [skip] {varname} — not found in session")

# 中間集計オブジェクトも保存
for varname in ['ci_agg', 'self_cite_agg', 'n_inv', 'hhi_lag', 'log_density']:
    obj = get_var(varname)
    if obj is not None and isinstance(obj, pd.DataFrame):
        print(f"\n  Saving intermediate: {varname} ({len(obj):,} rows) ...")
        save_df(obj, DIRS['pharma'], f'intermediate_{varname}', formats=('pkl',))


# ================================================================
# BLOCK C: 03_results — 推定結果
# ================================================================
print("\n" + "="*55)
print("BLOCK C: 03_results (推定結果)")
print("="*55)

# ── C1: VIF ──────────────────────────────────────────────────
vif_obj = get_var('vif_df')
if vif_obj is not None and isinstance(vif_obj, pd.DataFrame):
    print("\n  VIF results:")
    save_df(vif_obj, DIRS['results'], f'vif_{TS}', formats=('csv', 'pkl'))
else:
    print("  [skip] vif_df — not found")

# ── C2: 回帰結果 ─────────────────────────────────────────────
reg_obj = get_var('reg_results')
if reg_obj is not None and isinstance(reg_obj, dict):
    print("\n  Regression results (linearmodels fit objects):")
    # fit オブジェクト全体を pickle で保存
    save_pickle(reg_obj, f'{DIRS["results"]}/reg_results_{TS}.pkl')
    # キー係数だけ CSV で保存
    coef_rows = []
    for key, fit in reg_obj.items():
        outcome, model = key.rsplit('_', 1)
        try:
            params = fit.params
            bse    = fit.std_errors
            pvals  = fit.pvalues
            if hasattr(params, 'iloc') and params.ndim > 1:
                params = params.iloc[:, 0]
            if hasattr(bse, 'iloc') and bse.ndim > 1:
                bse = bse.iloc[:, 0]
            if hasattr(pvals, 'iloc') and pvals.ndim > 1:
                pvals = pvals.iloc[:, 0]
            for v in params.index:
                coef_rows.append({
                    'outcome': outcome, 'model': model, 'variable': v,
                    'coef': float(params[v]),
                    'se':   float(bse[v])   if v in bse.index   else None,
                    'pval': float(pvals[v]) if v in pvals.index else None,
                })
        except Exception as e:
            print(f"    [warn] {key}: {e}")
    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        save_df(coef_df, DIRS['results'], f'reg_coefs_{TS}', formats=('csv', 'pkl'))
else:
    print("  [skip] reg_results — not found")

# ── C3: SMM結果 ──────────────────────────────────────────────
smm_best = get_var('best')   # smm_v2 の戻り値
smm_msim = get_var('m_sim')

if smm_best is not None:
    print("\n  SMM results:")
    smm_summary = {
        'timestamp':   TS,
        'delta':       0.10,
        'J_hat':       float(smm_best.fun),
        'converged':   bool(smm_best.success),
        'message':     smm_best.message,
        'phi_s':       float(smm_best.x[0]),
        'pi_k':        float(smm_best.x[1]),
        'pi_g':        float(smm_best.x[2]),
        'm_sim_phi_s': float(smm_msim[0]) if smm_msim is not None else None,
        'm_sim_pi_k':  float(smm_msim[1]) if smm_msim is not None else None,
        'm_sim_pi_g':  float(smm_msim[2]) if smm_msim is not None else None,
        'target':      [0.976, 0.111, 0.109],
        'old_smm_v1':  [1.837, -0.049, 0.040],
    }
    save_json(smm_summary, f'{DIRS["results"]}/smm_v2_{TS}.json')
    save_pickle(smm_best, f'{DIRS["results"]}/smm_best_{TS}.pkl')
else:
    print("  [skip] SMM best result — not found")

# ── C4: run_all の出力 results dict ──────────────────────────
results_obj = get_var('results')
if results_obj is not None and isinstance(results_obj, dict):
    print("\n  run_all() results dict:")
    # pickleで保存（fit objectを含む）
    try:
        save_pickle(results_obj, f'{DIRS["results"]}/run_all_results_{TS}.pkl')
    except Exception as e:
        print(f"  [warn] run_all results pickle failed: {e}")

    # smm_table は CSV でも保存
    smm_tbl = results_obj.get('smm_table')
    if smm_tbl is not None and isinstance(smm_tbl, pd.DataFrame):
        save_df(smm_tbl, DIRS['results'], f'smm_table_{TS}', formats=('csv',))
    vif_tbl = results_obj.get('vif')
    if vif_tbl is not None and isinstance(vif_tbl, pd.DataFrame):
        save_df(vif_tbl, DIRS['results'], f'vif_from_results_{TS}', formats=('csv',))
else:
    print("  [skip] run_all() results — not found")


# ================================================================
# BLOCK D: 99_session_log — メモリ・変数一覧・ログ
# ================================================================
print("\n" + "="*55)
print("BLOCK D: 99_session_log (セッション情報)")
print("="*55)

# ── D1: メモリ使用量 ─────────────────────────────────────────
try:
    import psutil
    mem = psutil.virtual_memory()
    mem_info = {
        'timestamp':      TS,
        'total_GB':       round(mem.total / 1024**3, 2),
        'used_GB':        round(mem.used  / 1024**3, 2),
        'available_GB':   round(mem.available / 1024**3, 2),
        'percent_used':   mem.percent,
    }
except ImportError:
    mem_info = {'timestamp': TS, 'note': 'psutil not available'}

# DataFrame のメモリ使用量
df_memory = {}
for varname in ['ap', 'appl', 'cc', 'inv', 'hr', 'df', 'df_panel',
                'ci_agg', 'self_cite_agg', 'n_inv']:
    obj = get_var(varname)
    if obj is not None and isinstance(obj, pd.DataFrame):
        mb = obj.memory_usage(deep=True).sum() / 1024**2
        df_memory[varname] = {
            'rows': len(obj), 'cols': len(obj.columns),
            'memory_MB': round(mb, 1)
        }

session_log = {
    'timestamp':  TS,
    'python':     sys.version,
    'ram':        mem_info,
    'dataframes': df_memory,
    'saved_raw':    saved_raw,
    'saved_pharma': saved_pharma,
}
save_json(session_log, f'{DIRS["log"]}/session_{TS}.json')

# ── D2: 変数一覧（DataFrame のみ） ───────────────────────────
var_list = []
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame):
        mb = obj.memory_usage(deep=True).sum() / 1024**2
        var_list.append({
            'variable': name,
            'rows': len(obj),
            'cols': len(obj.columns),
            'columns': list(obj.columns),
            'memory_MB': round(mb, 1),
            'dtypes': {c: str(t) for c, t in obj.dtypes.items()},
        })

save_json(var_list, f'{DIRS["log"]}/dataframe_inventory_{TS}.json')

# ── D3: テキストサマリー ──────────────────────────────────────
summary_lines = [
    f"Patent Portfolio Dynamics — Session Save Summary",
    f"Saved at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"",
    f"=== RAM ===",
]
for k, v in mem_info.items():
    summary_lines.append(f"  {k}: {v}")

summary_lines += ["", "=== DataFrames in session ==="]
for row in var_list:
    summary_lines.append(
        f"  {row['variable']:20s}  {row['rows']:>10,} rows  "
        f"{row['cols']:3d} cols  {row['memory_MB']:7.1f} MB"
    )

summary_lines += ["", "=== Saved files ==="]
for root, _, files in os.walk(SAVE_ROOT):
    for fn in sorted(files):
        full = os.path.join(root, fn)
        rel  = full.replace(SAVE_ROOT + '/', '')
        mb   = os.path.getsize(full) / 1024**2
        summary_lines.append(f"  {rel:60s} {mb:7.1f} MB")

summary_text = '\n'.join(summary_lines)
log_path = f'{DIRS["log"]}/summary_{TS}.txt'
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)
size_kb = os.path.getsize(log_path) / 1024
print(f"  ✓ {os.path.basename(log_path):50s} {size_kb:6.1f} KB")

print(summary_text)


# ================================================================
# 完了メッセージ
# ================================================================
print("\n" + "="*55)
print("✓ Save complete.")
print(f"  Root: {SAVE_ROOT}")
print(f"  Timestamp: {TS}")
print("="*55)
print("\n【次回ランタイム再接続後の復帰手順】")
print("  1. Google Drive をマウント")
print("  2. パッケージインストール (00_colab_setup.py CELL 2-3)")
print("  3. 以下を実行して df / df_panel を復元:")
print(f"     import pandas as pd")
print(f"     df       = pd.read_pickle('{DIRS['pharma']}/df_application.pkl')")
print(f"     df_panel = pd.read_pickle('{DIRS['pharma']}/df_panel.pkl')")
print(f"  4. patent_robustness_v2_iip.py を実行")
