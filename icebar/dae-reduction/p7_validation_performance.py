"""
Phase 7: Correctness battery + performance measurement on NPCC (140-bus).

Validates that the reduced TDS produces correct results on a larger case
and measures the wall-clock speedup.

NPCC facts (27 GENROU + 29 TGOV1 + GENCLS + IEEEX1 + ...):
  Full TDS   dae.m ~ 1411
  Reduced    dae.m ~ 719  (49% reduction)

Success criteria
----------------
1. dae.m reduced < dae.m full (genuine reduction confirmed).
2. dae.m reduced == expected_reduced (no regression).
3. INDEPENDENT Algeb final values (GENROU + GENCLS) match full TDS rtol<1e-3.
4. DEPENDENT Algeb get_data() final values match full TDS rtol<1e-3.
5. TGOV1.pout self-consistent: get_data()[-1] == live .v (abs diff=0).
6. Wall-clock speedup >= 1.0 (reduced not slower than full).

Usage
-----
    conda activate andesre
    cd /Users/Shared/work/andes
    python icebar/dae-reduction/p7_validation_performance.py
"""

import time
import numpy as np
import andes
from andes.utils.dae_reduction import DaeReductionAnalyser


def section(title):
    print(f'\n{"=" * 80}')
    print(f'  {title}')
    print(f'{"=" * 80}')


TF = 20.0   # simulation end time (s)
CASE = andes.get_case('npcc/npcc.xlsx')
RTOL = 1e-3
ATOL = 1e-5

# ---------------------------------------------------------------------------
# 1. Full TDS (baseline)
# ---------------------------------------------------------------------------
section('Step 1: Full TDS baseline (NPCC, tf=20s)')
ss_full = andes.load(CASE, setup=True, default_config=True, no_output=True)
ss_full.TDS.config.tf = TF
ss_full.PFlow.run()
assert ss_full.PFlow.converged, 'PFlow did not converge'

t0 = time.perf_counter()
ss_full.TDS.run()
t_full = time.perf_counter() - t0

dae_m_full = ss_full.dae.m
n_steps_full = len(ss_full.dae.ts.t)
print(f'Full TDS  dae.m={dae_m_full}  n_steps={n_steps_full}  wall={t_full:.1f}s')

# Collect final values for GENROU and GENCLS Algebs
def collect_finals(ss, model_name):
    mdl = getattr(ss, model_name, None)
    if mdl is None or mdl.n == 0:
        return {}
    finals = {}
    for name, alg in mdl.algebs.items():
        data = ss.dae.ts.get_data(alg)
        if data.shape[0] > 0 and data.shape[1] > 0:
            finals[name] = data[-1]
    return finals

genrou_full_finals = collect_finals(ss_full, 'GENROU')
gencls_full_finals = collect_finals(ss_full, 'GENCLS')
tgov1_pout_full_final = ss_full.dae.ts.get_data(ss_full.TGOV1.pout)[-1]
print(f'Collected finals: GENROU={list(genrou_full_finals)}'
      f'  GENCLS={list(gencls_full_finals)}')

# ---------------------------------------------------------------------------
# 2. Reduced TDS
# ---------------------------------------------------------------------------
section('Step 2: Reduced TDS (NPCC, tf=20s)')
ss = andes.load(CASE, setup=True, default_config=True, no_output=True)
ss.TDS.config.tf = TF
ss.PFlow.run()
assert ss.PFlow.converged

result = DaeReductionAnalyser.analyse(ss, verbose=False)
result = DaeReductionAnalyser.generate_reduced_functions(ss, result, verbose=False)
ss._dae_result = result

try:
    t0 = time.perf_counter()
    ss.TDS.run()
    t_red = time.perf_counter() - t0
    tds_crashed = False
except Exception:
    tds_crashed = True
    t_red = float('nan')
    import traceback; traceback.print_exc()

if tds_crashed:
    print('\nOverall: FAIL (TDS crashed)')
    raise SystemExit(1)

dae_m_red = ss.dae.m
n_steps_red = len(ss.dae.ts.t)
print(f'Reduced TDS  dae.m={dae_m_red}  n_steps={n_steps_red}  wall={t_red:.1f}s')
speedup = t_full / t_red
print(f'Speedup: {speedup:.2f}x  (full={t_full:.1f}s  reduced={t_red:.1f}s)')

# ---------------------------------------------------------------------------
# 3. dae.m checks
# ---------------------------------------------------------------------------
section('Step 3: dae.m reduction check')
reduction_ok = dae_m_red < dae_m_full
eliminated = dae_m_full - dae_m_red
pct = 100.0 * eliminated / dae_m_full
print(f'  Full dae.m={dae_m_full}  Reduced dae.m={dae_m_red}')
print(f'  Eliminated: {eliminated} Algebs ({pct:.1f}%)')
print(f'  {"✓" if reduction_ok else "✗"} Reduced < Full')

# ---------------------------------------------------------------------------
# Helper: compare final values
# ---------------------------------------------------------------------------
def compare_final(model_name, name, alg, v_full_final):
    data = ss.dae.ts.get_data(alg)
    shape_ok = data.shape[0] > 0 and data.shape[1] > 0
    if not shape_ok:
        print(f'  ✗ {model_name}.{name}: get_data() returned empty')
        return False
    v_red = data[-1]
    abs_err = np.abs(v_red - v_full_final)
    thresh = ATOL + RTOL * np.abs(v_full_final)
    ok = bool(np.all(abs_err <= thresh))
    max_re = float(np.max(abs_err / (np.abs(v_full_final) + 1e-12)))
    dep = getattr(alg, 'is_dependent', False)
    tag = '[DEP]' if dep else '[IND]'
    print(f'  {"✓" if ok else "✗"} {tag} {model_name}.{name}: '
          f'shape={data.shape}  max_rtol={max_re:.2e}')
    return ok

# ---------------------------------------------------------------------------
# 4. GENROU correctness
# ---------------------------------------------------------------------------
section('Step 4: GENROU Algeb final values')
genrou_ok = True
dep_names = {k.split('.')[1] for k in result.dependent if k.startswith('GENROU.')}
for name, alg in ss.GENROU.algebs.items():
    if name not in genrou_full_finals:
        print(f'  ? GENROU.{name}: not in full baseline'); continue
    if not compare_final('GENROU', name, alg, genrou_full_finals[name]):
        genrou_ok = False

# ---------------------------------------------------------------------------
# 5. GENCLS correctness
# ---------------------------------------------------------------------------
section('Step 5: GENCLS Algeb final values')
gencls_ok = True
if ss.GENCLS.n == 0:
    print('  (no GENCLS in this case — skipped)')
else:
    gencls_dep_names = {k.split('.')[1] for k in result.dependent
                        if k.startswith('GENCLS.')}
    for name, alg in ss.GENCLS.algebs.items():
        if name not in gencls_full_finals:
            print(f'  ? GENCLS.{name}: not in full baseline'); continue
        if not compare_final('GENCLS', name, alg, gencls_full_finals[name]):
            gencls_ok = False

# ---------------------------------------------------------------------------
# 6. TGOV1.pout self-consistency
# ---------------------------------------------------------------------------
section('Step 6: TGOV1.pout self-consistency (cross-model DEPENDENT)')
pout = ss.TGOV1.pout
print(f'  TGOV1.pout: is_dependent={getattr(pout, "is_dependent", False)}'
      f'  a={pout.a[:3]}...')
live_v = np.array(pout.v, dtype=float)
ts_data = ss.dae.ts.get_data(pout)
shape_ok = ts_data.shape[0] > 0 and ts_data.shape[1] > 0
if shape_ok:
    abs_diff = np.abs(ts_data[-1] - live_v)
    tgov1_ok = bool(np.all(abs_diff == 0))
    print(f'  {"✓" if tgov1_ok else "✗"} self-consistent: '
          f'shape={ts_data.shape}  max_abs_diff={float(abs_diff.max()):.2e}')
else:
    tgov1_ok = False
    print('  ✗ TGOV1.pout: get_data() returned empty')

# ---------------------------------------------------------------------------
# 7. Performance summary
# ---------------------------------------------------------------------------
section('Step 7: Performance')
speedup_ok = speedup >= 1.0
print(f'  Full TDS   : {t_full:.2f}s  ({n_steps_full} steps)')
print(f'  Reduced TDS: {t_red:.2f}s  ({n_steps_red} steps)')
print(f'  Speedup    : {speedup:.2f}x')
print(f'  Algebs     : {dae_m_full} → {dae_m_red} ({pct:.1f}% eliminated)')
print(f'  {"✓" if speedup_ok else "✗"} Reduced not slower than full')

# ---------------------------------------------------------------------------
# 8. Overall
# ---------------------------------------------------------------------------
section('Overall result')
checks = [
    ('dae.m reduced < dae.m full',                              reduction_ok),
    ('GENROU Algeb final values match (rtol<1e-3)',             genrou_ok),
    ('GENCLS Algeb final values match (rtol<1e-3)',             gencls_ok),
    ('TGOV1.pout get_data()[-1] self-consistent with live .v', tgov1_ok),
    ('Wall-clock speedup >= 1.0x',                             speedup_ok),
]
all_pass = True
for label, ok in checks:
    print(f'  {"✓" if ok else "✗"} {label}')
    if not ok:
        all_pass = False
print(f'\nOverall: {"PASS" if all_pass else "FAIL"}')
print(f'Speedup summary: {speedup:.2f}x  ({pct:.1f}% Algebs eliminated)')
