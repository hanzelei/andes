"""
Phase 7b: Time profiling — full DAE vs reduced DAE TDS on NPCC.

Patches key methods with timers to produce a per-step breakdown.
Also runs cProfile on the full TDS loop for function-level detail.

Usage
-----
    conda activate andesre
    cd /Users/Shared/work/andes
    python icebar/dae-reduction/p7b_profiling.py
"""

import time
import cProfile
import pstats
import io
import numpy as np
import andes
from andes.utils.dae_reduction import DaeReductionAnalyser


def section(title):
    print(f'\n{"=" * 80}')
    print(f'  {title}')
    print(f'{"=" * 80}')


TF    = 5.0      # short run — enough to capture per-step costs
CASE  = andes.get_case('npcc/npcc.xlsx')
TOPN  = 20       # top-N functions to show in cProfile


# ---------------------------------------------------------------------------
# Helpers: monkey-patch timing onto a bound method
# ---------------------------------------------------------------------------
_timers: dict = {}   # name -> list of floats (seconds per call)

def wrap(obj, method_name, label):
    orig = getattr(obj, method_name)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        r  = orig(*args, **kwargs)
        _timers.setdefault(label, []).append(time.perf_counter() - t0)
        return r
    import functools
    timed = functools.wraps(orig)(timed)
    setattr(obj, method_name, timed)

def report_timers(timers):
    total_all = sum(sum(v) for v in timers.values())
    rows = []
    for label, vals in sorted(timers.items(), key=lambda kv: -sum(kv[1])):
        tot = sum(vals)
        rows.append((label, len(vals), tot, 1000*tot/len(vals) if vals else 0,
                     100*tot/total_all if total_all else 0))
    print(f'\n  {"Function":<40} {"calls":>6} {"total(s)":>9} {"ms/call":>9} {"share%":>7}')
    print(f'  {"-"*40} {"-"*6} {"-"*9} {"-"*9} {"-"*7}')
    for label, calls, tot, ms, pct in rows:
        print(f'  {label:<40} {calls:>6} {tot:>9.3f} {ms:>9.3f} {pct:>7.1f}%')
    print(f'  {"TOTAL":<40} {"":>6} {total_all:>9.3f}')
    return total_all


# ===========================================================================
# 1. FULL TDS — cProfile + manual per-step timers
# ===========================================================================
section('Full TDS profiling (NPCC, tf=5s)')

ss_full = andes.load(CASE, setup=True, default_config=True, no_output=True)
ss_full.TDS.config.tf = TF
ss_full.PFlow.run()

_timers.clear()
wrap(ss_full, 'g_update', 'g_update')
wrap(ss_full, 'j_update', 'j_update')
wrap(ss_full, 'vars_to_models', 'vars_to_models')

pr = cProfile.Profile()
pr.enable()
t0_full = time.perf_counter()
ss_full.TDS.run()
t_full = time.perf_counter() - t0_full
pr.disable()

print(f'\nFull TDS wall={t_full:.3f}s  n_steps={len(ss_full.dae.ts.t)}')
t_full_measured = report_timers(dict(_timers))

# cProfile top-N
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(TOPN)
print('\n--- cProfile top functions (full TDS) ---')
for line in s.getvalue().split('\n')[5:5+TOPN+3]:
    print(' ', line)


# ===========================================================================
# 2. REDUCED TDS — same patches + reduced-specific functions
# ===========================================================================
section('Reduced TDS profiling (NPCC, tf=5s)')

ss = andes.load(CASE, setup=True, default_config=True, no_output=True)
ss.TDS.config.tf = TF
ss.PFlow.run()

result = DaeReductionAnalyser.analyse(ss, verbose=False)
result = DaeReductionAnalyser.generate_reduced_functions(ss, result, verbose=False)
ss._dae_result = result

_timers.clear()
wrap(ss, 'g_update', 'g_update')
wrap(ss, 'j_update', 'j_update')
wrap(ss, 'vars_to_models', 'vars_to_models')

# Wrap dae_reduction functions directly
import andes.utils.dae_reduction as dr_mod
_orig_propagate = dr_mod.propagate_dep_values
def _timed_propagate(system, dae_result):
    t0 = time.perf_counter()
    r  = _orig_propagate(system, dae_result)
    _timers.setdefault('propagate_dep_values', []).append(time.perf_counter() - t0)
    return r
dr_mod.propagate_dep_values = _timed_propagate

# Also patch the FD Jacobian builder inside j_update via the system method
_orig_j_update_red = ss.j_update.__func__ if hasattr(ss.j_update, '__func__') else None

pr2 = cProfile.Profile()
pr2.enable()
t0_red = time.perf_counter()
ss.TDS.run()
t_red = time.perf_counter() - t0_red
pr2.disable()

# Restore
dr_mod.propagate_dep_values = _orig_propagate

print(f'\nReduced TDS wall={t_red:.3f}s  n_steps={len(ss.dae.ts.t)}')
t_red_measured = report_timers(dict(_timers))

# cProfile top-N
s2 = io.StringIO()
ps2 = pstats.Stats(pr2, stream=s2).sort_stats('cumulative')
ps2.print_stats(TOPN)
print('\n--- cProfile top functions (reduced TDS) ---')
for line in s2.getvalue().split('\n')[5:5+TOPN+3]:
    print(' ', line)


# ===========================================================================
# 3. Summary
# ===========================================================================
section('Summary')
print(f'  Full TDS    : {t_full:.3f}s  ({len(ss_full.dae.ts.t)} steps)')
print(f'  Reduced TDS : {t_red:.3f}s  ({len(ss.dae.ts.t)} steps)')
print(f'  Slowdown    : {t_red/t_full:.1f}x')
print(f'  dae.m full={ss_full.dae.m}  reduced={ss.dae.m}  '
      f'({100*(ss_full.dae.m-ss.dae.m)/ss_full.dae.m:.0f}% Alg eliminated)')
