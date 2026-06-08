"""
Offline DAE-reduction analysis for ANDES.

Identifies Algebs whose g-equation is directly solvable without Newton
iteration ("DEPENDENT Algebs") by running an iterative-peeling algorithm
on each model's Algeb-Algeb Jacobian sub-block.

Typical usage (after pflow has run to populate model .v arrays)::

    from andes.utils.dae_reduction import DaeReductionAnalyser
    result = DaeReductionAnalyser.analyse(ss)
    print(result.summary())

    # Phase 3: generate reduced callables (must be called right after analyse())
    result = DaeReductionAnalyser.generate_reduced_functions(ss, result)
    # result.reduced_fns['GENROU'].g_reduced_fn(**name_val) → residual tuple

Dict keys are ``"ModelName.algeb_name"`` strings — unique across the whole
system even when multiple models define an Algeb with the same local name
(e.g. both TGOV1 and IEEEG1 define ``wref``).

  result.dependent   — Algebs eliminated from Newton; each has a resolved
                       closed-form runtime formula (States/params/ExtAlgebs).
  result.independent — Algebs that remain in Newton.
  result.ext_contrib — ExtAlgeb output slots (diagonal=0 in Gy).
  result.eval_order  — topological evaluation order for DEPENDENT Algebs.
  result.reduced_fns — per-model reduced callables (populated by
                       generate_reduced_functions()).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlgebRecord:
    """Holds metadata for one classified Algeb."""
    model_name: str
    algeb_name: str
    category: str                      # 'dependent' | 'independent' | 'ext_contrib'
    formula_sympy: sp.Expr | None      # resolved; None for non-DEPENDENT
    formula_fn: Callable | None        # lambdified callable; None for non-DEPENDENT
    eval_round: int                    # peeling round (0 for non-peeled)
    depends_on: list[str]             # qualified keys of Algebs this formula uses


@dataclass
class ModelReducedFunctions:
    """Reduced system callables for one model (populated by Phase 3).

    ``g_reduced_fn(**name_val)`` evaluates residuals for the INDEPENDENT
    Algebs after all DEPENDENT Algeb symbols have been substituted by their
    closed-form formulas.  It accepts the same ``{symbol_name: value}``
    calling convention as ``_lambdify_formula`` callables.

    ``dep_eval_fns[local_name](**name_val)`` evaluates one DEPENDENT Algeb.

    Both sets of functions reference only States, params, EXT_CONTRIB values,
    and INDEPENDENT Alg values — no circular dependencies.
    """
    g_reduced_fn:        Callable | None          # (**kw) → tuple of IND residuals
    g_reduced_args:      list[str]                # sorted free-symbol names
    g_reduced_ind_names: list[str]                # IND Algeb names in output order
    dep_eval_fns:        dict[str, Callable]      # local_name → (**kw) → value
    dep_eval_args:       dict[str, list[str]]     # local_name → sorted free-sym names
    n_independent:       int
    n_dependent:         int
    # Schur-complement correction Jacobian entries (Phase 5).
    # Each entry is (ind_algeb_name, sym_name, deriv_fn) where deriv_fn(**name_val)
    # evaluates ∂(g_reduced - g_orig)/∂sym for all devices.  These are the missing
    # coupling terms that arise when DEPENDENT Algeb columns are eliminated from Newton.
    corr_jac_entries:    list                     # [(ind_name, sym_name, fn), ...]


@dataclass
class DaeReductionResult:
    """System-wide classification of all Algebs.

    All dicts are keyed by ``"ModelName.algeb_name"`` to avoid collisions
    when multiple models define an Algeb with the same local name.
    """
    dependent:    dict[str, AlgebRecord]         = field(default_factory=dict)
    independent:  dict[str, AlgebRecord]         = field(default_factory=dict)
    ext_contrib:  dict[str, AlgebRecord]         = field(default_factory=dict)
    eval_order:   list[str]                      = field(default_factory=list)
    errors:       dict[str, str]                 = field(default_factory=dict)
    # Populated by generate_reduced_functions() — empty until Phase 3 runs
    reduced_fns:  dict[str, ModelReducedFunctions] = field(default_factory=dict)

    def summary(self) -> str:
        n_dep = len(self.dependent)
        n_ind = len(self.independent)
        n_ext = len(self.ext_contrib)
        n_tot = n_dep + n_ind + n_ext
        pct   = 100 * n_dep / (n_dep + n_ind) if (n_dep + n_ind) > 0 else 0
        lines = [
            'DaeReductionResult',
            f'  Total Algebs        : {n_tot}',
            f'  DEPENDENT (peeled)  : {n_dep}  ({pct:.0f}% of non-EXT)',
            f'  INDEPENDENT (Newton): {n_ind}',
            f'  EXT_CONTRIB         : {n_ext}',
            f'  Newton size         : {n_dep + n_ind}×{n_dep + n_ind} → '
            f'{n_ind}×{n_ind}',
        ]
        if self.errors:
            lines.append(f'  Errors ({len(self.errors)} models): '
                         f'{list(self.errors.keys())}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _build_name_val_ref(model, device_idx: int = 0) -> dict[str, float]:
    """
    Return {symbol_name: float} for one device.
    Used to select the correct branch for nonlinear g-equations (e.g.
    ``psi2 = ±sqrt(...)`` → pick the positive root).

    Covers: States, Algebs, ExtStates, ExtAlgebs, Params, ConstServices,
    and discrete sub-variables (SL_z0 from ``model.discrete['SL'].z0``).
    """
    name_val: dict[str, float] = {}

    for name in dir(model):
        if name.startswith('_'):
            continue
        try:
            obj = getattr(model, name)
        except Exception:
            continue
        if not hasattr(obj, 'v'):
            continue
        val = obj.v
        if val is None:
            continue
        try:
            if hasattr(val, '__len__'):
                if len(val) > device_idx:
                    name_val[name] = float(np.real(val[device_idx]))
                elif len(val) == 1:
                    name_val[name] = float(np.real(val[0]))
            else:
                name_val[name] = float(np.real(val))
        except (TypeError, ValueError):
            pass

    # Discrete sub-variables stored as plain np.ndarray on the discrete object
    if hasattr(model, 'discrete'):
        for disc_name, disc_obj in model.discrete.items():
            for zattr in ('z0', 'z1', 'z2', 'zl', 'zu', 'zi'):
                if not hasattr(disc_obj, zattr):
                    continue
                val = getattr(disc_obj, zattr)
                if isinstance(val, np.ndarray) and len(val) > device_idx:
                    name_val[f'{disc_name}_{zattr}'] = float(val[device_idx])

    return name_val


def _pick_best_root(
    solutions: list[sp.Expr],
    sym_i: sp.Symbol,
    name_val_ref: dict[str, float],
) -> sp.Expr:
    """
    Select the physically correct root for a nonlinear g-equation.

    Strategy:
    1. If ``name_val_ref[str(sym_i)]`` is a valid non-zero float, pick the
       root closest to it (uses actual pflow/TDS state as guide).
    2. Otherwise, evaluate each root at a unit point (all free symbols = 1.0,
       overridden by any available reference values) and pick the one with
       the **largest** value.  For symmetric pairs like ``±sqrt(...)``, this
       selects the positive root, which is always the physical one.
    """
    ref_val = name_val_ref.get(str(sym_i))

    def _eval(sol: sp.Expr) -> float:
        subs = {s: name_val_ref.get(str(s), 1.0) for s in sol.free_symbols}
        try:
            return float(sol.subs(subs))
        except (TypeError, ValueError):
            return float('nan')

    if ref_val is not None and abs(ref_val) > 1e-10:
        # Reference state available — pick the root closest to the known value.
        return sp.simplify(min(solutions, key=lambda s: abs(_eval(s) - ref_val)))

    # No reference (dynamic model .v not yet populated after pflow-only).
    # Evaluate at unit point; prefer the largest result (positive > negative
    # for symmetric roots; does not matter for distinct roots).
    evaluated = [(sol, _eval(sol)) for sol in solutions]
    finite    = [(sol, v) for sol, v in evaluated if not np.isnan(v)]
    if finite:
        best = max(finite, key=lambda x: x[1])[0]
        return sp.simplify(best)
    return sp.simplify(solutions[0])


def _jac_entry_nonzero(entry: sp.Expr) -> bool:
    """Return True if the Jacobian entry is structurally non-zero."""
    if entry == 0:
        return False
    try:
        return sp.simplify(entry) != 0
    except Exception:
        return entry != 0


def _iterative_peel(
    model,
    jac_gg: sp.Matrix,
    name_val_ref: dict[str, float],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Run iterative peeling on the Algeb-Algeb Jacobian sub-block.

    Returns
    -------
    eval_order : list of dicts
        ``{round, name, sym, formula, depends_on_algebs}`` — topo order.
    independent_core : list of str
    ext_contrib : list of str
    """
    algeb_names = list(model.cache.algebs_and_ext.keys())
    n_states    = len(model.cache.states_and_ext)
    algeb_syms  = model.syms.vars_list[n_states:]
    name_to_idx = {n: i for i, n in enumerate(algeb_names)}

    ext_contrib: list[str] = []
    candidates:  list[int] = []
    for i, name in enumerate(algeb_names):
        diag = jac_gg[i, i]
        try:
            is_zero = (diag == 0) or (sp.simplify(diag) == 0)
        except Exception:
            is_zero = False
        if is_zero:
            ext_contrib.append(name)
        else:
            candidates.append(i)

    remaining:  list[int]  = list(candidates)
    eval_order: list[dict] = []
    rnd = 0

    while True:
        rnd += 1
        peeled: list[dict] = []

        for i in remaining:
            off_diag = [j for j in remaining
                        if j != i and _jac_entry_nonzero(jac_gg[i, j])]
            if off_diag:
                continue

            try:
                diag_entry = jac_gg[i, i]
                g_expr     = model.syms.g_matrix[i]
                sym_i      = algeb_syms[i]
                rest       = g_expr - diag_entry * sym_i
                formula    = sp.simplify(-rest / diag_entry)

                # Nonlinear g-equation: sym_i still appears in formula.
                if sym_i in formula.free_symbols:
                    solutions = sp.solve(g_expr, sym_i)
                    if solutions:
                        formula = _pick_best_root(solutions, sym_i, name_val_ref)
            except Exception as exc:
                logger.debug('Formula extraction failed for %s.%s: %s',
                             model.class_name, algeb_names[i], exc)
                # Skip this Algeb — leave it in remaining (treat as INDEPENDENT)
                continue

            free_names = {str(s) for s in formula.free_symbols}
            dep_algebs = sorted(free_names & set(algeb_names))
            peeled.append({
                'round':             rnd,
                'name':              algeb_names[i],
                'sym':               sym_i,
                'formula':           formula,
                'depends_on_algebs': dep_algebs,
            })

        if not peeled:
            break
        for rec in peeled:
            remaining.remove(name_to_idx[rec['name']])
            eval_order.append(rec)

    independent_core = [algeb_names[i] for i in remaining]
    return eval_order, independent_core, ext_contrib


def _resolve_formulas(eval_order: list[dict]) -> dict[str, sp.Expr]:
    """
    Substitute DEPENDENT-into-DEPENDENT in topological order until each
    resolved formula references only States, params, and ExtAlgebs.

    Returns ``{local_algeb_name: resolved_sp.Expr}``.
    """
    sym_map  = {rec['name']: rec['sym'] for rec in eval_order}
    resolved: dict[str, sp.Expr] = {}

    for rec in eval_order:
        formula = rec['formula']
        for dep_name in rec['depends_on_algebs']:
            if dep_name in resolved:
                formula = formula.subs(sym_map[dep_name], resolved[dep_name])
        resolved[rec['name']] = sp.simplify(formula)

    return resolved


def _lambdify_formula(formula: sp.Expr) -> Callable | None:
    """
    Lambdify a resolved formula into a numpy-compatible callable.

    The callable signature is ``fn(**name_val)`` where ``name_val`` maps
    symbol name strings to scalar floats or numpy arrays.
    Returns None on failure.
    """
    free_syms  = sorted(formula.free_symbols, key=str)
    sym_names  = [str(s) for s in free_syms]

    if not free_syms:
        val = float(formula)
        return lambda **_kw: val

    try:
        fn = sp.lambdify(free_syms, formula, modules='numpy')
        def _caller(**kwargs) -> np.ndarray:
            args = [kwargs[n] for n in sym_names]
            return fn(*args)
        _caller.__doc__ = f'formula({", ".join(sym_names)})'
        return _caller
    except Exception as exc:
        logger.warning('lambdify failed for formula %s: %s', formula, exc)
        return None


# ---------------------------------------------------------------------------
# Phase 3: reduced function builder
# ---------------------------------------------------------------------------

def _build_model_reduced_functions(
    model,
    dep_recs: dict,
    ind_recs: dict,
) -> ModelReducedFunctions:
    """
    Build ``ModelReducedFunctions`` for one model.

    Must be called while ``model.syms.g_matrix`` and ``model.syms.vars_list``
    are still populated (i.e., immediately after ``analyse()`` without
    ``generate_pycode()`` clearing the symbolic data).

    Parameters
    ----------
    model :
        ANDES model with symbolic data already built.
    dep_recs : dict
        ``{local_algeb_name: AlgebRecord}`` for DEPENDENT Algebs.
    ind_recs : dict
        ``{local_algeb_name: AlgebRecord}`` for INDEPENDENT Algebs.

    Returns
    -------
    ModelReducedFunctions
    """
    algeb_names = list(model.cache.algebs_and_ext.keys())
    n_states    = len(model.cache.states_and_ext)
    algeb_syms  = model.syms.vars_list[n_states:]
    name_to_sym = {n: s for n, s in zip(algeb_names, algeb_syms)}

    # DEPENDENT symbol → resolved SymPy formula  (formulas reference only
    # States / params / EXT_CONTRIB; no DEPENDENT Alg symbols remain)
    dep_subs = {}
    for name, rec in dep_recs.items():
        sym = name_to_sym.get(name)
        if sym is not None and rec.formula_sympy is not None:
            dep_subs[sym] = rec.formula_sympy

    # INDEPENDENT Algeb names in the order they appear in algeb_names
    ind_names_ordered = [n for n in algeb_names if n in ind_recs]

    # -----------------------------------------------------------------
    # g_reduced: substitute DEPENDENT symbols into INDEPENDENT g-rows
    # -----------------------------------------------------------------
    g_reduced_exprs: list[sp.Expr] = []
    for ind_name in ind_names_ordered:
        idx    = algeb_names.index(ind_name)
        g_expr = model.syms.g_matrix[idx]
        g_sub  = g_expr.subs(dep_subs)
        # Note: sp.simplify omitted here — very slow for large expressions.
        # Raw substituted form is sufficient for lambdification.
        g_reduced_exprs.append(g_sub)

    # Collect all free symbols across all reduced residuals
    all_free_syms = sorted(
        {s for expr in g_reduced_exprs for s in expr.free_symbols},
        key=str,
    )
    all_free_names = [str(s) for s in all_free_syms]

    if g_reduced_exprs:
        g_tuple  = sp.Tuple(*g_reduced_exprs)
        _g_raw   = sp.lambdify(all_free_syms, g_tuple, modules='numpy')
        def _g_reduced_fn(**kwargs) -> tuple:
            return _g_raw(*[kwargs[n] for n in all_free_names])
        _g_reduced_fn.__doc__ = f'g_reduced({", ".join(all_free_names)})'
    else:
        # Empty INDEPENDENT core (e.g. TGOV1)
        def _g_reduced_fn(**kwargs) -> tuple:
            return ()
        _g_reduced_fn.__doc__ = 'g_reduced() — empty IND core'

    # -----------------------------------------------------------------
    # Correction Jacobian (Schur complement)
    # corr_k = g_reduced_k - g_orig_k  for each INDEPENDENT Algeb k.
    # ∂corr_k/∂sym is the missing coupling added back to the Newton Jacobian
    # for every non-DEPENDENT symbol sym that has a DAE address.
    # -----------------------------------------------------------------
    dep_sym_names = set(dep_recs.keys())
    corr_entries: list = []   # (ind_algeb_name, sym_name, deriv_fn)

    for k, ind_name in enumerate(ind_names_ordered):
        idx_k  = algeb_names.index(ind_name)
        g_orig = model.syms.g_matrix[idx_k]
        g_red  = g_reduced_exprs[k]
        try:
            corr = g_red - g_orig
        except Exception:
            continue
        if corr == 0:
            continue

        for sym in sorted(corr.free_symbols, key=str):
            sname = str(sym)
            if sname in dep_sym_names:
                continue   # DEPENDENT Algeb — column has no DAE address
            try:
                deriv = sp.diff(corr, sym)
                if deriv == 0:
                    continue
            except Exception:
                continue
            deriv_fn = _lambdify_formula(deriv)
            if deriv_fn is not None:
                corr_entries.append((ind_name, sname, deriv_fn))

    # -----------------------------------------------------------------
    # dep_eval functions (already lambdified in analyse(); reuse formula_fn
    # when available, otherwise lambdify again from formula_sympy)
    # -----------------------------------------------------------------
    dep_eval_fns:  dict[str, Callable] = {}
    dep_eval_args: dict[str, list[str]] = {}

    for name, rec in dep_recs.items():
        if rec.formula_fn is not None:
            fn = rec.formula_fn
        elif rec.formula_sympy is not None:
            fn = _lambdify_formula(rec.formula_sympy)
        else:
            continue
        dep_eval_fns[name]  = fn
        dep_eval_args[name] = sorted(
            [str(s) for s in rec.formula_sympy.free_symbols]
        ) if rec.formula_sympy is not None else []

    return ModelReducedFunctions(
        g_reduced_fn        = _g_reduced_fn,
        g_reduced_args      = all_free_names,
        g_reduced_ind_names = ind_names_ordered,
        dep_eval_fns        = dep_eval_fns,
        dep_eval_args       = dep_eval_args,
        n_independent       = len(ind_names_ordered),
        n_dependent         = len(dep_recs),
        corr_jac_entries    = corr_entries,
    )


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class DaeReductionAnalyser:
    """
    Classifies all Algebs system-wide as DEPENDENT, INDEPENDENT, or
    EXT_CONTRIB by running iterative peeling on each model's symbolic
    Algeb-Algeb Jacobian sub-block.

    Call ``analyse(system)`` after power flow has converged so that model
    ``.v`` arrays hold a valid operating point (required for nonlinear root
    selection in models like GENROU).
    """

    @staticmethod
    def analyse(system, verbose: bool = False) -> DaeReductionResult:
        """
        Classify every Algeb in *system* and extract DEPENDENT formulas.

        Parameters
        ----------
        system : andes.System
            Fully set-up system with pflow converged.
        verbose : bool
            Print per-model progress to stdout.

        Returns
        -------
        DaeReductionResult
        """
        result = DaeReductionResult()

        for mname, model in system.models.items():
            n_alg = len(model.cache.algebs_and_ext)
            if n_alg == 0 or model.n == 0:
                continue

            if verbose:
                print(f'  Analysing {mname} ({model.n} devices, {n_alg} Algebs) ...')

            # Build reference state BEFORE symbolic rebuild so we read the
            # pflow .v arrays (which are valid after pflow convergence).
            # These values are used for nonlinear root selection (e.g. psi2 sign).
            name_val_ref = _build_name_val_ref(model, device_idx=0)

            # Rebuild symbolic data (may have been cleared for cached pycode).
            # generate_equations() and generate_jacobians() overwrite calls.f/g/j
            # with lambdify versions that lack the 4 select_args in their
            # signatures — incompatible with the pycode-loaded versions.
            # Save the full set of affected attributes and restore after analysis.
            _saved_calls = {
                'f':              model.calls.f,
                'g':              model.calls.g,
                'f_args':         list(model.calls.f_args),
                'g_args':         list(model.calls.g_args),
                'j':              dict(model.calls.j),
                'j_args':         {k: list(v) for k, v in model.calls.j_args.items()},
                'ijac':           {k: list(v) for k, v in model.calls.ijac.items()},
                'jjac':           {k: list(v) for k, v in model.calls.jjac.items()},
                'vjac':           {k: list(v) for k, v in model.calls.vjac.items()},
                'j_names':        list(model.calls.j_names),
                'need_diag_eps':  list(model.calls.need_diag_eps),
            }
            try:
                model.syms.generate_symbols()
                model.syms.generate_equations()
                model.syms.generate_jacobians()

                if model.syms.dg_syms.shape == (0, 0):
                    continue

                # Algeb-Algeb Jacobian sub-block
                n_states = len(model.cache.states_and_ext)
                jac_gg   = model.syms.dg_syms[:, n_states:]

                # Iterative peeling
                eval_order, ind_core, ext_contrib = _iterative_peel(
                    model, jac_gg, name_val_ref)

                # Resolve DEPENDENT formulas to State/param-only form
                resolved = _resolve_formulas(eval_order)

                # Lambdify resolved formulas
                formula_fns: dict[str, Callable | None] = {
                    name: _lambdify_formula(formula)
                    for name, formula in resolved.items()
                }

                # Record results — key = "ModelName.algeb_name" to avoid collisions
                for rec in eval_order:
                    local = rec['name']
                    key   = f'{mname}.{local}'
                    dep_keys = [f'{mname}.{d}' for d in rec['depends_on_algebs']]
                    result.dependent[key] = AlgebRecord(
                        model_name   = mname,
                        algeb_name   = local,
                        category     = 'dependent',
                        formula_sympy= resolved[local],
                        formula_fn   = formula_fns.get(local),
                        eval_round   = rec['round'],
                        depends_on   = dep_keys,
                    )
                    result.eval_order.append(key)

                for local in ind_core:
                    key = f'{mname}.{local}'
                    result.independent[key] = AlgebRecord(
                        model_name   = mname,
                        algeb_name   = local,
                        category     = 'independent',
                        formula_sympy= None,
                        formula_fn   = None,
                        eval_round   = 0,
                        depends_on   = [],
                    )

                for local in ext_contrib:
                    key = f'{mname}.{local}'
                    result.ext_contrib[key] = AlgebRecord(
                        model_name   = mname,
                        algeb_name   = local,
                        category     = 'ext_contrib',
                        formula_sympy= None,
                        formula_fn   = None,
                        eval_round   = 0,
                        depends_on   = [],
                    )

            except Exception as exc:
                logger.warning('Analysis failed for %s: %s', mname, exc)
                result.errors[mname] = str(exc)

            finally:
                # Restore all calls attributes overwritten by generate_equations()
                # and generate_jacobians(), so TDS uses the pycode-loaded functions.
                for _k, _v in _saved_calls.items():
                    setattr(model.calls, _k, _v)

        return result

    @staticmethod
    def generate_reduced_functions(
        system,
        result: DaeReductionResult,
        verbose: bool = False,
    ) -> DaeReductionResult:
        """
        Phase 3: generate reduced callable functions for each model.

        Must be called **immediately after** ``analyse()`` while
        ``model.syms.g_matrix`` / ``model.syms.vars_list`` are still in
        memory (``generate_pycode()`` may clear them).

        For each model that has classified Algebs the method builds:

        * ``g_reduced_fn(**name_val)`` — residuals for INDEPENDENT Algebs
          with DEPENDENT symbols fully substituted by their formulas.
        * ``dep_eval_fns[name](**name_val)`` — evaluation callable for each
          DEPENDENT Algeb (reused from ``AlgebRecord.formula_fn``).

        Results are stored in ``result.reduced_fns[model_name]`` and the
        same ``result`` object is returned.

        Parameters
        ----------
        system :
            Fully set-up ANDES system (pflow must have converged).
        result :
            ``DaeReductionResult`` returned by ``analyse()``.
        verbose : bool
            Print per-model progress to stdout.

        Returns
        -------
        DaeReductionResult
            The same object with ``reduced_fns`` populated.
        """
        from collections import defaultdict

        dep_by_model: dict = defaultdict(dict)
        ind_by_model: dict = defaultdict(dict)

        for key, rec in result.dependent.items():
            dep_by_model[rec.model_name][rec.algeb_name] = rec
        for key, rec in result.independent.items():
            ind_by_model[rec.model_name][rec.algeb_name] = rec

        for mname, model in system.models.items():
            dep_recs = dep_by_model.get(mname, {})
            ind_recs = ind_by_model.get(mname, {})

            if not dep_recs and not ind_recs:
                continue

            if model.syms.g_matrix is None or model.syms.g_matrix.shape[0] == 0:
                logger.debug('Skipping %s: g_matrix not populated', mname)
                continue

            if verbose:
                print(f'  Building reduced functions for {mname} '
                      f'(DEP={len(dep_recs)}, IND={len(ind_recs)}) ...')

            try:
                fns = _build_model_reduced_functions(model, dep_recs, ind_recs)
                result.reduced_fns[mname] = fns
            except Exception as exc:
                logger.warning('generate_reduced_functions failed for %s: %s',
                               mname, exc)
                if verbose:
                    print(f'    FAILED: {exc}')

        return result


# ---------------------------------------------------------------------------
# Phase 5: runtime helpers for reduced TDS integration
# ---------------------------------------------------------------------------

def _build_name_val_arrays(model) -> dict:
    """
    Return ``{symbol_name: array_or_scalar}`` for all devices simultaneously.

    Like ``_build_name_val_ref`` but vectorized — attribute ``.v`` values are
    returned as-is (numpy arrays of shape ``(ndevice,)`` or scalars), so that
    lambdified formulas evaluate over all devices in one call via numpy
    broadcasting.
    """
    name_val: dict = {}
    for name in dir(model):
        if name.startswith('_'):
            continue
        try:
            obj = getattr(model, name)
        except Exception:
            continue
        if not hasattr(obj, 'v'):
            continue
        val = obj.v
        if val is None:
            continue
        try:
            if hasattr(val, '__len__'):
                name_val[name] = np.asarray(val, dtype=float)
            else:
                name_val[name] = float(np.real(val))
        except (TypeError, ValueError):
            pass

    if hasattr(model, 'discrete'):
        for disc_name, disc_obj in model.discrete.items():
            for zattr in ('z0', 'z1', 'z2', 'zl', 'zu', 'zi'):
                if not hasattr(disc_obj, zattr):
                    continue
                val = getattr(disc_obj, zattr)
                if isinstance(val, np.ndarray):
                    name_val[f'{disc_name}_{zattr}'] = val

    return name_val


def fix_dep_ext_algeb_shapes(system, result: 'DaeReductionResult') -> None:
    """
    Repair the shape of ExtAlgebs that reference DEPENDENT source variables.

    After ``set_address()`` with DAE reduction, ``link_external`` raises
    IndexError for ExtAlgebs whose source has ``a = []``.  This leaves the
    ExtAlgeb with ``n=0``, ``v=zeros(0)``, which causes shape-mismatch crashes
    in ``s_update`` during ``system.init()``.

    This sets ``.n`` and ``.v`` to the correct number of devices (zero values).
    Call ``propagate_dep_values`` afterwards to fill correct values.

    Handles both direct model references (``ext.model = 'GENROU'``) and group
    references (``ext.model = 'SynGen'``).  The check is: if ``ext.n == 0`` but
    the indexer has devices, the ExtAlgeb was silently broken by DEPENDENT
    source, so fix the shape.
    """
    for mdl in system.models.values():
        if not hasattr(mdl, 'algebs_ext'):
            continue
        for ext in mdl.algebs_ext.values():
            if ext.n != 0:
                continue   # already has correct shape
            # Determine expected number of devices from the indexer
            try:
                if ext.indexer is not None:
                    idx_v = ext.indexer.v
                    if hasattr(idx_v, '__len__'):
                        n_expected = len(idx_v)
                    else:
                        n_expected = 0
                else:
                    n_expected = 0
            except Exception:
                n_expected = 0

            if n_expected == 0:
                continue   # nothing to fix

            # The ExtAlgeb has n=0 but indexer says there should be devices —
            # this means link_external failed because the source is DEPENDENT.
            ext.n = n_expected
            ext.v = np.zeros(n_expected)
            ext.e = np.zeros(n_expected)   # also fix .e so g_update doesn't crash
            logger.debug(
                'fix_dep_ext_algeb_shapes: %s.%s → size %d',
                mdl.class_name, ext.name, n_expected,
            )


def propagate_dep_values(system, result: 'DaeReductionResult') -> None:
    """
    Evaluate DEPENDENT Algeb ``.v`` buffers in topological order, then copy
    values to ExtAlgeb readers that reference DEPENDENT sources.

    Call this:

    * Inside ``system.init()`` after each model's init cycle (see the hook in
      ``System.init()``), so the next model's ExtAlgebs receive correct values.
    * At the start of ``System.g_update()`` each Newton step, to keep DEPENDENT
      values consistent with current State / INDEPENDENT Algeb values.
    """
    # Step A: evaluate DEPENDENT Algeb .v in topological order.
    # Guards:
    # (1) Only models that are initialized — uninitialized models have their
    #     DEPENDENT .v set to 0 so that v_str_add=True works correctly during
    #     the model's own init() call.  After init(), Step A runs again and
    #     overwrites with the formula result.
    # (2) Only algebs that are truly eliminated from Newton (a=[]).
    #     Algebs with a valid DAE address (PFlow-only models skipped by
    #     set_address()) get their .v from vars_to_models(); overwriting with
    #     the formula may produce divide-by-zero when limiters are inactive.
    for qkey in result.eval_order:
        mname, lname = qkey.split('.')
        mdl = system.models[mname]
        if mdl.n == 0:
            continue
        if not getattr(mdl.flags, 'initialized', False):
            continue  # guard (1): skip uninitialized models
        rec = result.dependent[qkey]
        if rec.formula_fn is None:
            continue
        algeb = mdl.algebs.get(lname)
        if algeb is None:
            continue
        if len(getattr(algeb, 'a', [])) > 0:
            continue  # guard (2): still has DAE address → vars_to_models() handles it
        # Refresh PostInitServices (e.g. vref0) so their .v is current before
        # building the name-val dict.  s_update_post() is normally called after
        # ALL models are initialised, but here we need it before the formula runs.
        try:
            mdl.s_update_post()
        except Exception:
            pass
        name_val = _build_name_val_arrays(mdl)
        try:
            vals = rec.formula_fn(**name_val)
            vals_arr = np.asarray(vals, dtype=float)
            if vals_arr.ndim == 0:
                algeb.v[:] = float(vals_arr)
            elif vals_arr.shape == algeb.v.shape:
                algeb.v[:] = vals_arr
            else:
                algeb.v[:] = vals_arr.ravel()[:len(algeb.v)]
        except Exception as exc:
            logger.debug('propagate_dep_values %s.%s: %s', mname, lname, exc)

    # Step B: copy DEPENDENT Algeb values to ExtAlgeb readers.
    # Handles both direct model refs (ext.model = 'GENROU') and Group refs
    # (ext.model = 'SynGen').  For Groups, use Group.get() which routes to
    # the correct member model.  We check `is_dependent` on the resolved source
    # variable to avoid touching non-DEPENDENT ExtAlgebs.
    dep_keys = result.dependent  # set for fast lookup
    for mdl in system.models.values():
        if not hasattr(mdl, 'algebs_ext'):
            continue
        for ext in mdl.algebs_ext.values():
            if len(ext.v) == 0:
                continue  # shape not yet fixed, nothing to do
            src_obj = system.__dict__.get(ext.model)
            if src_obj is None:
                continue
            if ext.indexer is None:
                continue
            idx_v = ext.indexer.v
            if not hasattr(idx_v, '__len__') or len(idx_v) == 0:
                continue

            is_group = hasattr(src_obj, '_idx2model')
            try:
                if is_group:
                    # Group reference — check if src is DEPENDENT in any member
                    any_dep = any(
                        f'{m.class_name}.{ext.src}' in dep_keys
                        for m in src_obj._idx2model.values()
                    )
                    if not any_dep:
                        continue
                    vals = src_obj.get(ext.src, list(idx_v), attr='v')
                    ext.v[:] = np.asarray(vals, dtype=float)
                else:
                    # Direct model reference
                    src_var = src_obj.__dict__.get(ext.src)
                    if src_var is None or not getattr(src_var, 'is_dependent', False):
                        continue
                    uid = src_obj.idx2uid(idx_v)
                    if len(uid) == 0:
                        continue
                    if len(ext.v) != len(uid):
                        ext.n = len(uid)
                        ext.v = np.zeros(len(uid))
                    ext.v[:] = src_var.v[uid]
            except Exception as exc:
                logger.debug(
                    'propagate_dep_values ExtAlgeb %s.%s→%s.%s: %s',
                    mdl.class_name, ext.name, ext.model, ext.src, exc,
                )
