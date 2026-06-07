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

            # Rebuild symbolic data (may have been cleared for cached pycode)
            try:
                model.syms.generate_symbols()
                model.syms.generate_equations()
                model.syms.generate_jacobians()
            except Exception as exc:
                logger.warning('Symbolic rebuild failed for %s: %s', mname, exc)
                result.errors[mname] = f'symbolic rebuild: {exc}'
                continue

            if model.syms.dg_syms.shape == (0, 0):
                continue

            # Algeb-Algeb Jacobian sub-block
            n_states = len(model.cache.states_and_ext)
            jac_gg   = model.syms.dg_syms[:, n_states:]

            # Iterative peeling
            try:
                eval_order, ind_core, ext_contrib = _iterative_peel(
                    model, jac_gg, name_val_ref)
            except Exception as exc:
                logger.warning('Peeling failed for %s: %s', mname, exc)
                result.errors[mname] = f'peeling: {exc}'
                continue

            # Resolve DEPENDENT formulas to State/param-only form
            try:
                resolved = _resolve_formulas(eval_order)
            except Exception as exc:
                logger.warning('Formula resolution failed for %s: %s', mname, exc)
                result.errors[mname] = f'resolution: {exc}'
                continue

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
