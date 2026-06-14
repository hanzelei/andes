"""
Integration methods for DAE.
"""

import logging
import numpy as np

from andes.routines.adaptive import accept_reject, check_adaptive_bust, weighted_rms_error
from andes.shared import sparse, matrix
from andes.routines.qndf import QNDF


logger = logging.getLogger(__name__)


class ImplicitIter:
    """
    Base class for implicit iterative methods.
    """
    nolte_event_steps = 0
    nolte_event_window = 0.0
    requires_variable_step = False

    @staticmethod
    def niter_next_h(niter, h, h_min, h_max):
        """
        Niter-based step-size heuristic shared by all methods.

        Returns a clamped next step size based on how many Newton iterations
        were needed for convergence.
        """
        if niter >= 15:
            h_new = h * 0.5
        elif niter <= 6:
            h_new = h * 1.1
        else:
            h_new = h * 0.95
        return min(max(h_new, h_min), h_max)

    @staticmethod
    def calc_h(tds):
        """
        Default step-size control: niter heuristic with fixt/shrinkt handling.
        """
        config = tds.config

        if tds.converged:
            tds.deltat = ImplicitIter.niter_next_h(
                tds.niter, tds.deltat, tds.deltatmin, tds.deltatmax)

            if config.fixt:
                tds.deltat = min(config.tstep, tds.deltat)

            if tds.chatter is True:
                tds.chatter = False
        else:
            if config.fixt and not config.shrinkt:
                tds.deltat = 0
                tds.busted = True
                tds.err_msg = (
                    f"Simulation did not converge with step size h={config.tstep:.4f}.\n"
                    "Reduce the step size `tstep`, or set `shrinkt = 1` to let it shrink."
                )
            else:
                tds.deltat *= 0.9
                if tds.deltat < tds.deltatmin:
                    tds.deltat = 0
                    tds.err_msg = "Time step reduced to zero. Convergence not likely."
                    tds.busted = True

    @staticmethod
    def calc_jac(tds, gxs, gys):
        pass

    @staticmethod
    def calc_q(x, f, Tf, h, x0, f0):
        pass

    @staticmethod
    def checkpoint_state(tds):
        """
        Snapshot DAE state for rollback.
        """
        dae = tds.system.dae
        return dae.x.copy(), dae.y.copy(), dae.f.copy()

    @staticmethod
    def restore_state(tds, state):
        """
        Restore DAE state from a checkpoint.
        """
        dae = tds.system.dae
        x_state, y_state, f_state = state
        dae.x[:] = x_state
        dae.y[:] = y_state
        dae.f[:] = f_state
        tds.system.vars_to_models()

    @staticmethod
    def solve_once(tds, h, method):
        """
        Run one implicit step with the given method and step size.
        """
        original_method = tds.method
        original_h = tds.h
        tds.method = method
        tds.h = h
        try:
            return ImplicitIter.step(tds)
        finally:
            tds.method = original_method
            tds.h = original_h

    @staticmethod
    def step(tds):
        """
        Integrate with Implicit Trapezoidal Method (ITM) to the current time.

        This function has an internal Newton-Raphson loop for algebraized semi-explicit DAE.
        The function returns the convergence status when done but does NOT progress simulation time.

        When ``tds.config.linesearch`` is enabled, applies nonmonotone backtracking
        line search using ``‖qg‖₂²`` as the merit function.  The ``g_scale * h``
        scaling ensures differential and algebraic residuals are comparable.
        The trial-point ``fg_update`` at the end of iteration *k* doubles as the
        initial evaluation for iteration *k+1*, making it zero-cost when the full
        step is accepted.

        Returns
        -------
        bool
            Convergence status in ``tds.converged``.

        """
        system = tds.system
        dae = tds.system.dae

        if tds.h == 0:
            logger.error("Current step size is zero. Integration is not permitted.")
            return False

        tds.mis = [1]
        tds.mis_inc = [1]

        tds.niter = 0
        tds.converged = False

        tds.x0[:] = dae.x
        tds.y0[:] = dae.y
        tds.f0[:] = dae.f

        use_ls = tds.config.linesearch
        if use_ls:
            merit_history = []
            _ls_merit_cache = None

        # initial residual evaluation (reused by first iteration)
        tds.fg_update(models=system.exist.pflow_tds)

        while True:
            # lazy Jacobian update

            reason = ''
            if dae.t == 0:
                reason = 't=0'
            elif tds.config.honest:
                reason = 'using honest method'
            elif tds.custom_event:
                reason = 'custom event set'
            elif not tds.last_converged:
                reason = 'non-convergence in the last step'
            elif tds.niter > 4 and (tds.niter + 1) % 3 == 0:
                reason = 'every 3 iterations beyond 4 iterations'
            elif dae.t - tds._last_switch_t < 0.1:
                reason = 'within 0.1s of event'

            if reason:
                system.j_update(models=system.exist.pflow_tds, info=reason)

                # set flag in `solver.worker.factorize`, not `solver.factorize`.
                tds.solver.worker.factorize = True

            # `Tf` should remain constant throughout the simulation, even if the corresponding diff. var.
            # is pegged by the anti-windup limiters.

            # solve implicit trapezoidal method (ITM) integration
            if tds.config.g_scale > 0:
                gxs = tds.config.g_scale * tds.h * dae.gx
                gys = tds.config.g_scale * tds.h * dae.gy
            else:
                gxs = dae.gx
                gys = dae.gy

            # calculate complete Jacobian matrix ``Ac```
            tds.Ac = tds.method.calc_jac(tds, gxs, gys)

            # equation `tds.qg[:dae.n] = 0` is the implicit form of differential equations using ITM
            tds.qg[:dae.n] = tds.method.calc_q(dae.x, dae.f, dae.Tf, tds.h, tds.x0, tds.f0)

            # reset the corresponding q elements for pegged anti-windup limiter
            for item in system.antiwindups:
                for key, _, eqval in item.x_set:
                    np.put(tds.qg, key, eqval)

            # set or scale the algebraic residuals
            if tds.config.g_scale > 0:
                tds.qg[dae.n:] = tds.config.g_scale * tds.h * dae.g
            else:
                tds.qg[dae.n:] = dae.g

            # calculate variable corrections; convert solver output to numpy 1D
            # array once so all downstream ops (nan check, tiny reset, slicing)
            # use fast numpy paths instead of repeated implicit CVXOPT conversions.
            if not tds.config.linsolve:
                inc = np.array(tds.solver.solve(tds.Ac, matrix(tds.qg))).ravel()
            else:
                inc = np.array(tds.solver.linsolve(tds.Ac, matrix(tds.qg))).ravel()

            # check for np.nan first
            if np.isnan(inc).any():
                tds.err_msg = 'NaN found in solution. Convergence is not likely'
                tds.niter = tds.config.max_iter + 1
                tds.busted = True
                break

            # reset tiny values to reduce chattering
            inc_abs = np.abs(inc)
            if tds.config.reset_tiny:
                inc[inc_abs < tds.tol_zero] = 0

            # store `inc` to tds for debugging
            tds.inc = inc

            # retrieve maximum abs. residual and maximum var. correction
            mis_arg = int(inc_abs.argmax())
            mis_inc = inc[mis_arg]

            mis_qg = float(np.max(np.abs(tds.qg)))

            # store initial maximum mismatch
            if tds.niter == 0:
                tds.mis[0] = mis_qg
                tds.mis_inc[0] = abs(mis_inc)
            else:
                tds.mis.append(mis_qg)
                tds.mis_inc.append(mis_inc)

            mis = float(inc_abs[mis_arg])

            # chattering detection
            if tds.niter > tds.config.chatter_iter:
                if (abs(sum(tds.mis_inc[-2:])) < 1e-6) and abs(tds.mis_inc[-1]) > 1e-4:
                    tds.chatter = True

                    logger.debug("Chattering detected at t=%s s", dae.t)
                    logger.debug("Chattering variable: %s", dae.xy_name[mis_arg])

            # --- apply step ---
            inc_x = inc[:dae.n]
            inc_y = inc[dae.n: dae.n + dae.m]

            if use_ls:
                # qg here is identical to the trial-point qg from the previous
                # NR iteration's line search, so merit_new from that iteration
                # equals merit_old for this one — reuse it when available.
                if _ls_merit_cache is not None:
                    merit_old = _ls_merit_cache
                else:
                    merit_old = float(np.dot(tds.qg, tds.qg))
                merit_history.append(merit_old)
                merit_ref = max(merit_history[-3:])

                tds.xs[:] = dae.x
                tds.ys[:] = dae.y

                alpha = 1.0
                for _ in range(4):
                    dae.x[:] = tds.xs - alpha * inc_x
                    dae.y[:] = tds.ys - alpha * inc_y
                    system.vars_to_models()
                    tds.fg_update(models=system.exist.pflow_tds)

                    # recompute qg at trial point
                    tds.qg[:dae.n] = tds.method.calc_q(
                        dae.x, dae.f, dae.Tf, tds.h, tds.x0, tds.f0)
                    for item in system.antiwindups:
                        for key, _, eqval in item.x_set:
                            np.put(tds.qg, key, eqval)
                    if tds.config.g_scale > 0:
                        tds.qg[dae.n:] = tds.config.g_scale * tds.h * dae.g
                    else:
                        tds.qg[dae.n:] = dae.g

                    merit_new = float(np.dot(tds.qg, tds.qg))
                    if merit_new < merit_ref:
                        break

                    alpha *= 0.5
                    logger.debug("TDS line search backtrack: alpha=%.4g at t=%.6f",
                                 alpha, dae.t)

                _ls_merit_cache = merit_new
            else:
                dae.x -= inc_x
                dae.y -= inc_y
                system.vars_to_models()
                tds.fg_update(models=system.exist.pflow_tds)

            tds.niter += 1

            # converged
            if abs(mis) <= tds.config.tol:
                tds.converged = True
                break

            if tds.chatter:
                tds.converged = True
                break

            # non-convergence cases
            if tds.niter > tds.config.max_iter:
                break

            if (abs(mis) > 1e6) and (abs(mis) > 1e6 * tds.mis[0]):
                tds.err_msg = 'Error increased too quickly.'
                break

        if not tds.converged:

            # restore variables and f
            dae.x[:] = np.array(tds.x0)
            dae.y[:] = np.array(tds.y0)
            dae.f[:] = np.array(tds.f0)
            system.vars_to_models()

            # debug outputs
            if tds.busted:
                logger.debug('* NaN in solution at t=%.6f, h=%.6f', dae.t, tds.h)
            else:
                logger.debug('* Max. iter. %d reached for t=%.6f, h=%.6f, max inc=%.4g',
                             tds.config.max_iter, dae.t, tds.h, mis)

                if logger.isEnabledFor(logging.DEBUG):
                    g_max = np.argmax(abs(dae.g))
                    inc_max = np.argmax(abs(inc))
                    tds._debug_g(g_max)
                    tds._debug_ac(inc_max)

        else:

            logger.debug('Converged in %d steps for t=%.6f, h=%.6f, max inc=%.4g',
                         tds.niter, dae.t, tds.h, mis)

        tds.last_converged = tds.converged

        return tds.converged


class BackEuler(ImplicitIter):
    """
    Backward Euler's integration method.
    """
    @staticmethod
    def calc_jac(tds, gxs, gys):
        """
        Build full Jacobian matrix ``Ac`` for Trapezoid method.
        """

        dae = tds.system.dae

        return sparse([[tds.Teye - tds.h * dae.fx, gxs],
                       [-tds.h * dae.fy, gys]], 'd')

    @staticmethod
    def calc_q(x, f, Tf, h, x0, f0):
        """
        Calculate the residual of algebraized differential equations.

        Notes
        -----
        Numba jit somehow slows down this function for the 14-bus
        and the 2k-bus systems.
        """

        return Tf * (x - x0) - h * f


class Trapezoid(ImplicitIter):
    """
    Trapezoidal methods.
    """

    @staticmethod
    def calc_jac(tds, gxs, gys):
        """
        Build full Jacobian matrix ``Ac`` for Trapezoid method.
        """

        dae = tds.system.dae

        return sparse([[tds.Teye - tds.h * 0.5 * dae.fx, gxs],
                       [-tds.h * 0.5 * dae.fy, gys]], 'd')

    @staticmethod
    def calc_q(x, f, Tf, h, x0, f0):
        """
        Calculate the residual of algebraized differential equations.

        Notes
        -----
        Numba jit somehow slows down this function for the 14-bus
        and the 2k-bus systems.
        """

        return Tf * (x - x0) - h * 0.5 * (f + f0)


class TrapezoidAdaptive(Trapezoid):
    """
    Adaptive trapezoid with step-doubling LTE estimation.

    The LTE estimate is based on the difference between one full step of size
    ``h`` and two half-steps of size ``h/2``.
    """
    nolte_event_steps = 4
    nolte_event_window = 0.1
    requires_variable_step = True
    _trap_solver = Trapezoid()

    @staticmethod
    def calc_h(tds):
        """
        Step size is set by ``step()``. Only check bust on failure.
        """
        check_adaptive_bust(tds)

    @staticmethod
    def _reject(tds, h_next, state=None):
        """
        Reject current candidate, optionally restoring the previous state.
        """
        if state is not None:
            ImplicitIter.restore_state(tds, state)
        # Keep predictor snapshots consistent with restored DAE state.
        dae = tds.system.dae
        if tds.x0 is not None:
            tds.x0[:] = dae.x
        if tds.y0 is not None:
            tds.y0[:] = dae.y
        if tds.f0 is not None:
            tds.f0[:] = dae.f
        tds.deltat = min(h_next, tds.deltatmax)
        tds.converged = False
        tds.last_converged = False
        return False

    @staticmethod
    def _nolte_next_h(tds, h):
        """
        Heuristic next step size when LTE control is disabled.
        """
        return ImplicitIter.niter_next_h(tds.niter, h, tds.deltatmin_adapt, tds.deltatmax)

    @staticmethod
    def step(tds):
        """
        One adaptive trapezoid step.

        Reads ``tds.h`` and writes ``tds.deltat``.
        Returns True when accepted, False when rejected.
        """
        dae = tds.system.dae
        h = tds.h

        if h == 0:
            logger.error("Current step size is zero. Integration is not permitted.")
            return False

        n = dae.n
        trap = TrapezoidAdaptive._trap_solver

        # Restart mode near events: use converged trapezoid steps without LTE
        # for a few steps and/or for a short event-time window.
        use_nolte = tds._adaptive_nolte_steps > 0
        if (not use_nolte) and (tds.method.nolte_event_window > 0.0):
            use_nolte = (dae.t - tds._last_switch_t) < tds.method.nolte_event_window

        if use_nolte:
            accepted = ImplicitIter.solve_once(tds, h, trap)
            if accepted:
                if tds._adaptive_nolte_steps > 0:
                    tds._adaptive_nolte_steps -= 1
                tds.deltat = TrapezoidAdaptive._nolte_next_h(tds, h)
                tds.converged = True
                tds.last_converged = True
                return True

            tds.deltat = min(max(h * 0.5, tds.deltatmin_adapt), tds.deltatmax)
            tds.converged = False
            tds.last_converged = False
            return False

        # algebraic-only systems: fallback to a single trapezoid solve
        if n == 0:
            accepted = ImplicitIter.solve_once(tds, h, trap)
            if accepted:
                tds.deltat = min(h, tds.deltatmax)
            else:
                tds.deltat = min(h * 0.5, tds.deltatmax)
            tds.converged = accepted
            tds.last_converged = accepted
            return accepted

        state0 = ImplicitIter.checkpoint_state(tds)
        x_prev = dae.x[:n].copy()

        # one full step with h
        ok_full = ImplicitIter.solve_once(tds, h, trap)
        if not ok_full:
            # Base Newton path already rolled state back.
            return TrapezoidAdaptive._reject(tds, h * 0.5)
        x_full = dae.x[:n].copy()

        # restore and run two half-steps
        ImplicitIter.restore_state(tds, state0)

        if not ImplicitIter.solve_once(tds, 0.5 * h, trap):
            return TrapezoidAdaptive._reject(tds, h * 0.5, state0)

        if not ImplicitIter.solve_once(tds, 0.5 * h, trap):
            return TrapezoidAdaptive._reject(tds, h * 0.5, state0)

        # second half-step result is already in dae.x / dae.y
        x_half = dae.x[:n]
        err_wt = np.empty_like(x_half)
        err_vec = (x_half - x_full) / 3.0
        err_est = weighted_rms_error(err_vec, x_prev, x_half,
                                     tds.config.abstol, tds.config.reltol, err_wt)

        accepted, h_next, _ = accept_reject(
            err_est=err_est,
            h=h,
            deltatmax=tds.deltatmax,
            order=2,
            accept_safety=0.9,
            accept_min_factor=0.2,
            accept_max_factor=2.0,
            reject_safety=0.9,
            reject_min_factor=0.2,
            reject_max_factor=0.9,
            repeat_reject_after=999,   # no failure counter for trapezoid-adaptive
            repeat_reject_factor=1.0,
        )

        if accepted:
            tds.deltat = h_next
            tds.converged = True
            tds.last_converged = True
            return True

        # Reject — skip LTE for enough steps that the 1.1x/step growth
        # can recover from the worst-case 0.2x shrink (1.1^20 ≈ 6.7 > 5).
        tds._adaptive_nolte_steps = max(tds._adaptive_nolte_steps, 20)
        return TrapezoidAdaptive._reject(tds, h_next, state0)


# --- solution method name-to-class mapping ---
# !!! add new solvers to below

method_map = {"trapezoid": Trapezoid,
              "trap_adapt": TrapezoidAdaptive,
              "backeuler": BackEuler,
              'qndf': QNDF,
              }
