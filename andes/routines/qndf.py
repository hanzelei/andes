"""QNDF: Variable-order quasi-constant-step NDF method (Shampine & Reichelt 1997)."""

import logging

import numpy as np

from andes.routines.adaptive import accept_reject, candidate_h, check_adaptive_bust, weighted_rms_error
from andes.shared import sparse, matrix

logger = logging.getLogger(__name__)

MAX_ORDER = 5
KAPPA = np.array([-37/200, -1/9, -823/10000, -83/2000, 0], dtype=float)
GAMMA = np.zeros(MAX_ORDER + 1)
for _i in range(1, MAX_ORDER + 1):
    GAMMA[_i] = GAMMA[_i - 1] + 1.0 / _i


class QNDFCache:
    """Pre-allocated workspace. Created once at TDS.init()."""

    def __init__(self, n, abstol, reltol, max_order=MAX_ORDER):
        self.n = n

        self.max_order = max_order
        self.abstol = abstol
        self.reltol = reltol

        ncols = max_order + 3
        self.D = np.zeros((n, ncols))
        self.D_prev = np.zeros((n, ncols))
        self.D_scratch = np.zeros((n, ncols))

        self.x_pred = np.zeros(n)
        self.phi = np.zeros(n)
        self.dx_corr = np.zeros(n)
        self.x_prev = np.zeros(n)
        self.err_wt = np.zeros(n)

        self.R = np.zeros((max_order, max_order))
        self.U = np.zeros((max_order, max_order))
        self.RU = np.zeros((max_order, max_order))

        self.order = 1
        self.order_prev = 1
        self.h_prev = 0.0
        self.consec_steps = 0
        self.consec_fail_count = 0
        self.err_est = 1.0
        self.err_est_km1 = float('inf')
        self.err_est_kp1 = float('inf')
        self.first_step = True

    def reset_for_event(self, x_current):
        self.D[:] = 0.0
        self.D[:, 0] = x_current
        self.D_prev[:] = 0.0
        self.order = 1
        self.order_prev = 1
        self.h_prev = 0.0
        self.consec_steps = 0
        self.consec_fail_count = 0
        self.err_est = 1.0
        self.err_est_km1 = float('inf')
        self.err_est_kp1 = float('inf')
        self.first_step = True


# --- Stateless helpers (zero-alloc in hot path) ---


def compute_beta0(k):
    return 1.0 / ((1.0 - KAPPA[k - 1]) * GAMMA[k])


def error_constant(k):
    return KAPPA[k - 1] * GAMMA[k] + 1.0 / (k + 1)


def compute_U(k, U):
    for r in range(k):
        U[0, r] = -(r + 1)
        for j in range(1, k):
            U[j, r] = U[j - 1, r] * (j - (r + 1)) / (j + 1)


def compute_R(k, rho, R):
    for r in range(k):
        R[0, r] = -(r + 1) * rho
        for j in range(1, k):
            R[j, r] = R[j - 1, r] * (j - (r + 1) * rho) / (j + 1)


def rescale_D(cache, rho, k):
    if k < 1:
        return
    compute_R(k, rho, cache.R[:k, :k])
    compute_U(k, cache.U[:k, :k])
    RU = cache.R[:k, :k] @ cache.U[:k, :k]
    cache.D[:, 1:k+1] = cache.D[:, 1:k+1] @ RU


def compute_predictor(cache, k):
    cache.x_pred[:] = cache.D[:, k]
    for j in range(k - 1, 0, -1):
        cache.x_pred += cache.D[:, j]
    cache.x_pred += cache.D[:, 0]

    cache.phi[:] = 0.0
    for j in range(1, k + 1):
        cache.phi += GAMMA[j] * cache.D[:, j]


def update_D(D, dx_corr, k):
    D[:, k + 2] = dx_corr - D[:, k + 1]
    D[:, k + 1] = dx_corr
    for j in range(k, 0, -1):
        D[:, j] += D[:, j + 1]


def select_order_and_step(cache, h, order, err_est):
    """Pick optimal (order, step_size) from order-1, order, order+1."""
    h_cur_order = candidate_h(err_est, h, order,
                              safety=(1.0 / 1.2),
                              min_factor=0.0,
                              max_factor=10.0)
    order_new, h_new = order, h_cur_order

    if order > 1 and cache.consec_steps >= order:
        err_est_km1 = cache.err_est_km1
        h_lower = candidate_h(err_est_km1, h, order - 1,
                              safety=(1.0 / 1.3),
                              min_factor=0.0,
                              max_factor=10.0)
        if h_lower > h_new:
            order_new, h_new = order - 1, h_lower

    if order < cache.max_order and cache.consec_steps >= order + 2:
        err_est_kp1 = cache.err_est_kp1
        h_higher = candidate_h(err_est_kp1, h, order + 1,
                               safety=(1.0 / 1.4),
                               min_factor=0.0,
                               max_factor=10.0)
        if h_higher > h_new:
            order_new, h_new = order + 1, h_higher

    # Standard NDF step dampening: avoid small changes during warm-up
    # to reduce unnecessary Jacobian refactorizations.
    prefer_const = (cache.consec_steps < order + 2)
    if prefer_const:
        q = h / h_new if h_new > 0 else 1.0
        if 0.6 < q < 1.2:
            h_new = h

    return order_new, h_new


class QNDF:
    """Variable-order (1-5) quasi-constant-step NDF method."""
    nolte_event_steps = 0
    nolte_event_window = 0.0
    requires_variable_step = True

    @staticmethod
    def calc_h(tds):
        """
        Step size is set by ``step()``. Only check bust on failure.
        """
        check_adaptive_bust(tds)

    @staticmethod
    def step(tds):
        """One QNDF step. Returns True (accepted) or False (rejected).

        NEVER writes tds.h. Reads tds.h, writes tds.deltat.
        On return (True or False), tds.deltat holds the recommended next h.
        """
        system = tds.system
        dae = system.dae
        cache = tds.qndf_cache
        if cache is None:
            raise RuntimeError(
                "QNDF cache not initialized. Call TDS.init() after setting method='qndf'.")
        n = dae.n
        if n == 0:
            raise RuntimeError(
                "QNDF requires differential equations (dae.n > 0). "
                "Use 'trapezoid' or 'backeuler' for algebraic-only systems.")
        h = tds.h          # read once, never mutated

        if h == 0:
            return False

        k = cache.order

        # --- Save state for rollback ---
        tds.x0[:] = dae.x
        tds.y0[:] = dae.y
        tds.f0[:] = dae.f
        cache.x_prev[:] = dae.x[:n]

        # --- Prepare D table ---
        if cache.first_step:
            cache.D[:] = 0.0
            cache.D[:, 0] = dae.x[:n]
            cache.first_step = False
        elif cache.h_prev > 0 and abs(h - cache.h_prev) > 1e-14 * max(abs(h), 1e-30):
            rho = h / cache.h_prev
            rescale_D(cache, rho, k)

        # Save D AFTER init/rescale so rollback restores valid state.
        # On first-step Newton failure, D[:] = D_prev restores D[:,0] = dae.x[:n]
        # so the predictor on retry produces x_pred = x0, not x_pred = 0.
        cache.D_prev[:] = cache.D

        beta0 = compute_beta0(k)
        compute_predictor(cache, k)

        # --- Newton initial guess from predictor ---
        dae.x[:n] = cache.x_pred
        system.vars_to_models()

        # --- Newton-Raphson loop (mirrors ImplicitIter.step) ---
        tds.mis = [1]
        tds.mis_inc = [1]
        tds.niter = 0
        tds.converged = False
        tds.chatter = False

        use_ls = tds.config.linesearch
        if use_ls:
            merit_history = []
            _ls_merit_cache = None

        tds.fg_update(models=system.exist.pflow_tds)

        while True:
            reason = ''
            if dae.t == 0:
                reason = 't=0'
            elif tds.config.honest:
                reason = 'honest'
            elif tds.custom_event:
                reason = 'custom event'
            elif not tds.last_converged:
                reason = 'prev non-convergence'
            elif tds.niter > 4 and (tds.niter + 1) % 3 == 0:
                reason = 'periodic'
            elif dae.t - tds._last_switch_t < 0.1:
                reason = 'near event'

            if reason:
                system.j_update(models=system.exist.pflow_tds, info=reason)
                tds.solver.worker.factorize = True

            if tds.config.g_scale > 0:
                gxs = tds.config.g_scale * h * dae.gx
                gys = tds.config.g_scale * h * dae.gy
            else:
                gxs = dae.gx
                gys = dae.gy

            tds.Ac = sparse([
                [tds.Teye - h * beta0 * dae.fx, gxs],
                [-h * beta0 * dae.fy, gys]
            ], 'd')

            tds.qg[:n] = (dae.Tf * (dae.x[:n] - cache.x_pred + beta0 * cache.phi)
                          - h * beta0 * dae.f)

            for item in system.antiwindups:
                for key, _, eqval in item.x_set:
                    np.put(tds.qg, key, eqval)

            if tds.config.g_scale > 0:
                tds.qg[n:] = tds.config.g_scale * h * dae.g
            else:
                tds.qg[n:] = dae.g

            if not tds.config.linsolve:
                inc = np.array(tds.solver.solve(tds.Ac, matrix(tds.qg))).ravel()
            else:
                inc = np.array(tds.solver.linsolve(tds.Ac, matrix(tds.qg))).ravel()

            if np.isnan(inc).any():
                logger.debug("QNDF: NaN in linear solve at t=%.6f, h=%.4g — treating as Newton failure",
                             dae.t, h)
                tds.err_msg = 'NaN in linear solve'
                break

            inc_abs = np.abs(inc)
            if tds.config.reset_tiny:
                inc[inc_abs < tds.tol_zero] = 0

            tds.inc = inc
            mis_arg = int(inc_abs.argmax())
            mis_inc = inc[mis_arg]
            mis_qg = float(np.max(np.abs(tds.qg)))

            if tds.niter == 0:
                tds.mis[0] = mis_qg
                tds.mis_inc[0] = abs(mis_inc)
            else:
                tds.mis.append(mis_qg)
                tds.mis_inc.append(mis_inc)

            mis = float(inc_abs[mis_arg])

            if tds.niter > tds.config.chatter_iter:
                if abs(sum(tds.mis_inc[-2:])) < 1e-6 and abs(tds.mis_inc[-1]) > 1e-4:
                    tds.chatter = True

            inc_x = inc[:n]
            inc_y = inc[n:n + dae.m]

            if use_ls:
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
                    tds.qg[:n] = (dae.Tf * (dae.x[:n] - cache.x_pred + beta0 * cache.phi)
                                  - h * beta0 * dae.f)
                    for item in system.antiwindups:
                        for key, _, eqval in item.x_set:
                            np.put(tds.qg, key, eqval)
                    if tds.config.g_scale > 0:
                        tds.qg[n:] = tds.config.g_scale * h * dae.g
                    else:
                        tds.qg[n:] = dae.g
                    merit_new = float(np.dot(tds.qg, tds.qg))
                    if merit_new < merit_ref:
                        break
                    alpha *= 0.5
                    logger.debug("QNDF line search backtrack: alpha=%.4g at t=%.6f",
                                 alpha, dae.t)

                _ls_merit_cache = merit_new
            else:
                dae.x -= inc_x
                dae.y -= inc_y
                system.vars_to_models()
                tds.fg_update(models=system.exist.pflow_tds)

            tds.niter += 1

            if mis <= tds.config.tol:
                tds.converged = True
                break
            if tds.chatter:
                tds.converged = True
                break
            if tds.niter > tds.config.max_iter:
                break
            if abs(mis) > 1e6 and abs(mis) > 1e6 * tds.mis[0]:
                tds.err_msg = 'Error diverged'
                break

        # === End Newton loop ===

        if not tds.converged:
            # NEWTON FAILURE — restore, set shrunk deltat, return False
            cache.D[:] = cache.D_prev
            dae.x[:] = tds.x0
            dae.y[:] = tds.y0
            dae.f[:] = tds.f0
            system.vars_to_models()
            cache.consec_fail_count += 1
            tds.deltat = min(h * 0.5, tds.deltatmax)
            tds.last_converged = False

            logger.debug('* QNDF: max iter %d reached for t=%.6f, h=%.6f, max inc=%.4g',
                         tds.config.max_iter, dae.t, h, mis)

            if logger.isEnabledFor(logging.DEBUG):
                g_max = np.argmax(abs(dae.g))
                inc_max = np.argmax(abs(inc))
                tds._debug_g(g_max)
                tds._debug_ac(inc_max)

            return False

        # === Error estimation ===
        x_new = dae.x[:n]
        cache.dx_corr[:] = x_new - cache.x_pred

        # Warm-up: accept unconditionally until the D-table has enough
        # history for meaningful error estimates (order+1 accepted steps).
        # With fewer points, D[:,k+1] differences are raw corrections
        # rather than smooth higher-order errors, producing wildly inflated err_est.
        if cache.consec_steps < k + 1:
            update_D(cache.D, cache.dx_corr, k)
            cache.D[:, 0] = x_new
            cache.h_prev = h
            cache.consec_steps += 1
            cache.consec_fail_count = 0
            cache.err_est = 0.0
            tds.deltat = h
            tds.last_converged = True
            return True

        update_D(cache.D, cache.dx_corr, k)

        # err_est at current order k
        err_k = error_constant(k) * cache.D[:, k + 1]
        err_est = weighted_rms_error(err_k, cache.x_prev, x_new,
                                     cache.abstol, cache.reltol, cache.err_wt)
        cache.err_est = err_est

        # err_est at order k-1
        if k > 1 and cache.consec_steps >= k:
            err_km1 = error_constant(k - 1) * cache.D[:, k]
            cache.err_est_km1 = weighted_rms_error(err_km1, cache.x_prev, x_new,
                                                   cache.abstol, cache.reltol, cache.err_wt)
        else:
            cache.err_est_km1 = float('inf')

        # err_est at order k+1
        if k < cache.max_order and cache.consec_steps >= k + 2:
            err_kp1 = error_constant(k + 1) * cache.D[:, k + 2]
            cache.err_est_kp1 = weighted_rms_error(err_kp1, cache.x_prev, x_new,
                                                   cache.abstol, cache.reltol, cache.err_wt)
        else:
            cache.err_est_kp1 = float('inf')

        if err_est <= 1.0:
            # ACCEPT
            cache.D[:, 0] = x_new
            order_new, h_new = select_order_and_step(cache, h, k, err_est)

            if order_new == k:
                cache.consec_steps += 1
            else:
                cache.consec_steps = 0
            cache.order_prev = k
            cache.order = order_new
            cache.h_prev = h
            cache.consec_fail_count = 0
            tds.deltat = min(h_new, tds.deltatmax)
            tds.last_converged = True
            return True
        else:
            # LTE REJECT — restore state, set shrunk deltat, return False
            cache.D[:] = cache.D_prev
            dae.x[:] = tds.x0
            dae.y[:] = tds.y0
            dae.f[:] = tds.f0
            system.vars_to_models()

            _, h_next, fail_count_next = accept_reject(
                err_est=err_est,
                h=h,
                deltatmax=tds.deltatmax,
                order=k,
                fail_count=cache.consec_fail_count,
                accept_threshold=1.0,
                # acceptance not used in this branch, keep same policy shape.
                accept_safety=(1.0 / 1.2),
                accept_min_factor=0.0,
                accept_max_factor=10.0,
                reject_safety=(1.0 / 1.2),
                reject_min_factor=0.2,
                reject_max_factor=1.0,
                repeat_reject_after=1,
                repeat_reject_factor=0.5,
            )
            tds.deltat = h_next
            cache.consec_fail_count = fail_count_next

            if cache.consec_fail_count > 2 and k > 1:
                cache.order = k - 1

            # After repeated LTE rejections, the D-table checkpoint becomes
            # unreliable (rescaling stale data produces poor predictors and
            # inflated error estimates). Reset to break the pathological cycle.
            if cache.consec_fail_count >= 3:
                cache.D[:] = 0.0
                cache.D[:, 0] = tds.x0[:n]
                cache.consec_steps = 0
                cache.order = 1
                cache.h_prev = 0.0

            logger.debug("QNDF LTE reject at t=%.6f: err_est=%.3g, h=%.4g, next deltat=%.4g",
                         dae.t, err_est, h, tds.deltat)
            tds.converged = False
            tds.last_converged = False
            return False
