"""B0-cal (plan 10.1a): bias-only dev calibration of frozen conditional A/B/C log-probs.

Objective  f(u,v) = mean_dev[-sum_c q_c log softmax(l + b)_c] + 0.01 ||b||^2,  b = (u, v, -u-v).
SciPy L-BFGS-B in FP64 with analytic gradient, x0=(0,0), maxiter=1000, ftol=1e-15, gtol=1e-10.
Convergence is judged by finite objective/parameters and analytic gradient inf-norm <= 1e-8;
solver status is recorded separately. Failure raises (no silent primary switch, no extra
hyperparameters, no temperature). Fitted on dev only, before any test scoring.
"""

from __future__ import annotations

import numpy as np

from ..constants import CALIBRATION_GTOL_ACCEPT, CALIBRATION_L2

SOLVER_OPTIONS = {"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10}
_J = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])  # db/d(u,v)


class CalibrationError(RuntimeError):
    pass


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def objective_and_grad(params: np.ndarray, logp: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    b = _J @ np.asarray(params, dtype=np.float64)
    ls = _log_softmax(logp + b)
    f = float(np.mean(-(q * ls).sum(axis=1)) + CALIBRATION_L2 * float(b @ b))
    grad_b = (np.exp(ls) - q).mean(axis=0) + 2 * CALIBRATION_L2 * b  # rows of q sum to 1
    return f, _J.T @ grad_b


def fit_bias(dev_cond_logprobs, dev_q) -> dict:
    try:
        import scipy
        from scipy import optimize
    except ImportError as exc:
        raise CalibrationError("SciPy is required for the locked B0-cal solver") from exc
    logp = np.asarray(dev_cond_logprobs, dtype=np.float64)
    q = np.asarray(dev_q, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 3 or q.shape != logp.shape or len(logp) == 0:
        raise CalibrationError("expected [N,3] conditional log-probs and vote distributions")
    if not (np.all(np.isfinite(logp)) and np.all(np.isfinite(q))):
        raise CalibrationError("nonfinite calibration input")
    if np.any(q < 0) or np.any(np.abs(q.sum(axis=1) - 1) > 1e-9):
        raise CalibrationError("vote distributions must be nonnegative and sum to 1")
    if np.any(np.abs(np.log(np.exp(logp).sum(axis=1))) > 1e-9):
        raise CalibrationError("inputs must be normalized conditional log-probs")
    res = optimize.minimize(objective_and_grad, np.zeros(2), args=(logp, q), jac=True,
                            method="L-BFGS-B", options=dict(SOLVER_OPTIONS))
    f, g = objective_and_grad(res.x, logp, q)
    grad_inf = float(np.max(np.abs(g)))
    record = {
        "params_uv": [float(v) for v in res.x], "bias": [float(v) for v in _J @ res.x],
        "objective": f, "grad_inf_norm": grad_inf, "solver_status": int(res.status),
        "solver_message": str(res.message), "solver_success_flag": bool(res.success), "nit": int(res.nit),
        "n_dev": int(len(logp)), "l2": CALIBRATION_L2, "options": dict(SOLVER_OPTIONS),
        "scipy_version": scipy.__version__, "numpy_version": np.__version__,
    }
    converged = np.isfinite(f) and np.all(np.isfinite(res.x)) and grad_inf <= CALIBRATION_GTOL_ACCEPT
    record["converged"] = bool(converged)
    if not converged:
        raise CalibrationError(f"B0-cal not converged (grad_inf={grad_inf:.3e}); stage comparison incomplete: {record}")
    return record


def apply_bias(cond_logprobs, bias) -> np.ndarray:
    """Calibrated conditional log-probs log softmax(l + b). Not native generation."""
    return _log_softmax(np.asarray(cond_logprobs, dtype=np.float64) + np.asarray(bias, dtype=np.float64))


def calibrated_rows(b0_rows, bias) -> list[dict]:
    """Derive B0-cal prediction rows from native B0 rows (conditional space only)."""
    from .normlens import validate_full_logprobs

    out = []
    for r in b0_rows:
        if r.get("seed") is not None:
            raise CalibrationError("B0 rows must be the frozen seed-less (seed=null) run")
        lp, log_mass = validate_full_logprobs(r["full_logprobs"], r["example_id"])
        cond = lp - log_mass
        out.append({"example_id": r["example_id"], "seed": r.get("seed"), "condition": r.get("condition", "image"),
                    "cond_logprobs": apply_bias(cond, bias).tolist(), "calibrated": True})
    return out
