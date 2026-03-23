"""Fit subcommands: Approach 3 (parametric L(N,D)) and Approach 1 (IsoFLOP envelope)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from experiments.chinchilla_lib.config import ALL_ARCHS
from experiments.chinchilla_lib.helpers import (
    _fits_approach1_path, _fits_path, _results_path,
)


def fit(args: argparse.Namespace) -> None:
    """Fit L(N,D) = E + A/N^alpha + B/D^beta per (task, arch). Write fits.json.

    Master equation (Chinchilla Approach 3):
        L(N, D) = E + A / N^alpha + B / D^beta

    Where:
        E     = irreducible loss floor (task difficulty lower bound)
        A, alpha  = model-size scaling coefficient and exponent
        B, beta  = data-size scaling coefficient and exponent

    Optimal compute allocation:
        N*(C) ~ C^(beta/(alpha+beta))   -- optimal model size given budget C
        D*(C) ~ C^(alpha/(alpha+beta))   -- optimal data size given budget C
        Ratio  a = beta/(alpha+beta), b = alpha/(alpha+beta)  (saved as N_exponent, D_exponent)

    Reads: results.json (from collect)
    Writes: fits.json
    """
    from scipy.optimize import curve_fit

    chinchilla_dir = args.chinchilla_dir
    tasks = [t.strip() for t in args.tasks.split(",")]

    for task_id in tasks:
        res_path = _results_path(chinchilla_dir, task_id)
        if not os.path.exists(res_path):
            print(f"[WARN] No results.json for '{task_id}'. Run collect first.", file=sys.stderr)
            continue

        with open(res_path) as f:
            results = json.load(f)

        # best_by_size_d: one clean (N, D, L) point per (arch, size, d_name)
        # -- best LR already selected in collect. This is the proper Chinchilla dataset:
        # each point corresponds to a separately trained model with schedule-optimal LR.
        best_by_size_d = results.get("best_by_size_d", {})

        arch_fits: dict[str, dict] = {}
        for arch in ALL_ARCHS:
            Ns, Ds, Ls = [], [], []
            for cell_key, traj in best_by_size_d.items():
                if traj["arch"] != arch:
                    continue
                pt = traj["terminal"]
                vr = pt.get("violation_rate")
                D_seen = pt.get("D_seen")
                n_p = traj.get("n_params")
                if vr is None or D_seen is None or n_p is None:
                    continue
                Ns.append(n_p)
                Ds.append(D_seen)
                Ls.append(vr)

            if len(Ns) < 6:
                print(f"[{task_id}/{arch}] Insufficient data ({len(Ns)} pts), skipping fit")
                continue

            N_arr = np.array(Ns, dtype=float)
            D_arr = np.array(Ds, dtype=float)
            L_arr = np.array(Ls, dtype=float)

            # Remove NaN/Inf
            valid = np.isfinite(N_arr) & np.isfinite(D_arr) & np.isfinite(L_arr)
            N_arr, D_arr, L_arr = N_arr[valid], D_arr[valid], L_arr[valid]
            if len(N_arr) < 6:
                print(f"[{task_id}/{arch}] Too few finite points, skipping fit")
                continue

            # Normalize to avoid numerical instability
            N_scale = float(np.median(N_arr))
            D_scale = float(np.median(D_arr))

            def _model(ND, E, A, alpha, B, beta):
                N_norm = ND[0] / N_scale
                D_norm = ND[1] / D_scale
                return E + A * np.power(N_norm, -alpha) + B * np.power(D_norm, -beta)

            ND = np.stack([N_arr, D_arr], axis=0)

            def _try_fit(L_fit, bounds, p0):
                popt, _ = curve_fit(
                    _model, ND, L_fit, p0=p0, bounds=bounds,
                    maxfev=100_000, method="trf",
                )
                return popt

            # ── Direct fit: L = E + A/N^alpha + B/D^beta ──────────────────────
            L_floor = max(float(L_arr.min()) * 0.1, 1e-6)
            p0_direct = [L_floor, 1.0, 0.5, 1.0, 0.5]
            bounds_direct = ([0, 1e-6, 0.01, 1e-6, 0.01], [1.0, 1e4, 5.0, 1e4, 5.0])
            fit_direct: dict = {}
            try:
                popt = _try_fit(L_arr, bounds_direct, p0_direct)
                E, A_norm, alpha, B_norm, beta = popt
                A = float(A_norm * (N_scale ** alpha))
                B = float(B_norm * (D_scale ** beta))
                E = float(E)
                L_pred = E + A * np.power(N_arr, -alpha) + B * np.power(D_arr, -beta)
                ss_res = float(np.sum((L_arr - L_pred) ** 2))
                ss_tot = float(np.sum((L_arr - L_arr.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                N_exponent = float(beta / (alpha + beta))
                D_exponent = float(alpha / (alpha + beta))
                fit_direct = {
                    "E": E, "A": A, "alpha": float(alpha),
                    "B": B, "beta": float(beta),
                    "r_squared": r2,
                    "N_exponent": N_exponent,
                    "D_exponent": D_exponent,
                }
                print(
                    f"[{task_id}/{arch}] direct  E={E:.4f}  A={A:.3e}  alpha={alpha:.3f}  "
                    f"B={B:.3e}  beta={beta:.3f}  R2={r2:.3f}  "
                    f"N_exp={N_exponent:.3f}  D_exp={D_exponent:.3f}"
                )
            except Exception as e:
                print(f"[{task_id}/{arch}] Direct fit failed: {e}", file=sys.stderr)

            # ── Logit fit: logit(L) = E' + A'/N^alpha + B'/D^beta ──────────────
            # Applicable when L in (0,1) (e.g. violation_rate).
            # Logit transform linearizes the bounded metric, giving better
            # power-law fits and interpretable E' = logit(irreducible floor).
            fit_logit: dict = {}
            vr_in_range = (L_arr > 0) & (L_arr < 1)
            if vr_in_range.sum() >= 6:
                eps = 1e-6
                L_clip = np.clip(L_arr, eps, 1.0 - eps)
                L_logit = np.log(L_clip / (1.0 - L_clip))
                logit_floor = float(L_logit.min()) - 1.0
                p0_logit = [logit_floor, 1.0, 0.5, 1.0, 0.5]
                # E' in (-inf, +inf): widen lower bound; keep A,B,alpha,beta positive
                bounds_logit = ([-30, 1e-6, 0.01, 1e-6, 0.01], [30, 1e4, 5.0, 1e4, 5.0])
                try:
                    popt_l = _try_fit(L_logit, bounds_logit, p0_logit)
                    E_l, A_l_norm, alpha_l, B_l_norm, beta_l = popt_l
                    A_l = float(A_l_norm * (N_scale ** alpha_l))
                    B_l = float(B_l_norm * (D_scale ** beta_l))
                    E_l = float(E_l)
                    # R2 in logit space
                    L_logit_pred = E_l + A_l * np.power(N_arr, -alpha_l) + B_l * np.power(D_arr, -beta_l)
                    ss_res_l = float(np.sum((L_logit - L_logit_pred) ** 2))
                    ss_tot_l = float(np.sum((L_logit - L_logit.mean()) ** 2))
                    r2_logit = 1.0 - ss_res_l / ss_tot_l if ss_tot_l > 0 else 0.0
                    # R2 in original space (inverse-logit prediction vs raw VR)
                    L_raw_pred = 1.0 / (1.0 + np.exp(-L_logit_pred))
                    ss_res_r = float(np.sum((L_arr - L_raw_pred) ** 2))
                    ss_tot_r = float(np.sum((L_arr - L_arr.mean()) ** 2))
                    r2_raw = 1.0 - ss_res_r / ss_tot_r if ss_tot_r > 0 else 0.0
                    N_exp_l = float(beta_l / (alpha_l + beta_l))
                    D_exp_l = float(alpha_l / (alpha_l + beta_l))
                    fit_logit = {
                        "E": E_l, "A": A_l, "alpha": float(alpha_l),
                        "B": B_l, "beta": float(beta_l),
                        "r_squared_logit": r2_logit,
                        "r_squared_raw": r2_raw,
                        "N_exponent": N_exp_l,
                        "D_exponent": D_exp_l,
                        "transform": "logit",
                    }
                    print(
                        f"[{task_id}/{arch}] logit   E'={E_l:.3f}  A'={A_l:.3e}  alpha={alpha_l:.3f}  "
                        f"B'={B_l:.3e}  beta={beta_l:.3f}  R2_logit={r2_logit:.3f}  R2_raw={r2_raw:.3f}  "
                        f"N_exp={N_exp_l:.3f}  D_exp={D_exp_l:.3f}"
                    )
                except Exception as e:
                    print(f"[{task_id}/{arch}] Logit fit failed: {e}", file=sys.stderr)

            if fit_direct or fit_logit:
                # Prefer logit fit for reporting (better for bounded metrics),
                # fall back to direct if logit failed.
                primary = fit_logit if fit_logit else fit_direct
                arch_fits[arch] = {
                    **primary,
                    "n_points": int(len(N_arr)),
                    "N_scale": N_scale,
                    "D_scale": D_scale,
                    "fit_direct": fit_direct,
                    "fit_logit": fit_logit,
                }

        if arch_fits:
            fits_path = _fits_path(chinchilla_dir, task_id)
            with open(fits_path, "w") as f:
                json.dump({"task": task_id, "fits": arch_fits}, f, indent=2)
            print(f"Fits saved: {fits_path}")


def fit_approach1(args: argparse.Namespace) -> None:
    """Approach 1: IsoFLOP profile minima -> N*(C), D*(C) power-law fit.

    For each arch:
      1. Group terminal (C, N, D, VR) by D-budget level (D1-D4).
      2. For each D level, fit quadratic in log(N) vs log(VR) to find the
         IsoFLOP curve minimum -> N*(D_k), C*(D_k), D*(D_k).
      3. Fit power laws on these IsoFLOP-optimal points:
             N*(C) = a_N * C^n_exp    (OLS in log-log)
             D*(C) = a_D * C^d_exp
      4. Sanity check: |n_exp + d_exp - 1.0| > 0.2 -> flag as unreliable.

    This is the correct Chinchilla Approach 1: optimal allocation is read
    off the minima of IsoFLOP profiles, not from a progressive-min envelope.

    Reads: results.json (from collect)
    Writes: fits_approach1.json
    """
    from scipy.interpolate import PchipInterpolator

    chinchilla_dir = args.chinchilla_dir
    tasks = [t.strip() for t in args.tasks.split(",")]

    for task_id in tasks:
        res_path = _results_path(chinchilla_dir, task_id)
        if not os.path.exists(res_path):
            print(f"[WARN] No results.json for '{task_id}'. Run collect first.", file=sys.stderr)
            continue

        with open(res_path) as f:
            results = json.load(f)

        best_by_size_d = results.get("best_by_size_d", {})

        arch_fits: dict[str, dict] = {}
        for arch in ALL_ARCHS:
            # Collect terminal points, grouped by d_name
            by_d: dict[str, list[dict]] = {}
            for cell_key, traj in best_by_size_d.items():
                if traj["arch"] != arch:
                    continue
                pt = traj["terminal"]
                vr = pt.get("violation_rate")
                D_seen = pt.get("D_seen")
                n_p = traj.get("n_params")
                fps = traj.get("flops_per_step", 0)
                step = pt.get("step", 0)
                d_name = traj.get("d_name", cell_key.split("/")[-1])
                C = fps * step if fps and step else pt.get("total_flops", 0)
                if None in (vr, D_seen, n_p) or C <= 0:
                    continue
                by_d.setdefault(d_name, []).append({
                    "C": float(C), "N": float(n_p), "D": float(D_seen),
                    "VR": float(vr), "d_name": d_name,
                })

            if not by_d:
                continue

            # ── IsoFLOP minima: for each D level, find optimal N ──────────
            opt_pts: list[dict] = []
            for d_name in sorted(by_d.keys()):
                d_pts = sorted(by_d[d_name], key=lambda p: p["N"])
                if len(d_pts) < 2:
                    continue

                Ns = np.array([p["N"] for p in d_pts])
                VRs = np.array([p["VR"] for p in d_pts])
                Cs_d = np.array([p["C"] for p in d_pts])
                Ds_d = np.array([p["D"] for p in d_pts])

                # Fit quadratic in log(N) vs log(VR) to find smooth minimum
                log_N = np.log(Ns)
                log_VR = np.log(np.clip(VRs, 1e-6, None))

                if len(d_pts) >= 3:
                    coeffs = np.polyfit(log_N, log_VR, min(2, len(d_pts) - 1))
                    if len(coeffs) == 3 and coeffs[0] > 0:
                        # Parabola opens up -> minimum exists
                        log_N_star = -coeffs[1] / (2 * coeffs[0])
                        log_N_star = np.clip(log_N_star, log_N.min(), log_N.max())
                        N_star = float(np.exp(log_N_star))
                    else:
                        best_idx = int(np.argmin(VRs))
                        N_star = float(Ns[best_idx])
                else:
                    best_idx = int(np.argmin(VRs))
                    N_star = float(Ns[best_idx])

                # Interpolate C at N_star
                try:
                    interp_C = PchipInterpolator(log_N, np.log(Cs_d))
                    C_star = float(np.exp(interp_C(np.log(N_star))))
                except Exception:
                    best_idx = int(np.argmin(np.abs(Ns - N_star)))
                    C_star = float(Cs_d[best_idx])

                D_star = float(np.median(Ds_d))
                VR_star = float(np.min(VRs))
                regime = (
                    "not_converged"   if VR_star >= 0.99
                    else "chinchilla_valid" if D_star >= 6 * N_star
                    else "data_limited"
                )
                opt_pts.append({
                    "C": C_star, "N": N_star, "D": D_star, "VR": VR_star,
                    "d_name": d_name, "regime": regime,
                })

            # Filter to converged points
            env_valid = [p for p in opt_pts if p["VR"] < 0.99]

            if len(env_valid) < 2:
                print(
                    f"[{task_id}/{arch}] Approach 1: only {len(env_valid)} valid IsoFLOP "
                    f"minima (need >=2), skipping fit",
                    file=sys.stderr,
                )
                arch_fits[arch] = {
                    "fit_available": False,
                    "all_points": opt_pts,
                }
                continue

            Cs  = np.array([p["C"]  for p in env_valid], dtype=float)
            Ns  = np.array([p["N"]  for p in env_valid], dtype=float)
            Ds  = np.array([p["D"]  for p in env_valid], dtype=float)
            VRs = np.array([p["VR"] for p in env_valid], dtype=float)

            # ── Power-law fits in log-log space (OLS) ────────────────────
            log_C = np.log(Cs)
            n_exp, log_a_N = np.polyfit(log_C, np.log(Ns), 1)
            d_exp, log_a_D = np.polyfit(log_C, np.log(Ds), 1)
            a_N = float(np.exp(log_a_N))
            a_D = float(np.exp(log_a_D))

            def _r2_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                log_t = np.log(y_true)
                log_p = np.log(y_pred)
                ss_res = float(np.sum((log_t - log_p) ** 2))
                ss_tot = float(np.sum((log_t - log_t.mean()) ** 2))
                return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            r2_N = _r2_log(Ns, a_N * np.power(Cs, n_exp))
            r2_D = _r2_log(Ds, a_D * np.power(Cs, d_exp))

            exp_sum = float(n_exp + d_exp)
            exp_sum_ok = abs(exp_sum - 1.0) <= 0.2

            regime_counts = {}
            for p in env_valid:
                regime_counts[p["regime"]] = regime_counts.get(p["regime"], 0) + 1

            arch_fits[arch] = {
                "fit_available": True,
                "n_exp": float(n_exp),
                "a_N": a_N,
                "d_exp": float(d_exp),
                "a_D": a_D,
                "r2_N": r2_N,
                "r2_D": r2_D,
                "n_exp_d_exp_sum": exp_sum,
                "n_exp_d_exp_sum_ok": exp_sum_ok,
                "n_envelope_points": len(env_valid),
                "regime_counts": regime_counts,
                "envelope_C":  [p["C"]      for p in env_valid],
                "envelope_N":  [p["N"]      for p in env_valid],
                "envelope_D":  [p["D"]      for p in env_valid],
                "envelope_VR": [p["VR"]     for p in env_valid],
                "envelope_regime": [p["regime"] for p in env_valid],
                "all_points": opt_pts,
            }
            warn = "" if exp_sum_ok else "  !! n+d!=1 (data-limited regime)"
            print(
                f"[{task_id}/{arch}] Approach 1: "
                f"n_exp={n_exp:.3f}  a_N={a_N:.3e}  "
                f"d_exp={d_exp:.3f}  a_D={a_D:.3e}  "
                f"n+d={exp_sum:.2f}  "
                f"R2_N={r2_N:.3f}  R2_D={r2_D:.3f}  "
                f"({len(env_valid)} pts, regimes={regime_counts}){warn}"
            )

        if arch_fits:
            fits_path = _fits_approach1_path(chinchilla_dir, task_id)
            with open(fits_path, "w") as f:
                json.dump({"task": task_id, "fits": arch_fits}, f, indent=2)
            print(f"Approach 1 fits saved: {fits_path}")
