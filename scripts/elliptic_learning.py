"""
spectral_method_no_mask.py  (per-view operator rewrite)
========================================================

Key change from the previous version
--------------------------------------
Previously:  one consensus elliptic operator learned from view 0 (SS-S_A)
             only; all four views projected through that single eigenbasis.

Now:         each of the four PWI views (SS-S_A, SS-L_A, SS-S_B, SS-L_B)
             gets its own consensus operator and its own eigenbasis learned
             independently in Stage 1/2.  Stage 3 projects each view through
             its own physics-matched basis and concatenates the coefficients.
             The 4×N_modes = 256-dimensional descriptor fed to Stage 4 is
             identical in shape to before, so Stage 4 is unchanged.

Rationale
---------
SS-S and SS-L views arise from different wave-mode combinations; their
spatial structures are governed by different effective elliptic operators
(different anisotropy, different dominant scale).  Learning the basis for
each view from its own population removes cross-modal basis mismatch and
should give a more faithful low-dimensional representation of each view's
information content.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path('src')))

from helpers import (
    BSplineBasis2D,
    LearnedOperator,
    SpectralDecomposition,
    SurfaceData,
    bspline_basis_1d,
    bspline_deriv_1d,
    ellipticity_penalty,
    smoothness_l2,
    normalization_loss,
)

import jax
import jax.numpy as jnp
from jax import random
import optax

from baselines import CrackCharacterizer

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
THICKNESS_MM = 10.0
STANDOFF_MM  = 20.0
ARRAY_A_X_MM = -23.0
ARRAY_B_X_MM = +23.0
IMG_X_MIN_MM = 13.0
IMG_X_MAX_MM = 21.0
IMG_Z_MIN_MM = 0.0
IMG_Z_MAX_MM = 12.0
N_PIX        = 32
VIEW_NAMES   = ["SS-S_A", "SS-L_A", "SS-S_B", "SS-L_B"]
N_VIEWS      = 4
FIELD_KEYS   = ("a11", "a12", "a22", "b1", "b2")


# ---------------------------------------------------------------------------
# Lightweight descriptor-manifold diagnostics
# ---------------------------------------------------------------------------

def _safe_participation_ratio_from_singular_values(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    s2 = s ** 2
    denom = float(np.sum(s2 ** 2))
    if denom <= 1e-30:
        return float("nan")
    return float((np.sum(s2) ** 2) / denom)


def _modes_for_energy_fraction_from_singular_values(s: np.ndarray, frac: float = 0.95) -> float:
    s = np.asarray(s, dtype=np.float64)
    energy = s ** 2
    total = float(np.sum(energy))
    if total <= 1e-30:
        return float("nan")
    cum = np.cumsum(energy) / total
    return float(np.searchsorted(cum, frac, side="left") + 1)


def _descriptor_basis_from_data(X: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] == 0 or Xc.shape[1] == 0:
        return np.zeros((Xc.shape[1], 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    r = int(max(0, min(rank, vt.shape[0], vt.shape[1])))
    if r == 0:
        return np.zeros((Xc.shape[1], 0), dtype=np.float64), s
    return vt[:r].T.copy(), s


def _grassmann_alignment_summary(X_a: np.ndarray, X_b: np.ndarray, rank: int) -> Dict[str, float]:
    basis_a, s_a = _descriptor_basis_from_data(X_a, rank)
    basis_b, s_b = _descriptor_basis_from_data(X_b, rank)
    r = int(min(basis_a.shape[1], basis_b.shape[1]))
    out = {
        "rank": float(r),
        "alignment": float("nan"),
        "grassmann_dist": float("nan"),
        "mean_principal_angle": float("nan"),
        "max_principal_angle": float("nan"),
        "descriptor_pr_a": _safe_participation_ratio_from_singular_values(s_a),
        "descriptor_pr_b": _safe_participation_ratio_from_singular_values(s_b),
        "modes_for_95pct_a": _modes_for_energy_fraction_from_singular_values(s_a, 0.95),
        "modes_for_95pct_b": _modes_for_energy_fraction_from_singular_values(s_b, 0.95),
    }
    if r <= 0:
        return out

    M = basis_a[:, :r].T @ basis_b[:, :r]
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    theta = np.arccos(sigma)
    out.update({
        "alignment": float(np.sum(sigma ** 2) / r),
        "grassmann_dist": float(np.linalg.norm(theta)),
        "mean_principal_angle": float(np.mean(theta)),
        "max_principal_angle": float(np.max(theta)),
    })
    return out


def _subsample_rows(X: Optional[np.ndarray], max_rows: int, seed: int) -> Optional[np.ndarray]:
    if X is None:
        return None
    X = np.asarray(X)
    if X.shape[0] <= max_rows or max_rows <= 0:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=max_rows, replace=False))
    return X[idx]


# ===========================================================================
# STAGE 1 CONFIG & PDE RESIDUAL
# ===========================================================================

@dataclass
class PDELearnerConfig:
    """Hyper-parameters for Stage 1 (one instance per view)."""
    n_ctrl_u:      int   = 8
    n_ctrl_v:      int   = 8
    degree:        int   = 3
    n_coll:        int   = 20
    lam_data:      float = 1.0 # R&D parameter not used for final results
    lam_pde:       float = 1.0
    lam_consensus: float = 2.0
    lam_elliptic:  float = 0.5
    lam_norm:      float = 1.0
    lam_smooth_f:  float = 1e-2
    lam_smooth_op: float = 5e-3
    lr:              float = 5e-3
    steps:           int   = 3000
    log_every:       int   = 500
    seed:            int   = 0
    sigma:           float = 5.0   # Rényi base bandwidth; scaled per-field by metric
    n_metric_update: int   = 50    # steps between metric/sigma recomputation


def _compute_pde_residual_homogeneous(u, v, f_ctrl, op_params, basis):
    d = basis.derivs(u, v, f_ctrl)

    g_a11 = basis.eval_with_grad(u, v, op_params["a11"])
    g_a12 = basis.eval_with_grad(u, v, op_params["a12"])
    g_a22 = basis.eval_with_grad(u, v, op_params["a22"])

    a11, a12, a22 = g_a11["f"], g_a12["f"], g_a22["f"]

    diffusion_2nd = a11 * d["fuu"] + 2 * a12 * d["fuv"] + a22 * d["fvv"]
    divA_1 = g_a11["fu"] + g_a12["fv"]
    divA_2 = g_a12["fu"] + g_a22["fv"]
    diffusion_1st = divA_1 * d["fu"] + divA_2 * d["fv"]

    b1 = basis.eval(u, v, op_params["b1"])
    b2 = basis.eval(u, v, op_params["b2"])
    advection = b1 * d["fu"] + b2 * d["fv"]

    return diffusion_2nd + diffusion_1st + advection


# ---------------------------------------------------------------------------
# Induced-metric helpers — used during training for sigma scaling + diagnostics
# ---------------------------------------------------------------------------

def _build_L_alpha_matrices(
    basis: "BSplineBasis2D",
    u_np: np.ndarray,   # (M,) collocation u coords
    v_np: np.ndarray,   # (M,) collocation v coords
) -> List[np.ndarray]:
    """
    Build the 5 constant (M, m) derivative collocation matrices L_α = ∂L/∂ā_α.

    The PDE operator acting on image c is:
        L(θ)c = a11·fuu + 2·a12·fuv + a22·fvv + b1·fu + b2·fv

    Treating ā_α as the scalar mean of each operator field:
        L_α = [B_uu, 2·B_uv, B_vv, B_u, B_v]   (one matrix per field)

    These are constant matrices (independent of the current θ or c),
    built once before training.
    """
    u_j = jnp.array(u_np); v_j = jnp.array(v_np)
    Bu   = np.array(bspline_basis_1d(u_j, basis.knots_u, basis.degree))
    Bv   = np.array(bspline_basis_1d(v_j, basis.knots_v, basis.degree))
    dBu  = np.array(bspline_deriv_1d(u_j, basis.knots_u, basis.degree, 1))
    dBv  = np.array(bspline_deriv_1d(v_j, basis.knots_v, basis.degree, 1))
    d2Bu = np.array(bspline_deriv_1d(u_j, basis.knots_u, basis.degree, 2))
    d2Bv = np.array(bspline_deriv_1d(v_j, basis.knots_v, basis.degree, 2))

    M  = Bu.shape[0]
    m  = basis.n_ctrl_u * basis.n_ctrl_v

    def tp(A, B):
        return (A[:, :, None] * B[:, None, :]).reshape(M, m)

    return [
        tp(d2Bu, Bv),        # L_a11: fuu  (factor 1)
        2.0 * tp(dBu, dBv),  # L_a12: 2·fuv (factor 2 from PDE)
        tp(Bu, d2Bv),        # L_a22: fvv  (factor 1)
        tp(dBu, Bv),         # L_b1:  fu
        tp(Bu, dBv),         # L_b2:  fv
    ]

def _op_batch_to_list(op_batch: Dict[str, jnp.ndarray]) -> List[Dict[str, jnp.ndarray]]:
    K = int(op_batch["a11"].shape[0])
    return [{key: op_batch[key][k] for key in FIELD_KEYS} for k in range(K)]


def _to_numpy_fC_list(fC_data) -> List[np.ndarray]:
    arr = np.asarray(fC_data)
    if arr.ndim == 2:
        return [arr]
    return [arr[k] for k in range(arr.shape[0])]


def _to_numpy_op_list(op_data) -> List[Dict[str, np.ndarray]]:
    if isinstance(op_data, dict):
        K = int(np.asarray(op_data["a11"]).shape[0])
        return [{key: np.asarray(op_data[key][k]) for key in FIELD_KEYS} for k in range(K)]
    return [{key: np.asarray(op[key]) for key in FIELD_KEYS} for op in op_data]


def _compute_ghat_and_indicators(
    D_list:   List[np.ndarray],
    fC_list,
    op_list,
    lam_pde:  float,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute per-field metric diagonal ĝ_αα, rho_eff, and rho_comp_approx.

    Accepts either batched arrays/dicts or the older list-based structure.
    """
    fC_np_list = _to_numpy_fC_list(fC_list)
    op_np_list = _to_numpy_op_list(op_list)

    K  = len(fC_np_list)
    M  = D_list[0].shape[0]
    m  = D_list[0].shape[1]
    n_alpha = len(FIELD_KEYS)

    t_list = [np.asarray(fC, dtype=np.float64).ravel() for fC in fC_np_list]

    g_hat = np.zeros(n_alpha, dtype=np.float64)
    for a, D in enumerate(D_list):
        Dt = np.stack([D @ t for t in t_list], axis=0)
        sq = float(np.sum(Dt * Dt))
        g_hat[a] = 2.0 * lam_pde * sq / (K * M)

    tr_g  = float(g_hat.sum())
    tr_g2 = float((g_hat ** 2).sum())
    rho_eff = float(tr_g ** 2 / tr_g2) if tr_g2 > 1e-30 else 1.0

    op_means = np.array([
        float(np.mean([np.asarray(op[key]).mean() for op in op_np_list]))
        for key in FIELD_KEYS
    ], dtype=np.float64)

    L = sum(op_means[a] * D_list[a] for a in range(n_alpha))
    LtL = L.T @ L + 1e-10 * np.eye(m)
    try:
        LtL_inv = np.linalg.inv(LtL)
    except np.linalg.LinAlgError:
        LtL_inv = np.linalg.pinv(LtL)

    num = denom = 0.0
    for D in D_list:
        Dt = np.stack([D @ t for t in t_list], axis=0)
        Pt = (L @ (LtL_inv @ (L.T @ Dt.T))).T
        num += float(np.sum(Pt * Pt))
        denom += float(np.sum(Dt * Dt))

    rho_comp = float(num / denom) if denom > 1e-30 else 0.0
    return g_hat, rho_eff, rho_comp


def _metric_scaled_renyi_consensus(
    op_batch: Dict[str, jnp.ndarray],
    sigma_per_field: "jnp.ndarray",
) -> "jnp.ndarray":
    """Metric-scaled Rényi consensus for a batched operator pytree."""
    total = jnp.array(0.0, dtype=jnp.float32)
    for alpha, key_ in enumerate(FIELD_KEYS):
        sigma_a = jnp.maximum(sigma_per_field[alpha], 1e-8)
        stacked = op_batch[key_].reshape(op_batch[key_].shape[0], -1)
        diff    = stacked[:, None, :] - stacked[None, :, :]
        sq      = jnp.sum(diff ** 2, axis=-1)
        K_      = stacked.shape[0]
        log_pot = jax.scipy.special.logsumexp(
            -sq / (2.0 * sigma_a ** 2)
        ) - jnp.log(jnp.asarray(K_ * K_, dtype=stacked.dtype))
        total = total + (-log_pot)
    return total


def _normalization_loss_batched(op_batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compatibility wrapper around the original normalization penalty."""
    return normalization_loss(_op_batch_to_list(op_batch))


# ===========================================================================
# STAGE 1 — PDE OPERATOR LEARNER  (one instance per view)
# ===========================================================================

class HomogeneousPDELearner:
    """
    Stage 1 for ONE view: learn a consensus elliptic operator from a set of
    single-channel image surfaces with no instance-specific forcing.
    """

    def __init__(self, cfg: Optional[PDELearnerConfig] = None, view_idx: int = 0):
        self.cfg      = cfg or PDELearnerConfig()
        self.view_idx = view_idx
        self.basis    = BSplineBasis2D(
            self.cfg.n_ctrl_u, self.cfg.n_ctrl_v, self.cfg.degree
        )
        self._consensus_op: Optional[LearnedOperator] = None
        self._params:       Optional[Dict]            = None
        self._history:      List[Dict]                = []

        c = self.cfg.n_coll
        u_lin = jnp.linspace(0.02, 0.98, c)
        v_lin = jnp.linspace(0.02, 0.98, c)
        uu, vv = jnp.meshgrid(u_lin, v_lin, indexing="ij")
        self._u_coll = uu.ravel()
        self._v_coll = vv.ravel()

    # ------------------------------------------------------------------
    def images_to_surface_data(
        self,
        images:      np.ndarray,            # (N, 4, 32, 32)
        labels:      Optional[np.ndarray],  # (N, 2)
        P_mm_arr:    Optional[np.ndarray],
        max_samples: int = 400,
    ) -> Tuple[List[SurfaceData], np.ndarray]:
        """
        Extract surfaces from `self.view_idx` channel, subsample to
        `max_samples`, return surface list and chosen indices.
        """
        N   = images.shape[0]
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.choice(N, size=min(max_samples, N), replace=False)
        idx = np.sort(idx)

        rows, cols = np.meshgrid(
            np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij"
        )
        u_base = jnp.array(cols.ravel(), dtype=jnp.float32)
        v_base = jnp.array(rows.ravel(), dtype=jnp.float32)

        surface_list = []
        for i in idx:
            img = images[i, self.view_idx]   # (32, 32)
            L   = float(labels[i, 0])   if labels   is not None else 2.5
            th  = float(labels[i, 1])   if labels   is not None else 0.0
            P   = float(P_mm_arr[i])    if P_mm_arr is not None else 17.0
            sd  = SurfaceData(
                u=u_base, v=v_base,
                y=jnp.array(img.ravel(), dtype=jnp.float32),
                w=jnp.ones(N_PIX * N_PIX, dtype=jnp.float32),
                input_field=None,
                metadata={"L": L, "theta": th, "P": P, "view": self.view_idx},
            )
            surface_list.append(sd)

        return surface_list, idx

    # ------------------------------------------------------------------
    def train(
        self,
        surface_list: List[SurfaceData],
        verbose: bool = True,
    ) -> "HomogeneousPDELearner":
        """Coupled homogeneous PDE learning over `surface_list`."""
        cfg = self.cfg
        K   = len(surface_list)
        vname = VIEW_NAMES[self.view_idx]

        key  = random.PRNGKey(cfg.seed)
        keys = random.split(key, K + 1)

        fC_init = jnp.stack([
            random.normal(keys[k + 1], (cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.01
            for k in range(K)
        ], axis=0)
        op_init = {
            "a11": jnp.stack([
                jnp.ones((cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.5
                + random.normal(keys[k + 1], (cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.01
                for k in range(K)
            ], axis=0),
            "a22": jnp.stack([
                jnp.ones((cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.5
                + random.normal(keys[k + 1], (cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.01
                for k in range(K)
            ], axis=0),
            "a12": jnp.stack([
                random.normal(keys[k + 1], (cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.01
                for k in range(K)
            ], axis=0),
            "b1": jnp.stack([
                random.normal(keys[k + 1], (cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.01
                for k in range(K)
            ], axis=0),
            "b2": jnp.stack([
                random.normal(keys[k + 1], (cfg.n_ctrl_u, cfg.n_ctrl_v)) * 0.01
                for k in range(K)
            ], axis=0),
        }

        params = {"fC": fC_init, "op": op_init}
        coll_mask = jnp.ones_like(self._u_coll, dtype=jnp.float32)
        opt       = optax.adam(cfg.lr)
        opt_state = opt.init(params)

        u_data = surface_list[0].u
        v_data = surface_list[0].v
        y_data = jnp.stack([ds.y for ds in surface_list], axis=0)
        w_data = jnp.stack([ds.w for ds in surface_list], axis=0)

        D_list_np = _build_L_alpha_matrices(
            self.basis,
            np.array(self._u_coll),
            np.array(self._v_coll),
        )
        sigma_per_field = jnp.ones(5, dtype=jnp.float32) * cfg.sigma
        _rho_eff  = float("nan")
        _rho_comp = float("nan")

        def _per_sample_losses(fC, op, y, w):
            f_hat = self.basis.eval(u_data, v_data, fC)
            data_term = jnp.mean(w * (f_hat - y) ** 2)

            r = _compute_pde_residual_homogeneous(
                self._u_coll, self._v_coll, fC, op, self.basis
            )
            pde_term = jnp.mean(coll_mask * r ** 2)

            smooth_f_term = smoothness_l2(fC)
            smooth_op_term = sum(smoothness_l2(op[key_]) for key_ in FIELD_KEYS)
            elliptic_term = ellipticity_penalty(
                op, self.basis, self._u_coll, self._v_coll
            )
            return data_term, pde_term, smooth_f_term, smooth_op_term, elliptic_term

        per_sample_vmap = jax.vmap(_per_sample_losses, in_axes=(0, 0, 0, 0))

        def loss_fn(params, sigma_pf):
            data_terms, pde_terms, smooth_f_terms, smooth_op_terms, elliptic_terms = per_sample_vmap(
                params["fC"], params["op"], y_data, w_data
            )

            data_loss = jnp.mean(data_terms)
            pde_loss = jnp.mean(pde_terms)
            smooth_f = jnp.mean(smooth_f_terms)
            smooth_op = jnp.mean(smooth_op_terms)
            elliptic_loss = jnp.mean(elliptic_terms)

            cons_loss = _metric_scaled_renyi_consensus(params["op"], sigma_pf)
            norm_loss = _normalization_loss_batched(params["op"])

            total = (
                cfg.lam_data        * data_loss
                + cfg.lam_pde       * pde_loss
                + cfg.lam_consensus * cons_loss
                + cfg.lam_norm      * norm_loss
                + cfg.lam_elliptic  * elliptic_loss
                + cfg.lam_smooth_f  * smooth_f
                + cfg.lam_smooth_op * smooth_op
            )
            aux = dict(
                data_loss=data_loss,
                pde_loss=pde_loss,
                consensus_loss=cons_loss,
                elliptic_loss=elliptic_loss,
            )
            return total, aux

        @jax.jit
        def step(params, opt_state, sigma_pf):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, sigma_pf)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, aux

        self._history = []
        t0 = time.time()
        for it in range(cfg.steps + 1):
            params, opt_state, loss, aux = step(params, opt_state, sigma_per_field)

            if it > 0 and it % cfg.n_metric_update == 0:
                try:
                    g_hat, _rho_eff, _rho_comp = _compute_ghat_and_indicators(
                        D_list_np, np.array(params["fC"]), {k: np.array(v) for k, v in params["op"].items()}, cfg.lam_pde
                    )
                    sigma_per_field = jnp.array([
                        cfg.sigma / float(np.sqrt(max(float(g), 1e-8)))
                        for g in g_hat
                    ], dtype=jnp.float32)
                except Exception:
                    pass

            if it % cfg.log_every == 0:
                g_hat, _rho_eff, _rho_comp = _compute_ghat_and_indicators(
                    D_list_np, np.array(params["fC"]), {k: np.array(v) for k, v in params["op"].items()}, cfg.lam_pde
                )
                rec = {
                    "iter": it,
                    "loss": float(loss),
                    "rho_eff": _rho_eff,
                    "rho_comp": _rho_comp,
                    **{k: float(v) for k, v in aux.items()},
                }
                self._history.append(rec)
                if verbose:
                    print(
                        f"  [{vname} Stage1 {it:5d}/{cfg.steps}] "
                        f"L={loss:.4e}  data={float(aux['data_loss']):.4e}  "
                        f"pde={float(aux['pde_loss']):.4e}  "
                        f"cons={float(aux['consensus_loss']):.4e}  "
                        f"ρ_eff={_rho_eff:.3f}  ρ_comp={_rho_comp:.3f}  "
                        f"({time.time()-t0:.0f}s)"
                    )

        self._params = params

        mean_op = {key_: jnp.mean(params["op"][key_], axis=0) for key_ in FIELD_KEYS}
        self._consensus_op = LearnedOperator.from_dict(mean_op)

        if verbose:
            tr = float(jnp.mean(mean_op["a11"]) + jnp.mean(mean_op["a22"]))
            print(f"  [{vname}] Stage 1 complete. Operator trace = {tr:.4f}")

        return self

    @property
    def consensus_operator(self) -> LearnedOperator:
        if self._consensus_op is None:
            raise RuntimeError("Call train() first.")
        return self._consensus_op


# ===========================================================================
# STAGE 2 — EIGENDECOMPOSITION  (one instance per view)
# ===========================================================================

@dataclass
class EigenConfig:
    n_modes:         int   = 64
    nq:              int   = 40
    is_self_adjoint: bool  = True


class MaskedEigendecomposition:
    """Stage 2: eigenbasis for one view's consensus operator."""

    def __init__(self, cfg: Optional[EigenConfig] = None):
        self.cfg    = cfg or EigenConfig()
        self.decomp: Optional[SpectralDecomposition] = None

    def compute(
        self,
        operator: LearnedOperator,
        basis:    BSplineBasis2D,
        verbose:  bool = True,
        view_name: str = "",
    ) -> SpectralDecomposition:
        cfg = self.cfg
        nq  = cfg.nq

        xq = jnp.linspace(0.02, 0.98, nq)
        Uq, Vq = jnp.meshgrid(xq, xq, indexing="ij")
        uq, vq = Uq.ravel(), Vq.ravel()
        h      = float(xq[1] - xq[0])

        wq  = jnp.ones(uq.shape[0]) * h ** 2
        Bu  = bspline_basis_1d(uq, basis.knots_u, basis.degree)
        Bv  = bspline_basis_1d(vq, basis.knots_v, basis.degree)
        dBu = bspline_deriv_1d(uq, basis.knots_u, basis.degree, 1)
        dBv = bspline_deriv_1d(vq, basis.knots_v, basis.degree, 1)

        B   = (Bu[:, :, None] * Bv[:, None, :]).reshape(len(uq), -1)
        B_u = (dBu[:, :, None] * Bv[:, None, :]).reshape(len(uq), -1)
        B_v = (Bu[:, :, None] * dBv[:, None, :]).reshape(len(uq), -1)

        a11 = basis.eval(uq, vq, operator.a11)
        a12 = basis.eval(uq, vq, operator.a12)
        a22 = basis.eval(uq, vq, operator.a22)
        b1  = basis.eval(uq, vq, operator.b1)
        b2  = basis.eval(uq, vq, operator.b2)

        K_diff = (
            (B_u.T * (a11 * wq)) @ B_u
            + (B_u.T * (a12 * wq)) @ B_v
            + (B_v.T * (a12 * wq)) @ B_u
            + (B_v.T * (a22 * wq)) @ B_v
        )
        K_adv  = (B.T * (b1 * wq)) @ B_u + (B.T * (b2 * wq)) @ B_v
        M_mat  = (B.T * wq) @ B
        M_mat  = 0.5 * (M_mat + M_mat.T) + 1e-10 * jnp.eye(M_mat.shape[0])

        if cfg.is_self_adjoint:
            K = K_diff + 0.5 * (K_adv + K_adv.T)
            K = 0.5 * (K + K.T)
        else:
            K = K_diff + K_adv

        L_chol = jnp.linalg.cholesky(M_mat)
        L_inv  = jnp.linalg.inv(L_chol)
        A_sym  = L_inv @ K @ L_inv.T
        A_sym  = 0.5 * (A_sym + A_sym.T)

        lam, y = jnp.linalg.eigh(A_sym)
        C      = L_inv.T @ y

        n_modes     = min(cfg.n_modes, C.shape[1])
        eigenvalues = lam[:n_modes]
        eigenvectors = [
            C[:, k].reshape(basis.n_ctrl_u, basis.n_ctrl_v)
            for k in range(n_modes)
        ]

        self.decomp = SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            mass_matrix=M_mat,
            operator=operator,
            basis=basis,
        )

        if verbose:
            tag = f"[{view_name}] " if view_name else ""
            print(f"  {tag}Eigenvalues λ[0..4] = "
                  f"{[float(lam[i]) for i in range(min(5, n_modes))]}")

        return self.decomp


# ===========================================================================
# STAGE 3 — PER-VIEW SPECTRAL PROJECTION
# ===========================================================================

class PerViewSpectralProjector:
    """
    Stage 3: project each of the four PWI views through its own eigenbasis.

    For image k:
      d_k = [d^{0}_k || d^{1}_k || d^{2}_k || d^{3}_k]  ∈ ℝ^{4*n_modes}
    where
      d^{v}_k = [<img_k^v, φ_n^v>_{M^v}]_{n=1}^{N_modes}

    and (φ_n^v, M^v) are the eigenbasis and mass matrix learned for view v.
    """

    def __init__(
        self,
        decomps:    List[SpectralDecomposition],  # one per view, len==4
        bases:      List[BSplineBasis2D],          # one per view, len==4
        n_modes:    int   = 64,
        lam_smooth: float = 1e-3,
    ):
        assert len(decomps) == N_VIEWS, f"Need {N_VIEWS} decompositions, got {len(decomps)}"
        assert len(bases)   == N_VIEWS

        self.decomps    = decomps
        self.bases      = bases
        self.n_modes    = n_modes
        self.lam_smooth = lam_smooth

        rows, cols = np.meshgrid(
            np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij"
        )
        self._u_grid = jnp.array(cols.ravel(), dtype=jnp.float32)
        self._v_grid = jnp.array(rows.ravel(), dtype=jnp.float32)
        self._project_view_batch_fns = [self._build_project_view_batch_fn(v) for v in range(N_VIEWS)]

    def _project_view_single_jax(self, view_idx: int, y_flat: jnp.ndarray) -> jnp.ndarray:
        C = self.bases[view_idx].fit_surface(
            self._u_grid, self._v_grid, y_flat,
            lam_smooth=self.lam_smooth,
        )
        d_v = self.decomps[view_idx].project(C, self.n_modes)
        return jnp.real(jnp.asarray(d_v, dtype=jnp.float32))

    def _build_project_view_batch_fn(self, view_idx: int):
        def _batch_fn(y_batch: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda y: self._project_view_single_jax(view_idx, y), in_axes=0)(y_batch)
        try:
            return jax.jit(_batch_fn)
        except Exception:
            return _batch_fn

    def project_single(self, image_4ch: np.ndarray) -> np.ndarray:
        """
        Project one (4, 32, 32) image through the four per-view bases.
        Returns (4 * n_modes,) float32.
        """
        parts: List[np.ndarray] = []
        for v in range(N_VIEWS):
            y = jnp.array(image_4ch[v].ravel(), dtype=jnp.float32)
            d_v = np.array(self._project_view_single_jax(v, y))
            parts.append(d_v)

        return np.concatenate(parts).real.astype(np.float32)

    def project_batch(
        self,
        images:   np.ndarray,              # (N, 4, 32, 32)
        labels:   np.ndarray,              # (N, 2) kept for API compat
        P_mm_arr: Optional[np.ndarray] = None,
        verbose:  bool = True,
        chunk:    int  = 100,
    ) -> np.ndarray:
        """Returns (N, 4*n_modes) float32."""
        N   = images.shape[0]
        dim = N_VIEWS * self.n_modes
        D   = np.zeros((N, dim), dtype=np.float32)

        t0 = time.time()
        fallback_count = 0
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            chunk_imgs = images[start:end]
            out_parts: List[np.ndarray] = []
            for v in range(N_VIEWS):
                y_batch = jnp.asarray(chunk_imgs[:, v].reshape(end - start, -1), dtype=jnp.float32)
                try:
                    d_v = np.asarray(self._project_view_batch_fns[v](y_batch), dtype=np.float32)
                except Exception:
                    fallback_count += 1
                    d_v = np.stack([
                        np.array(self._project_view_single_jax(v, y_batch[i]), dtype=np.float32)
                        for i in range(end - start)
                    ], axis=0)
                out_parts.append(d_v)
            D[start:end] = np.concatenate(out_parts, axis=1)
            if verbose and (end % (5 * chunk) == 0 or end == N):
                extra = f", fallback={fallback_count}" if fallback_count else ""
                print(f"  Projected {end}/{N} images  ({time.time()-t0:.1f}s{extra})")

        return D

    # expose a .decomp property pointing to view-0 decomp for
    # backward-compat with visualise code that uses self.projector.decomp
    @property
    def decomp(self) -> SpectralDecomposition:
        return self.decomps[0]


# ===========================================================================
# STAGE 4 — SPECTRAL CRACK REGRESSOR  (unchanged)
# ===========================================================================

@dataclass
class RegressorConfig:
    mode:       str   = "mlp"
    hidden:     int   = 256
    dropout:    float = 0.1
    lr:         float = 1e-3
    epochs:     int   = 1000
    patience:   int   = 50
    batch_size: int   = 256
    reg_lambda: float = 1e-3
    l2_weight:  float = 1e-4
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:       int   = 42


class _SpectralMLP(nn.Module):
    """ℝ^{4N} → ℝ²  [L̂, θ̂]"""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class SpectralCrackRegressor:
    """Stage 4: d → (L̂, θ̂) in two modes: 'linear' (Tikhonov) or 'mlp'."""

    def __init__(self, cfg: Optional[RegressorConfig] = None):
        self.cfg      = cfg or RegressorConfig()
        self._W:      Optional[np.ndarray] = None
        self._b:      Optional[np.ndarray] = None
        self._mlp:    Optional[_SpectralMLP] = None
        self._scaler_D = None
        self._scaler_Y = None
        self.history:  Dict = {}

    def fit(
        self,
        D_train: np.ndarray, Y_train: np.ndarray,
        D_val:   Optional[np.ndarray] = None,
        Y_val:   Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "SpectralCrackRegressor":
        from sklearn.preprocessing import StandardScaler
        self._scaler_D = StandardScaler().fit(D_train)
        self._scaler_Y = StandardScaler().fit(Y_train)

        D_tr = self._scaler_D.transform(D_train)
        Y_tr = self._scaler_Y.transform(Y_train)

        if self.cfg.mode == "linear":
            self._fit_linear(D_tr, Y_tr, verbose)
        else:
            D_v = self._scaler_D.transform(D_val) if D_val is not None else None
            Y_v = self._scaler_Y.transform(Y_val) if Y_val is not None else None
            self._fit_mlp(D_tr, Y_tr, D_v, Y_v, verbose)

        return self

    def predict(self, D: np.ndarray) -> np.ndarray:
        D_s = self._scaler_D.transform(D)
        if self.cfg.mode == "linear":
            Y_s = D_s @ self._W.T + self._b
        else:
            device = torch.device(self.cfg.device)
            self._mlp.eval()
            with torch.no_grad():
                Dt  = torch.from_numpy(D_s.astype(np.float32)).to(device)
                Y_s = self._mlp(Dt).cpu().numpy()
        return self._scaler_Y.inverse_transform(Y_s).astype(np.float32)

    def _fit_linear(self, D: np.ndarray, Y: np.ndarray, verbose: bool):
        lam  = self.cfg.reg_lambda
        N, p = D.shape
        A    = D.T @ D + lam * np.eye(p)
        W_T  = np.linalg.solve(A, D.T @ Y)
        self._W = W_T.T
        self._b = np.zeros(2, dtype=np.float32)
        if verbose:
            res = float(np.mean((D @ W_T - Y) ** 2))
            print(f"  Linear regression: train MSE = {res:.6f}")

    def _fit_mlp(self, D_tr, Y_tr, D_v, Y_v, verbose):
        torch.manual_seed(self.cfg.seed)
        device = torch.device(self.cfg.device)
        in_dim = D_tr.shape[1]

        self._mlp = _SpectralMLP(in_dim, self.cfg.hidden, self.cfg.dropout).to(device)
        optimizer = optim.Adam(self._mlp.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.l2_weight)
        criterion = nn.MSELoss()

        def to_loader(D, Y, shuffle):
            Dt = torch.from_numpy(D.astype(np.float32))
            Yt = torch.from_numpy(Y.astype(np.float32))
            return DataLoader(TensorDataset(Dt, Yt),
                              batch_size=self.cfg.batch_size, shuffle=shuffle)

        tr_loader  = to_loader(D_tr, Y_tr, shuffle=True)
        val_loader = to_loader(D_v, Y_v, shuffle=False) if D_v is not None else None

        best_val, best_state, wait = float("inf"), None, 0
        train_losses, val_losses   = [], []

        for epoch in range(1, self.cfg.epochs + 1):
            self._mlp.train()
            batch_losses = []
            for Db, Yb in tr_loader:
                Db, Yb = Db.to(device), Yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self._mlp(Db), Yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            tr_loss = float(np.mean(batch_losses))
            train_losses.append(tr_loss)

            if val_loader is not None:
                self._mlp.eval()
                with torch.no_grad():
                    vl = [criterion(self._mlp(Db.to(device)), Yb.to(device)).item()
                          for Db, Yb in val_loader]
                val_loss = float(np.mean(vl))
                val_losses.append(val_loss)

                if val_loss < best_val:
                    best_val  = val_loss
                    best_state = {k: v.clone() for k, v in self._mlp.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.cfg.patience:
                        if verbose:
                            print(f"  [MLP] Early stop at epoch {epoch}")
                        break

            if verbose and epoch % 50 == 0:
                vl_str = f"{val_losses[-1]:.5f}" if val_losses else "—"
                print(f"  [MLP] Ep {epoch:4d} | train={tr_loss:.5f} | val={vl_str}")

        if best_state is not None:
            self._mlp.load_state_dict(best_state)
        self.history = {"train_loss": train_losses, "val_loss": val_losses}


# ===========================================================================
# FULL PIPELINE CONFIG
# ===========================================================================

@dataclass
class LightweightMetricConfig:
    enabled: bool = True
    rank: int = 8
    max_samples: int = 1024
    random_seed: int = 0


@dataclass
class SpectralMethodConfig:
    """Master config.  `view_for_pde` is removed; all views are learned."""
    pde:              PDELearnerConfig   = field(default_factory=PDELearnerConfig)
    eigen:            EigenConfig        = field(default_factory=EigenConfig)
    regressor:        RegressorConfig    = field(default_factory=RegressorConfig)
    light_metrics:    LightweightMetricConfig = field(default_factory=LightweightMetricConfig)
    n_modes:          int   = 64
    n_stage1_samples: int   = 400   # operator-learning pool per view
    lam_smooth_proj:  float = 1e-3


# ===========================================================================
# FULL PIPELINE — SpectralCrackCharacterizer
# ===========================================================================

class SpectralCrackCharacterizer(CrackCharacterizer):
    """
    Per-view spectral operator pipeline.

    Stage 1: learn one consensus operator per view (4 independent operators)
    Stage 2: eigendecompose each operator (4 independent eigenbases)
    Stage 3: project each view through its own eigenbasis, concatenate
    Stage 4: MLP or linear regression on the 4*N_modes descriptor
    """

    def __init__(self, cfg: Optional[SpectralMethodConfig] = None):
        self.cfg         = cfg or SpectralMethodConfig()
        # one PDE learner + eigendecomp per view
        self.pde_learners:   List[HomogeneousPDELearner]       = []
        self.eigen_decomps:  List[MaskedEigendecomposition]    = []
        self.projector:      Optional[PerViewSpectralProjector] = None
        self.regressor:      Optional[SpectralCrackRegressor]   = None
        self._bases:         List[BSplineBasis2D]               = []
        self._representation_metrics: Dict = {}

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train:  np.ndarray,
        Y_train:  np.ndarray,
        X_val:    Optional[np.ndarray] = None,
        Y_val:    Optional[np.ndarray] = None,
        P_train:  Optional[np.ndarray] = None,
        P_val:    Optional[np.ndarray] = None,
        verbose:  bool = True,
        **kwargs,
    ) -> "SpectralCrackCharacterizer":
        cfg = self.cfg

        decomps: List[SpectralDecomposition] = []
        bases:   List[BSplineBasis2D]        = []

        # ---- Stages 1 & 2: once per view --------------------------------
        for v in range(N_VIEWS):
            print(f"\n══ Stage 1 [{VIEW_NAMES[v]}]: Homogeneous PDE Learning ══")
            learner = HomogeneousPDELearner(cfg.pde, view_idx=v)
            self._bases.append(learner.basis)
            bases.append(learner.basis)

            surface_list, _ = learner.images_to_surface_data(
                X_train, Y_train, P_train,
                max_samples=cfg.n_stage1_samples,
            )
            learner.train(surface_list, verbose=verbose)
            self.pde_learners.append(learner)

            print(f"\n══ Stage 2 [{VIEW_NAMES[v]}]: Eigendecomposition ══")
            ed = MaskedEigendecomposition(cfg.eigen)
            decomp = ed.compute(
                learner.consensus_operator, learner.basis,
                verbose=verbose, view_name=VIEW_NAMES[v],
            )
            self.eigen_decomps.append(ed)
            decomps.append(decomp)

        # ---- Stage 3: per-view projection --------------------------------
        print("\n══ Stage 3: Per-View Spectral Projection ══")
        self.projector = PerViewSpectralProjector(
            decomps, bases,
            n_modes=cfg.n_modes,
            lam_smooth=cfg.lam_smooth_proj,
        )

        print(f"  Projecting {len(X_train)} training images …")
        D_train = self.projector.project_batch(X_train, Y_train, P_train, verbose)

        D_val = None
        if X_val is not None:
            print(f"  Projecting {len(X_val)} validation images …")
            D_val = self.projector.project_batch(X_val, Y_val, P_val, verbose)

        self._representation_metrics = self._compute_lightweight_metrics(D_train, D_val)

        # ---- Stage 4: regression head ------------------------------------
        print("\n══ Stage 4: Spectral Regression ══")
        self.regressor = SpectralCrackRegressor(cfg.regressor)
        self.regressor.fit(D_train, Y_train, D_val, Y_val, verbose)

        return self

    # ------------------------------------------------------------------
    def _compute_lightweight_metrics(
        self,
        D_train: np.ndarray,
        D_val: Optional[np.ndarray],
    ) -> Dict:
        cfg = self.cfg.light_metrics
        if not cfg.enabled:
            return {}

        D_train_eval = _subsample_rows(D_train, cfg.max_samples, cfg.random_seed)
        D_val_eval = _subsample_rows(D_val, cfg.max_samples, cfg.random_seed + 1) if D_val is not None else None

        metrics: Dict[str, Dict] = {"global": {}, "per_view": {}}

        global_train_basis, global_train_s = _descriptor_basis_from_data(D_train_eval, cfg.rank)
        metrics["global"]["descriptor_pr_train"] = _safe_participation_ratio_from_singular_values(global_train_s)
        metrics["global"]["modes_for_95pct_train"] = _modes_for_energy_fraction_from_singular_values(global_train_s, 0.95)
        if D_val_eval is not None:
            global_val_basis, global_val_s = _descriptor_basis_from_data(D_val_eval, cfg.rank)
            metrics["global"]["descriptor_pr_val"] = _safe_participation_ratio_from_singular_values(global_val_s)
            metrics["global"]["modes_for_95pct_val"] = _modes_for_energy_fraction_from_singular_values(global_val_s, 0.95)
            metrics["global"]["grassmann"] = _grassmann_alignment_summary(D_train_eval, D_val_eval, cfg.rank)
        else:
            metrics["global"]["descriptor_pr_val"] = float("nan")
            metrics["global"]["modes_for_95pct_val"] = float("nan")
            metrics["global"]["grassmann"] = None

        for v, view_name in enumerate(VIEW_NAMES):
            sl = slice(v * self.cfg.n_modes, (v + 1) * self.cfg.n_modes)
            Dtr_v = D_train_eval[:, sl]
            _, s_tr_v = _descriptor_basis_from_data(Dtr_v, cfg.rank)
            view_metrics = {
                "descriptor_pr_train": _safe_participation_ratio_from_singular_values(s_tr_v),
                "modes_for_95pct_train": _modes_for_energy_fraction_from_singular_values(s_tr_v, 0.95),
            }
            if D_val_eval is not None:
                Dva_v = D_val_eval[:, sl]
                _, s_va_v = _descriptor_basis_from_data(Dva_v, cfg.rank)
                view_metrics["descriptor_pr_val"] = _safe_participation_ratio_from_singular_values(s_va_v)
                view_metrics["modes_for_95pct_val"] = _modes_for_energy_fraction_from_singular_values(s_va_v, 0.95)
                view_metrics["grassmann"] = _grassmann_alignment_summary(Dtr_v, Dva_v, min(cfg.rank, self.cfg.n_modes))
            else:
                view_metrics["descriptor_pr_val"] = float("nan")
                view_metrics["modes_for_95pct_val"] = float("nan")
                view_metrics["grassmann"] = None
            metrics["per_view"][view_name] = view_metrics

        return metrics

    # ------------------------------------------------------------------
    def get_diagnostics_dict(self) -> Dict:
        """
        Return Stage-1 training indicators (rho_eff, rho_comp) per view
        plus cross-view summary, for saving alongside regression results.
        """
        out: Dict = {"per_view": {}}
        rho_effs: List[float] = []
        rho_comps: List[float] = []

        for v, learner in enumerate(self.pde_learners):
            hist = learner._history
            # Final values (last logged entry)
            final_rho_eff  = float("nan")
            final_rho_comp = float("nan")
            if hist:
                final_rho_eff  = hist[-1].get("rho_eff",  float("nan"))
                final_rho_comp = hist[-1].get("rho_comp", float("nan"))

            out["per_view"][VIEW_NAMES[v]] = {
                "rho_eff_final":  final_rho_eff,
                "rho_comp_final": final_rho_comp,
                # Full training trajectory for post-analysis
                "history": [
                    {k: v_ for k, v_ in rec.items()}
                    for rec in hist
                ],
            }
            if not np.isnan(final_rho_eff):
                rho_effs.append(final_rho_eff)
            if not np.isnan(final_rho_comp):
                rho_comps.append(final_rho_comp)

        light = self._representation_metrics or {}
        global_light = light.get("global", {}) if isinstance(light, dict) else {}
        per_view_light = light.get("per_view", {}) if isinstance(light, dict) else {}

        for view_name, view_metrics in per_view_light.items():
            if view_name in out["per_view"]:
                out["per_view"][view_name]["light_metrics"] = view_metrics

        grass = global_light.get("grassmann") if isinstance(global_light, dict) else None
        out["representation"] = light
        out["summary"] = {
            "G_effective_rank_mean": float(np.mean(rho_effs)) if rho_effs else float("nan"),
            "manifold_euclidean_ratio_mean": float(global_light.get("descriptor_pr_train", float("nan"))),
            "reconstruction_error_mean": float("nan"),
            "modes_for_95pct_mean": float(global_light.get("modes_for_95pct_train", float("nan"))),
            "descriptor_pr_train_mean": float(global_light.get("descriptor_pr_train", float("nan"))),
            "descriptor_pr_val_mean": float(global_light.get("descriptor_pr_val", float("nan"))),
            "grassmann_alignment_mean": float(grass.get("alignment", float("nan"))) if isinstance(grass, dict) else float("nan"),
            "grassmann_geodesic_mean": float(grass.get("grassmann_dist", float("nan"))) if isinstance(grass, dict) else float("nan"),
            "mean_principal_angle_mean": float(grass.get("mean_principal_angle", float("nan"))) if isinstance(grass, dict) else float("nan"),
            "max_principal_angle_mean": float(grass.get("max_principal_angle", float("nan"))) if isinstance(grass, dict) else float("nan"),
        }
        return out

    # ------------------------------------------------------------------
    def predict(
        self,
        X:    np.ndarray,
        Y:    Optional[np.ndarray] = None,
        P_mm: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if self.projector is None:
            raise RuntimeError("Call fit() first.")
        D = self.projector.project_batch(X, Y, P_mm, verbose=False)
        return self.regressor.predict(D)

    def predict_with_uncertainty(
        self,
        X, Y=None, P_mm=None,
        n_mc: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.regressor.cfg.mode != "mlp":
            Y_hat = self.predict(X, Y, P_mm)
            return Y_hat, np.zeros_like(Y_hat)

        D   = self.projector.project_batch(X, Y, P_mm, verbose=False)
        D_s = self.regressor._scaler_D.transform(D)
        Dt  = torch.from_numpy(D_s.astype(np.float32))
        device = torch.device(self.regressor.cfg.device)

        self.regressor._mlp.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_mc):
                preds.append(self.regressor._mlp(Dt.to(device)).cpu().numpy())
        preds = np.stack(preds, axis=0)
        mean  = self.regressor._scaler_Y.inverse_transform(preds.mean(0))
        std   = preds.std(0) * self.regressor._scaler_Y.scale_[None, :]
        self.regressor._mlp.eval()
        return mean.astype(np.float32), std.astype(np.float32)

    # ------------------------------------------------------------------
    def _reconstruct_view_from_coeffs(
        self,
        coeffs: np.ndarray,    # (n_modes,) for one view
        view:   int = 0,
    ) -> np.ndarray:
        """Reconstruct (32, 32) image from spectral coeffs of one view."""
        if self.projector is None:
            raise RuntimeError("Call fit() first.")
        decomp = self.projector.decomps[view]
        basis  = self._bases[view]
        n_modes = self.cfg.n_modes

        C_hat = np.zeros_like(np.array(decomp.eigenvectors[0]), dtype=np.float64)
        for n in range(n_modes):
            C_hat += float(coeffs[n]) * np.array(decomp.eigenvectors[n], dtype=np.float64)

        u = np.array(self.projector._u_grid, dtype=np.float32)
        v = np.array(self.projector._v_grid, dtype=np.float32)
        f = np.array(basis.eval(u, v, C_hat), dtype=np.float32).reshape(32, 32)
        return f

    # ------------------------------------------------------------------
    def _feature_influence_per_mode(self, D_single: np.ndarray) -> np.ndarray:
        """
        Gradient-times-input saliency, aggregated to (n_modes,) by summing
        the absolute contributions across the four view blocks.
        """
        if self.regressor is None:
            raise RuntimeError("Call fit() first.")
        n_modes = self.cfg.n_modes
        dim     = N_VIEWS * n_modes
        assert D_single.shape == (dim,)

        d_s = self.regressor._scaler_D.transform(
            D_single[None, :]
        ).astype(np.float32)[0]

        if self.regressor.cfg.mode == "linear":
            W      = self.regressor._W.astype(np.float32)   # (2, dim)
            contrib = np.sum(np.abs(W * d_s[None, :]), axis=0)
        else:
            device = torch.device(self.regressor.cfg.device)
            self.regressor._mlp.eval()
            x   = torch.tensor(d_s[None, :], dtype=torch.float32,
                                device=device, requires_grad=True)
            y   = self.regressor._mlp(x)
            y.sum().backward()
            grad    = x.grad.detach().cpu().numpy()[0]
            contrib = np.abs(grad * d_s)

        infl_mode = np.zeros(n_modes, dtype=np.float32)
        for m in range(n_modes):
            for v in range(N_VIEWS):
                infl_mode[m] += float(contrib[v * n_modes + m])
        return infl_mode

    # ------------------------------------------------------------------
    def visualise_random_test_samples(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        P_test: np.ndarray,
        out_dir: Union[str, Path],
        *,
        n_samples: int = 10,
        seed: int = 0,
        top_k_modes: int = 9,
    ) -> List[int]:
        """Save per-view reconstruction and top-mode plots."""
        if self.projector is None or self.regressor is None:
            raise RuntimeError("Call fit() first.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        N   = X_test.shape[0]
        idx = np.sort(rng.choice(N, size=min(n_samples, N), replace=False))

        X_sel = X_test[idx]
        Y_sel = Y_test[idx]
        P_sel = P_test[idx]
        D_sel = self.projector.project_batch(X_sel, Y_sel, P_sel, verbose=False)
        Y_hat = self.regressor.predict(D_sel)

        n_modes = self.cfg.n_modes

        # Pre-compute eigenfunction grids for each view
        u = np.array(self.projector._u_grid, dtype=np.float32)
        v = np.array(self.projector._v_grid, dtype=np.float32)
        phi_grids: List[List[np.ndarray]] = []
        for vv in range(N_VIEWS):
            grids = []
            for n in range(n_modes):
                Cn  = np.array(self.projector.decomps[vv].eigenvectors[n], dtype=np.float64)
                phi = np.array(self._bases[vv].eval(u, v, Cn),
                               dtype=np.float32).reshape(32, 32)
                grids.append(phi)
            phi_grids.append(grids)

        for j, i in enumerate(idx):
            d       = D_sel[j]   # (4*n_modes,)
            d_views = [d[vv*n_modes:(vv+1)*n_modes] for vv in range(N_VIEWS)]

            recon  = [self._reconstruct_view_from_coeffs(d_views[vv], vv)
                      for vv in range(N_VIEWS)]
            orig   = [X_test[i, vv] for vv in range(N_VIEWS)]
            resid  = [orig[vv] - recon[vv] for vv in range(N_VIEWS)]

            infl_mode = self._feature_influence_per_mode(d)
            top       = np.argsort(-infl_mode)[:top_k_modes]

            # Figure A: original / recon / residual
            fig, axes = plt.subplots(4, 3, figsize=(10, 12))
            for r in range(4):
                axes[r, 0].imshow(orig[r], aspect="equal")
                axes[r, 0].set_title(f"{VIEW_NAMES[r]}: original")
                axes[r, 1].imshow(recon[r], aspect="equal")
                axes[r, 1].set_title(f"{VIEW_NAMES[r]}: recon (own basis)")
                axes[r, 2].imshow(resid[r], aspect="equal")
                axes[r, 2].set_title(f"{VIEW_NAMES[r]}: residual")
                for c in range(3):
                    axes[r, c].axis("off")

            L_true, th_true = float(Y_test[i, 0]), float(Y_test[i, 1])
            L_pred, th_pred = float(Y_hat[j, 0]),  float(Y_hat[j, 1])
            fig.suptitle(
                f"idx={i} | true (L,θ)=({L_true:.3f},{th_true:.3f})  "
                f"pred=({L_pred:.3f},{th_pred:.3f})",
                fontsize=12
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            fig.savefig(out_dir / f"sample_{i:05d}_recon.png", dpi=160)
            plt.close(fig)

            # Figure B: top-k influential eigenfunctions (from view 0 for reference)
            k    = min(top_k_modes, len(top))
            nrow = int(np.ceil(k / 3))
            fig, axes = plt.subplots(nrow, 3, figsize=(9, 3 * nrow))
            axes = np.array(axes).reshape(nrow, 3)
            for t in range(nrow * 3):
                rr, cc = divmod(t, 3)
                ax = axes[rr, cc]
                ax.axis("off")
                if t < k:
                    n = int(top[t])
                    ax.imshow(phi_grids[0][n], aspect="equal")
                    ax.set_title(f"mode {n} | infl={infl_mode[n]:.3e}")

            fig.suptitle(f"idx={i} | top-{k} influential eigenfunctions (view 0)",
                         fontsize=12)
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            fig.savefig(out_dir / f"sample_{i:05d}_top_modes.png", dpi=160)
            plt.close(fig)

        return idx.tolist()
    
