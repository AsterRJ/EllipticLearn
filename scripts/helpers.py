#!/usr/bin/env python3
"""Helper functions and data structures for the coupled surface + operator learning pipeline.
Author: Alastair Poole, University of Strathclyde
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import jax.numpy as jnp

# =============================================================================
# B-SPLINE INFRASTRUCTURE (self-contained)
# =============================================================================

def open_uniform_knots(n_ctrl: int, degree: int) -> jnp.ndarray:
    """Open-uniform knot vector on [0,1]."""
    p = degree
    n_internal = n_ctrl - p - 1
    if n_internal <= 0:
        internal = jnp.array([], dtype=jnp.float32)
    else:
        internal = jnp.linspace(0, 1, n_internal + 2, dtype=jnp.float32)[1:-1]
    return jnp.concatenate([
        jnp.zeros(p + 1, dtype=jnp.float32),
        internal,
        jnp.ones(p + 1, dtype=jnp.float32),
    ])


def bspline_basis_1d(x: jnp.ndarray, knots: jnp.ndarray, degree: int) -> jnp.ndarray:
    """B-spline basis matrix (N, n_ctrl) via Cox-de Boor."""
    t, p = knots, degree
    n_ctrl = t.shape[0] - p - 1
    left, right = t[:-1], t[1:]
    cond = (x[:, None] >= left) & (x[:, None] < right)
    cond = cond.at[:, -1].set(cond[:, -1] | (x == t[-1]))
    B = cond[:, :n_ctrl].astype(jnp.float32)
    for k in range(1, p + 1):
        ti, tik = t[:n_ctrl], t[k:n_ctrl + k]
        ti1, tik1 = t[1:n_ctrl + 1], t[k + 1:n_ctrl + k + 1]
        d1, d2 = tik - ti, tik1 - ti1
        w1 = jnp.where(d1 > 0, (x[:, None] - ti) / d1, 0.0)
        w2 = jnp.where(d2 > 0, (tik1 - x[:, None]) / d2, 0.0)
        B = w1 * B + w2 * jnp.pad(B[:, 1:], ((0, 0), (0, 1)))
    return B


def bspline_deriv_1d(x: jnp.ndarray, knots: jnp.ndarray, degree: int, order: int) -> jnp.ndarray:
    """B-spline derivative basis."""
    if order == 0:
        return bspline_basis_1d(x, knots, degree)
    if degree == 0:
        return jnp.zeros((x.shape[0], knots.shape[0] - 1), dtype=jnp.float32)
    t, p = knots, degree
    n_ctrl = t.shape[0] - p - 1
    Bm1 = bspline_basis_1d(x, t, p - 1)
    ti, tip = t[:n_ctrl], t[p:n_ctrl + p]
    ti1, tip1 = t[1:n_ctrl + 1], t[p + 1:n_ctrl + p + 1]
    dA, dB = tip - ti, tip1 - ti1
    A = jnp.where(dA > 0, p / dA, 0.0)
    Bcoef = jnp.where(dB > 0, p / dB, 0.0)
    dBasis = A * Bm1[:, :n_ctrl] - Bcoef * Bm1[:, 1:n_ctrl + 1]
    if order == 1:
        return dBasis
    if p - 1 == 0:
        return jnp.zeros_like(dBasis)
    dBm1 = bspline_deriv_1d(x, t, p - 1, 1)
    return A * dBm1[:, :n_ctrl] - Bcoef * dBm1[:, 1:n_ctrl + 1]


@dataclass
class BSplineBasis2D:
    """2D tensor-product B-spline basis."""
    n_ctrl_u: int
    n_ctrl_v: int
    degree: int = 3
    
    def __post_init__(self):
        self.knots_u = open_uniform_knots(self.n_ctrl_u, self.degree)
        self.knots_v = open_uniform_knots(self.n_ctrl_v, self.degree)
        self.n_dof = self.n_ctrl_u * self.n_ctrl_v
    
    def eval(self, u: jnp.ndarray, v: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        """Evaluate at scattered points."""
        Bu = bspline_basis_1d(u, self.knots_u, self.degree)
        Bv = bspline_basis_1d(v, self.knots_v, self.degree)
        return jnp.einsum("ni,ij,nj->n", Bu, C, Bv)
    
    def eval_grid(self, u: jnp.ndarray, v: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        """Evaluate on tensor grid."""
        Bu = bspline_basis_1d(u, self.knots_u, self.degree)
        Bv = bspline_basis_1d(v, self.knots_v, self.degree)
        return Bu @ C @ Bv.T
    
    def derivs(self, u: jnp.ndarray, v: jnp.ndarray, C: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute f, fu, fv, fuu, fuv, fvv at scattered points."""
        Bu = bspline_basis_1d(u, self.knots_u, self.degree)
        Bv = bspline_basis_1d(v, self.knots_v, self.degree)
        dBu = bspline_deriv_1d(u, self.knots_u, self.degree, 1)
        dBv = bspline_deriv_1d(v, self.knots_v, self.degree, 1)
        d2Bu = bspline_deriv_1d(u, self.knots_u, self.degree, 2)
        d2Bv = bspline_deriv_1d(v, self.knots_v, self.degree, 2)
        
        return {
            "f": jnp.einsum("ni,ij,nj->n", Bu, C, Bv),
            "fu": jnp.einsum("ni,ij,nj->n", dBu, C, Bv),
            "fv": jnp.einsum("ni,ij,nj->n", Bu, C, dBv),
            "fuu": jnp.einsum("ni,ij,nj->n", d2Bu, C, Bv),
            "fuv": jnp.einsum("ni,ij,nj->n", dBu, C, dBv),
            "fvv": jnp.einsum("ni,ij,nj->n", Bu, C, d2Bv),
        }

    def eval_with_grad(self, u: jnp.ndarray, v: jnp.ndarray, C: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute only f, fu, fv at scattered points (no second derivatives).
        
        Use this instead of derivs() when only the function value and first
        partial derivatives are needed (e.g. for evaluating operator coefficient
        fields and their spatial gradients).  Avoids 3 unnecessary second-
        derivative B-spline evaluations per call.
        """
        Bu = bspline_basis_1d(u, self.knots_u, self.degree)
        Bv = bspline_basis_1d(v, self.knots_v, self.degree)
        dBu = bspline_deriv_1d(u, self.knots_u, self.degree, 1)
        dBv = bspline_deriv_1d(v, self.knots_v, self.degree, 1)
        
        return {
            "f": jnp.einsum("ni,ij,nj->n", Bu, C, Bv),
            "fu": jnp.einsum("ni,ij,nj->n", dBu, C, Bv),
            "fv": jnp.einsum("ni,ij,nj->n", Bu, C, dBv),
        }

    def fit_surface(
        self,
        u: jnp.ndarray,
        v: jnp.ndarray,
        y: jnp.ndarray,
        lam_smooth: float = 1e-4,
    ) -> jnp.ndarray:
        """Fit B-spline surface to scattered data via regularised least squares.
        
        Solves: min ||B·c - y||² + λ·||D²c||²
        where B is the tensor-product basis matrix and D² is the second-
        difference smoothness penalty.
        
        Args:
            u, v: (N,) parameter coordinates
            y: (N,) observed values
            lam_smooth: smoothness regularisation weight
            
        Returns:
            C: (n_ctrl_u, n_ctrl_v) control point array
        """
        Bu = bspline_basis_1d(u, self.knots_u, self.degree)  # (N, nu)
        Bv = bspline_basis_1d(v, self.knots_v, self.degree)  # (N, nv)
        
        # Tensor-product basis: Phi[i, (j*nv + k)] = Bu[i,j] * Bv[i,k]
        Phi = (Bu[:, :, None] * Bv[:, None, :]).reshape(len(u), -1)  # (N, nu*nv)
        
        # Normal equations with Tikhonov regularisation
        n_dof = self.n_ctrl_u * self.n_ctrl_v
        A = Phi.T @ Phi + lam_smooth * jnp.eye(n_dof)
        b = Phi.T @ y
        
        c_flat = jnp.linalg.solve(A, b)
        return c_flat.reshape(self.n_ctrl_u, self.n_ctrl_v)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SurfaceData:
    """Observations of a single input-output pair.
    
    For parametric PDE problems (e.g. Darcy flow):
        - input_field: the varying PDE coefficient (permeability a(x))
        - y: the output solution field (pressure u(x))
        
    For time-series without separate inputs:
        - input_field: None (falls back to transition dynamics)
        - y: the observed field at this time step
    """
    u: jnp.ndarray                         # (N,) parameter coordinates
    v: jnp.ndarray                         # (N,)
    y: jnp.ndarray                         # (N,) output field observations
    w: jnp.ndarray                         # (N,) observation weights
    input_field: Optional[jnp.ndarray] = None  # (N,) input field at same (u,v)
    time_index: Optional[int] = None       # for time-ordered ensembles
    metadata: Dict = field(default_factory=dict)

@dataclass
class LearnedOperator:
    """Elliptic operator coefficients (all as spline control nets)."""
    a11: jnp.ndarray    # (n_ctrl_u, n_ctrl_v)
    a12: jnp.ndarray    # cross-diffusion (makes non-separable)
    a22: jnp.ndarray
    b1: jnp.ndarray     # advection
    b2: jnp.ndarray
    # Note: g (forcing) intentionally excluded - it's surface-specific
    
    def to_dict(self) -> Dict[str, jnp.ndarray]:
        return {"a11": self.a11, "a12": self.a12, "a22": self.a22, 
                "b1": self.b1, "b2": self.b2}
    
    @classmethod
    def from_dict(cls, d: Dict[str, jnp.ndarray]) -> "LearnedOperator":
        return cls(a11=d["a11"], a12=d["a12"], a22=d["a22"],
                   b1=d["b1"], b2=d["b2"])
    
    def mean_coeffs(self) -> Dict[str, float]:
        """Spatial mean of each coefficient."""
        return {k: float(jnp.mean(v)) for k, v in self.to_dict().items()}

@dataclass
class SpectralDecomposition:
    """Eigenbasis of the learned operator."""
    eigenvalues: jnp.ndarray              # (n_modes,)
    eigenvectors: List[jnp.ndarray]       # list of (n_ctrl_u, n_ctrl_v)
    mass_matrix: jnp.ndarray              # (n_dof, n_dof)
    operator: LearnedOperator             # the operator this came from
    basis: BSplineBasis2D                 # the spline basis
    
    @property
    def n_modes(self) -> int:
        return len(self.eigenvalues)

    def project(self, f_ctrl: jnp.ndarray, n_modes: Optional[int] = None) -> jnp.ndarray:
        """
        Project surface to spectral coefficients: c_k = <f, φ_k>_M
        
        Uses Hermitian inner product for complex eigenvectors (non-self-adjoint
        operators with advection).
        """
        if n_modes is None:
            n_modes = self.n_modes
            
        f_flat = f_ctrl.ravel()
        coeffs = []
        
        for k in range(n_modes):
            phi_flat = self.eigenvectors[k].ravel()
            c_k = f_flat @ self.mass_matrix @ phi_flat.conj()
            coeffs.append(c_k)
                
        return jnp.array(coeffs)    

    def reconstruct(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct from spectral coefficients: f = Σ c_k φ_k"""
        result = jnp.zeros_like(self.eigenvectors[0])
        for k, c in enumerate(coeffs):
            result = result + c * self.eigenvectors[k]
        return result.real
    
    def reconstruction_error(self, f_ctrl: jnp.ndarray, n_modes: int) -> float:
        """Relative L2 reconstruction error with n_modes."""
        n_modes = min(n_modes, self.n_modes)
        f_flat = f_ctrl.ravel()
        f_norm = jnp.sqrt(f_flat @ self.mass_matrix @ f_flat)
        
        coeffs = self.project(f_ctrl, n_modes)
        f_recon = self.reconstruct(coeffs)
        err_flat = (f_ctrl - f_recon).ravel()
        err_norm = jnp.sqrt(err_flat @ self.mass_matrix @ err_flat)
        
        return float(err_norm / (f_norm + 1e-10))


# =============================================================================
# COUPLED SURFACE + OPERATOR LEARNING
# =============================================================================

def compute_pde_residual(
    u: jnp.ndarray, v: jnp.ndarray,
    f_ctrl: jnp.ndarray, op_params: Dict[str, jnp.ndarray], g_ctrl: jnp.ndarray,
    basis: BSplineBasis2D,
    use_divergence_form: bool = True,
) -> jnp.ndarray:
    """
    Compute PDE residual of the divergence-form elliptic operator:

        ∇·(A ∇f) + b·∇f - g = 0

    where A = [[a11, a12], [a12, a22]].

    Expanding ∇·(A ∇f) via the product rule:

        a11·f_uu + 2a12·f_uv + a22·f_vv          (A : Hess(f))
      + (∂_u a11 + ∂_v a12)·f_u                   (row-divergence of A, row 1)
      + (∂_u a12 + ∂_v a22)·f_v                   (row-divergence of A, row 2)

    The learned b = (b1, b2) captures ONLY genuine advection, not the spatial
    variation of A.  Without the ∂A terms the optimizer is free to set A
    arbitrarily and let b absorb ∂A, making A meaningless and the ellipticity
    constraint vacuous.

    Args:
        use_divergence_form: If True (default, correct), include ∂A terms so
            that A is the true diffusion tensor.  If False, use the non-
            divergence form where b absorbs ∂A (for ablation only).
    """
    d = basis.derivs(u, v, f_ctrl)

    if use_divergence_form:
        g_a11 = basis.eval_with_grad(u, v, op_params["a11"])
        g_a12 = basis.eval_with_grad(u, v, op_params["a12"])
        g_a22 = basis.eval_with_grad(u, v, op_params["a22"])

        a11, a12, a22 = g_a11["f"], g_a12["f"], g_a22["f"]

        diffusion_2nd = a11 * d["fuu"] + 2 * a12 * d["fuv"] + a22 * d["fvv"]

        divA_1 = g_a11["fu"] + g_a12["fv"]
        divA_2 = g_a12["fu"] + g_a22["fv"]
        diffusion_1st = divA_1 * d["fu"] + divA_2 * d["fv"]

        Lf_diff = diffusion_2nd + diffusion_1st
    else:
        a11 = basis.eval(u, v, op_params["a11"])
        a12 = basis.eval(u, v, op_params["a12"])
        a22 = basis.eval(u, v, op_params["a22"])
        Lf_diff = a11 * d["fuu"] + 2 * a12 * d["fuv"] + a22 * d["fvv"]

    b1 = basis.eval(u, v, op_params["b1"])
    b2 = basis.eval(u, v, op_params["b2"])
    advection = b1 * d["fu"] + b2 * d["fv"]

    g = basis.eval(u, v, g_ctrl)

    return Lf_diff + advection - g


def smoothness_l2(C: jnp.ndarray) -> jnp.ndarray:
    """Second-difference smoothness penalty."""
    d2u = C[2:, :] - 2*C[1:-1, :] + C[:-2, :]
    d2v = C[:, 2:] - 2*C[:, 1:-1] + C[:, :-2]
    return jnp.mean(d2u**2) + jnp.mean(d2v**2)


def ellipticity_penalty(
    op_params: Dict[str, jnp.ndarray],
    basis: BSplineBasis2D,
    u_eval: jnp.ndarray,
    v_eval: jnp.ndarray,
) -> jnp.ndarray:
    """
    Penalty for violating ellipticity of the diffusion tensor A(u,v).

    Evaluates the spline functions at real domain points (not raw control
    nets, which would be invalid).
    """
    a11_vals = basis.eval(u_eval, v_eval, op_params["a11"])
    a12_vals = basis.eval(u_eval, v_eval, op_params["a12"])
    a22_vals = basis.eval(u_eval, v_eval, op_params["a22"])

    eps = 0.01
    neg_a11 = jnp.mean(jnp.maximum(-a11_vals + eps, 0) ** 2)
    neg_a22 = jnp.mean(jnp.maximum(-a22_vals + eps, 0) ** 2)

    det_vals = a11_vals * a22_vals - a12_vals ** 2
    neg_det = jnp.mean(jnp.maximum(-det_vals + eps, 0) ** 2)

    return neg_a11 + neg_a22 + 10 * neg_det


def normalization_loss(op_list: List[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
    """Encourage mean(a11 + a22) ≈ 1 for identifiability."""
    loss = jnp.array(0.0)
    for op in op_list:
        trace = jnp.mean(op["a11"]) + jnp.mean(op["a22"])
        loss = loss + (trace - 1.0)**2
    return loss / len(op_list)
