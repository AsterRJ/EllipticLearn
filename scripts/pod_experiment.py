"""
POD (Proper Orthogonal Decomposition) baseline for the Bristol PWI
crack-sizing benchmark.

Purpose
-------
We argue that a PDE-operator-derived eigenbasis
outperforms a purely data-covariance-derived basis.  This script provides the
critical control experiment: replace Stages 1–2 of the spectral pipeline with
a per-view SVD (POD / Karhunen–Loève) basis learned from the same unlabelled
simulated images, keeping Stage 3 (spectral projection) and Stage 4 (MLP) 
identical.

If the spectral MLP's advantage over this POD MLP is significant, the benefit
comes specifically from the operator structure — not merely from dimensionality
reduction followed by nonlinear decoding.

Design (mirrors spectral pipeline exactly)
------------------------------------------
  Stage 1  →  SVD of the flattened image ensemble (per view, unlabelled)
  Stage 2  →  Retain top-N_modes left singular vectors as the POD basis
  Stage 3  →  Project each view of each image onto its POD basis  ←  same as spectral Stage 3
  Stage 4  →  MLP: ℝ^{4·N_modes} → ℝ²   [L̂, θ̂]                ←  identical to spectral

Key design choices
------------------
- POD basis is learned from the FULL sim_train pool (same pool the spectral
  method uses for operator learning), without using labels at all.
- n_modes = 64 per view → 256-D descriptor, identical to spectral.
- The MLP architecture, learning rate, epochs, patience, and batch size are
  copied verbatim from make_spectral_cfg() so the comparison
  is purely about the basis, not the regressor.
- The same data splits, subsampling seeds, and checkpoint format
  are used so results files are directly comparable.

Usage
-----
  python run_pod_experiment.py --data_root /path/to/dataset
  python run_pod_experiment.py --data_root /path/to/dataset --dataset speed_vary
  python run_pod_experiment.py --data_root /path/to/dataset --quick

Output
------
  <results_dir>/pod_experiment/<dataset>/results_partial.json   ← rolling checkpoint
  <results_dir>/pod_experiment/<dataset>/data_size_results.json ← final raw results
  <results_dir>/pod_experiment/<dataset>/summary.json           ← aggregated table
  <results_dir>/pod_experiment/<dataset>/run_config.json        ← run parameters
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ===========================================================================
# DATA LOADING 
# ===========================================================================

@dataclass(frozen=True)
class SplitIdx:
    sim_train: np.ndarray
    sim_val:   np.ndarray
    exp_val:   np.ndarray
    exp_test:  np.ndarray


def make_splits(n_sim: int, n_exp: int, *, sim_train_frac: float, seed: int) -> SplitIdx:
    rng = np.random.default_rng(seed)
    sim = np.arange(n_sim, dtype=np.int64)
    rng.shuffle(sim)
    n_tr = int(round(sim_train_frac * n_sim))
    sim_train = sim[:n_tr]
    sim_val   = sim[n_tr:]
    exp = np.arange(n_exp, dtype=np.int64)
    rng.shuffle(exp)
    n_ev = n_exp // 2
    exp_val  = exp[:n_ev]
    exp_test = exp[n_ev:]
    return SplitIdx(sim_train=sim_train, sim_val=sim_val,
                    exp_val=exp_val,   exp_test=exp_test)


class RawBristolMMap:

    def __init__(self, *, root: Path, dataset_type: str,
                 seed: int, sim_train_frac: float = 0.85):
        self.root = Path(root)
        sub = "StandardDataSet" if dataset_type == "standard" else "SpeedVaryDataSet"
        self.dir  = self.root / sub
        self.seed = seed
        self.sim_train_frac = float(sim_train_frac)
        self.splits: Optional[SplitIdx] = None
        self.X_sim = self.Y_sim = self.X_exp = self.Y_exp = None

    def load(self) -> "RawBristolMMap":
        self.X_sim = np.load(self.dir / "PWI_S.npy", mmap_mode="r")
        self.Y_sim = np.load(self.dir / "Y_S.npy",   mmap_mode="r")
        self.X_exp = np.load(self.dir / "PWI_E.npy", mmap_mode="r")
        self.Y_exp = np.load(self.dir / "Y_E.npy",   mmap_mode="r")
        assert self.X_sim.shape[1:] == (32, 32, 4)
        assert self.X_exp.shape[1:] == (32, 32, 4)
        self.splits = make_splits(
            n_sim=self.X_sim.shape[0], n_exp=self.X_exp.shape[0],
            sim_train_frac=self.sim_train_frac, seed=self.seed,
        )
        return self

    @staticmethod
    def materialise(Xsrc, Ysrc, idx, *, chunk: int = 2048
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64)
        N   = idx.shape[0]
        X_out = np.empty((N, 4, 32, 32), dtype=np.float32)
        Y_out = np.empty((N, 2),         dtype=np.float32)
        P_out = np.empty((N,),           dtype=np.float32)
        for s in range(0, N, chunk):
            e   = min(s + chunk, N)
            gidx = idx[s:e]
            xb  = np.asarray(Xsrc[gidx], dtype=np.float32).transpose(0, 3, 1, 2)
            yb  = np.asarray(Ysrc[gidx], dtype=np.float32)
            X_out[s:e] = xb
            Y_out[s:e] = yb[:, :2]
            P_out[s:e] = yb[:, 2] if yb.shape[1] > 2 else np.nan
        return X_out, Y_out, P_out


def subsample_indices(N: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=min(n, N), replace=False)).astype(np.int64)


def P_or_none(P: np.ndarray) -> Optional[np.ndarray]:
    return None if np.all(~np.isfinite(P)) else P


# ===========================================================================
# POD BASIS  (Stages 1–2 replacement)
# ===========================================================================

VIEW_NAMES = ["SS-S_A", "SS-L_A", "SS-S_B", "SS-L_B"]
N_VIEWS    = 4
N_PIX      = 32   # 32×32 images → 1024 pixels per view


class PerViewPODBasis:
    """
    Compute per-view POD (truncated SVD) bases from an unlabelled image pool.

    For each view v:
        X_v  ∈ ℝ^{K × 1024}   (K images, 1024 pixels each, mean-centred)
        U_v  ∈ ℝ^{1024 × N}   (top-N left singular vectors)
        mean_v ∈ ℝ^{1024}      (per-pixel mean over pool)

    Projection of a new image x ∈ ℝ^{1024}:
        c_v = U_v^T (x − mean_v)   ∈ ℝ^N

    All four views concatenated:
        d = [c_0 | c_1 | c_2 | c_3]  ∈ ℝ^{4N}
    """

    def __init__(self, n_modes: int = 64):
        self.n_modes  = n_modes
        self.bases:  List[np.ndarray] = []   # list of U_v  (1024, n_modes)
        self.means:  List[np.ndarray] = []   # list of mean_v (1024,)
        self._fitted = False

    def fit(self, X_pool: np.ndarray, verbose: bool = True) -> "PerViewPODBasis":
        """
        Parameters
        ----------
        X_pool : (K, 4, 32, 32) float32  —  unlabelled image pool
        """
        K = X_pool.shape[0]
        if verbose:
            print(f"  POD: fitting {N_VIEWS} per-view bases from {K} images "
                  f"({self.n_modes} modes each) …")
        t0 = time.time()

        self.bases = []
        self.means = []

        for v in range(N_VIEWS):
            # Flatten: (K, 1024)
            Xv = X_pool[:, v, :, :].reshape(K, -1).astype(np.float64)

            # Mean-centre
            mean_v = Xv.mean(axis=0)
            Xv_c   = Xv - mean_v

            # Truncated SVD  (economy SVD, keep top n_modes)
            n_keep = min(self.n_modes, K - 1, N_PIX * N_PIX)
            U, s, Vt = np.linalg.svd(Xv_c, full_matrices=False)
            # Vt rows are right singular vectors → left singular vectors of Xv_c
            # We want U_v s.t. projection = U_v^T (x - mean_v)
            # Standard POD: U_v = Vt[:n_keep].T  (1024, n_keep)
            U_v = Vt[:n_keep].T      # (1024, n_keep)

            self.bases.append(U_v.astype(np.float32))
            self.means.append(mean_v.astype(np.float32))

            var_frac = (s[:n_keep]**2).sum() / (s**2).sum()
            if verbose:
                print(f"    [{VIEW_NAMES[v]}] variance captured by {n_keep} modes: "
                      f"{100*var_frac:.1f}%")

        self._fitted = True
        if verbose:
            print(f"  POD: basis fitting complete ({time.time()-t0:.1f}s)")
        return self

    def project_single(self, image_4ch: np.ndarray) -> np.ndarray:
        """
        Project one (4, 32, 32) image → (4 * n_modes,) float32 descriptor.
        """
        parts = []
        for v in range(N_VIEWS):
            x   = image_4ch[v].ravel().astype(np.float32)
            c_v = self.bases[v].T @ (x - self.means[v])   # (n_modes,)
            parts.append(c_v)
        return np.concatenate(parts)

    def project_batch(self, images: np.ndarray, verbose: bool = True,
                      chunk: int = 2048) -> np.ndarray:
        """
        Project (N, 4, 32, 32) → (N, 4*n_modes) float32.
        """
        N   = images.shape[0]
        dim = N_VIEWS * self.n_modes
        D   = np.zeros((N, dim), dtype=np.float32)
        t0  = time.time()

        # Vectorised per view for speed
        for v in range(N_VIEWS):
            Xv = images[:, v, :, :].reshape(N, -1).astype(np.float32)
            Xv -= self.means[v]          # broadcast subtract mean
            # projection: (N, 1024) @ (1024, n_modes) → (N, n_modes)
            D[:, v*self.n_modes:(v+1)*self.n_modes] = Xv @ self.bases[v]

        if verbose:
            print(f"  Projected {N} images ({time.time()-t0:.1f}s)")
        return D


# ===========================================================================
# MLP REGRESSION HEAD
# ===========================================================================

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)


class MLPRegressor:
    """Tikhonov linear ('linear') or MLP ('mlp') regression head."""

    def __init__(self, *, mode: str = "mlp", hidden: int = 256,
                 dropout: float = 0.1, lr: float = 1e-3,
                 epochs: int = 1000, patience: int = 50,
                 batch_size: int = 256, l2_weight: float = 1e-4,
                 reg_lambda: float = 1e-3,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 seed: int = 42):
        self.mode       = mode
        self.hidden     = hidden
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.patience   = patience
        self.batch_size = batch_size
        self.l2_weight  = l2_weight
        self.reg_lambda = reg_lambda
        self.device     = device
        self.seed       = seed

        self._mlp       = None
        self._W = self._b = None
        self._scaler_D = self._scaler_Y = None

    def fit(self, D_train, Y_train, D_val=None, Y_val=None, verbose=True):
        from sklearn.preprocessing import StandardScaler
        self._scaler_D = StandardScaler().fit(D_train)
        self._scaler_Y = StandardScaler().fit(Y_train)
        D_tr = self._scaler_D.transform(D_train)
        Y_tr = self._scaler_Y.transform(Y_train)

        if self.mode == "linear":
            lam = self.reg_lambda
            p   = D_tr.shape[1]
            W_T = np.linalg.solve(D_tr.T @ D_tr + lam * np.eye(p), D_tr.T @ Y_tr)
            self._W = W_T.T
            self._b = np.zeros(2, dtype=np.float32)
        else:
            self._fit_mlp(D_tr, Y_tr,
                          self._scaler_D.transform(D_val) if D_val is not None else None,
                          self._scaler_Y.transform(Y_val) if D_val is not None else None,
                          verbose)
        return self

    def predict(self, D: np.ndarray) -> np.ndarray:
        D_s = self._scaler_D.transform(D)
        if self.mode == "linear":
            Y_s = D_s @ self._W.T + self._b
        else:
            device = torch.device(self.device)
            self._mlp.eval()
            with torch.no_grad():
                Dt  = torch.from_numpy(D_s.astype(np.float32)).to(device)
                Y_s = self._mlp(Dt).cpu().numpy()
        return self._scaler_Y.inverse_transform(Y_s).astype(np.float32)

    def _fit_mlp(self, D_tr, Y_tr, D_v, Y_v, verbose):
        torch.manual_seed(self.seed)
        device = torch.device(self.device)
        in_dim = D_tr.shape[1]

        self._mlp = _MLP(in_dim, self.hidden, self.dropout).to(device)
        optimizer = optim.Adam(self._mlp.parameters(), lr=self.lr,
                               weight_decay=self.l2_weight)
        criterion = nn.MSELoss()

        def to_loader(D, Y, shuffle):
            Dt = torch.from_numpy(D.astype(np.float32))
            Yt = torch.from_numpy(Y.astype(np.float32))
            return DataLoader(TensorDataset(Dt, Yt),
                              batch_size=self.batch_size, shuffle=shuffle)

        tr_loader  = to_loader(D_tr, Y_tr, shuffle=True)
        val_loader = to_loader(D_v, Y_v, shuffle=False) if D_v is not None else None

        best_val, best_state, wait = float("inf"), None, 0

        for epoch in range(1, self.epochs + 1):
            self._mlp.train()
            for Db, Yb in tr_loader:
                Db, Yb = Db.to(device), Yb.to(device)
                optimizer.zero_grad()
                criterion(self._mlp(Db), Yb).backward()
                optimizer.step()

            if val_loader is not None:
                self._mlp.eval()
                with torch.no_grad():
                    vl = np.mean([criterion(self._mlp(Db.to(device)),
                                            Yb.to(device)).item()
                                  for Db, Yb in val_loader])
                if vl < best_val:
                    best_val  = vl
                    best_state = {k: v.clone() for k, v in self._mlp.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        if verbose:
                            print(f"  [MLP] Early stop at epoch {epoch}")
                        break

        if best_state is not None:
            self._mlp.load_state_dict(best_state)


# ===========================================================================
# FULL POD CHARACTERIZER
# ===========================================================================

def crack_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
    """MAE, RMSE for L, theta, through-wall extent h, and code-compliance rate CR."""
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    err    = Y_pred - Y_true
    # Through-wall extent h = L * cos(theta_deg)
    h_true = Y_true[:, 0] * np.cos(np.deg2rad(Y_true[:, 1]))
    h_pred = Y_pred[:, 0] * np.cos(np.deg2rad(Y_pred[:, 1]))
    h_err  = h_pred - h_true
    return {
        "MAE_L":      float(np.mean(np.abs(err[:, 0]))),
        "MAE_theta":  float(np.mean(np.abs(err[:, 1]))),
        "MAE_h":      float(np.mean(np.abs(h_err))),
        "RMSE_L":     float(np.sqrt(np.mean(err[:, 0]**2))),
        "RMSE_theta": float(np.sqrt(np.mean(err[:, 1]**2))),
        "RMSE_h":     float(np.sqrt(np.mean(h_err**2))),
        "CR":         float(np.mean(np.abs(h_err) <= 1.5)),   # BS 7910 ±1.5 mm
        "std_L":      float(np.std(err[:, 0])),
        "std_theta":  float(np.std(err[:, 1])),
    }


class PODCrackCharacterizer:
    """
    POD + MLP pipeline.

    fit(X_train, Y_train, X_val, Y_val)
    predict(X_test)
    """

    def __init__(self, n_modes: int = 64, mlp_hidden: int = 256,
                 mlp_dropout: float = 0.1, mlp_lr: float = 1e-3,
                 mlp_epochs: int = 1000, mlp_patience: int = 50,
                 mlp_batch_size: int = 256, mlp_l2: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 seed: int = 42,
                 pod_pool_size: int = 400,     # max images used to build POD basis
                 mode: str = "mlp"):           # "mlp" or "linear"
        self.n_modes       = n_modes
        self.pod_pool_size = pod_pool_size
        self.mode          = mode
        self.device        = device
        self.seed          = seed

        self._mlp_kwargs = dict(
            mode=mode, hidden=mlp_hidden, dropout=mlp_dropout,
            lr=mlp_lr, epochs=mlp_epochs, patience=mlp_patience,
            batch_size=mlp_batch_size, l2_weight=mlp_l2,
            device=device, seed=seed,
        )

        self.basis:    Optional[PerViewPODBasis] = None
        self.regressor: Optional[MLPRegressor]  = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray,
            X_val:   Optional[np.ndarray] = None,
            Y_val:   Optional[np.ndarray] = None,
            verbose: bool = True) -> "PODCrackCharacterizer":

        # ── Stage 1+2: POD basis from unlabelled training images ─────────
        # Use a random subsample of up to pod_pool_size images for the SVD,
        # mirroring the operator-learning pool in the spectral method.
        rng     = np.random.default_rng(self.seed)
        pool_n  = min(self.pod_pool_size, X_train.shape[0])
        pool_idx = np.sort(rng.choice(X_train.shape[0], size=pool_n, replace=False))
        X_pool  = X_train[pool_idx]

        print(f"\n══ POD Stage 1–2: SVD basis from {pool_n} unlabelled images ══")
        self.basis = PerViewPODBasis(n_modes=self.n_modes)
        self.basis.fit(X_pool, verbose=verbose)

        # ── Stage 3: project all training and validation images ──────────
        print(f"\n══ POD Stage 3: Projection ══")
        print(f"  Projecting {len(X_train)} training images …")
        D_train = self.basis.project_batch(X_train, verbose=verbose)

        D_val = None
        if X_val is not None:
            print(f"  Projecting {len(X_val)} validation images …")
            D_val = self.basis.project_batch(X_val, verbose=False)

        # ── Stage 4: MLP regression ──────────────────────────────────────
        print(f"\n══ POD Stage 4: MLP Regression ══")
        self.regressor = MLPRegressor(**self._mlp_kwargs)
        self.regressor.fit(D_train, Y_train, D_val, Y_val, verbose=verbose)

        return self

    def predict(self, X: np.ndarray, **_) -> np.ndarray:
        D = self.basis.project_batch(X, verbose=False)
        return self.regressor.predict(D)


# ===========================================================================
# EXPERIMENT ORCHESTRATION
# ===========================================================================

def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _checkpoint(results: List[Dict], out_dir: Path) -> None:
    _atomic_write(out_dir / "results_partial.json", results)
    _atomic_write(out_dir / "data_size_results.json", results)


def _mlp_kwargs_for_n(n_train: int, device: str, quick: bool) -> Dict:
    """Mirror the MLP hypers from make_spectral_cfg() in renm2.py."""
    if quick:
        return dict(mlp_hidden=64, mlp_epochs=20, mlp_patience=10,
                    mlp_batch_size=min(64, max(8, n_train)), device=device)
    return dict(mlp_hidden=256, mlp_epochs=1000, mlp_patience=50,
                mlp_batch_size=min(256, max(8, n_train)), device=device)


def _pod_pool_for_n(n_train: int, quick: bool) -> int:
    """Pool size for SVD: same scaling as n_stage1_samples in renm2.py."""
    if quick:
        return min(n_train, 50)
    return min(max(n_train, 10), 400)


def run_single_trial(
    *,
    n_train: int,
    method: str,              # "pod_mlp" or "pod_linear"
    X_train_full: np.ndarray,
    Y_train_full: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    quick: bool,
    seed: int,
    device: str,
    n_modes: int = 64,
) -> Dict:

    sub_idx = subsample_indices(X_train_full.shape[0], n_train, seed)
    X_sub   = X_train_full[sub_idx]
    Y_sub   = Y_train_full[sub_idx]
    actual_n = X_sub.shape[0]
    print(f"\n  [{method}] n_train = {actual_n}")

    mode = "linear" if method == "pod_linear" else "mlp"
    t0   = time.time()

    model = PODCrackCharacterizer(
        n_modes       = n_modes,
        pod_pool_size = _pod_pool_for_n(actual_n, quick),
        mode          = mode,
        seed          = seed,
        **_mlp_kwargs_for_n(actual_n, device, quick),
    )
    model.fit(X_sub, Y_sub, X_val, Y_val, verbose=False)
    Y_pred = model.predict(X_test)

    train_time = time.time() - t0
    metrics    = crack_metrics(Y_test, Y_pred)

    result = {
        "method":       method,
        "n_train":      int(actual_n),
        "mae_L":        float(metrics["MAE_L"]),
        "mae_theta":    float(metrics["MAE_theta"]),
        "mae_h":        float(metrics["MAE_h"]),
        "CR":           float(metrics["CR"]),
        "train_time_s": float(round(train_time, 3)),
        "metrics":      {k: float(v) for k, v in metrics.items()},
    }
    print(
        f"    MAE(L)={result['mae_L']:.4f} mm  "
        f"MAE(h)={result['mae_h']:.4f} mm  "
        f"CR={result['CR']:.3f}  "
        f"MAE(θ)={result['mae_theta']:.4f}°  "
        f"time={train_time:.1f}s"
    )
    return result


def run_experiment(
    *,
    data_root:    Path,
    results_dir:  Path,
    dataset_type: str,
    methods:      List[str],
    sizes:        List[int],
    n_repeats:    int,
    quick:        bool,
    seed:         int,
    device:       str,
    n_modes:      int,
) -> List[Dict]:

    out_dir = results_dir / "pod_experiment" / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  POD BASELINE EXPERIMENT — {dataset_type.upper()}")
    print(f"{'═'*70}")

    dm = RawBristolMMap(root=data_root, dataset_type=dataset_type,
                        seed=seed, sim_train_frac=0.85).load()

    X_train_full, Y_train_full, _ = dm.materialise(
        dm.X_sim, dm.Y_sim, dm.splits.sim_train, chunk=2048)
    X_val, Y_val, _   = dm.materialise(
        dm.X_sim, dm.Y_sim, dm.splits.sim_val,   chunk=2048)
    X_test, Y_test, _ = dm.materialise(
        dm.X_exp, dm.Y_exp, dm.splits.exp_test,  chunk=2048)

    N_full = X_train_full.shape[0]
    print(f"  Full train pool: {N_full}  |  Val: {len(X_val)}  |"
          f"  Test (exp): {len(X_test)}")
    print(f"  Sizes:   {sizes}")
    print(f"  Methods: {methods}")
    print(f"  Repeats: {n_repeats}  |  n_modes: {n_modes}")

    all_results: List[Dict] = []

    for n_train in sizes:
        if n_train > N_full:
            print(f"\n  ⚠ Skipping n_train={n_train} (> full set {N_full})")
            continue

        for rep in range(n_repeats):
            trial_seed = seed + rep * 1000 + int(n_train)
            print(f"\n{'─'*60}")
            print(f"  SIZE={n_train}  REPEAT={rep+1}/{n_repeats}  seed={trial_seed}")
            print(f"{'─'*60}")

            for method in methods:
                record = None
                try:
                    record = run_single_trial(
                        n_train      = int(n_train),
                        method       = method,
                        X_train_full = X_train_full,
                        Y_train_full = Y_train_full,
                        X_val        = X_val,
                        Y_val        = Y_val,
                        X_test       = X_test,
                        Y_test       = Y_test,
                        quick        = quick,
                        seed         = trial_seed,
                        device       = device,
                        n_modes      = n_modes,
                    )
                    record["repeat"]  = int(rep)
                    record["seed"]    = int(trial_seed)
                    record["dataset"] = dataset_type

                except Exception as exc:
                    import traceback
                    print(f"    ✗ {method} n={n_train} rep={rep} FAILED: {exc}")
                    traceback.print_exc()
                    record = {
                        "method":   method,
                        "n_train":  int(n_train),
                        "repeat":   int(rep),
                        "seed":     int(trial_seed),
                        "dataset":  dataset_type,
                        "error":    str(exc),
                    }
                finally:
                    all_results.append(record)
                    _checkpoint(all_results, out_dir)

    _atomic_write(out_dir / "data_size_results.json", all_results)
    print(f"\n  Results saved → {out_dir / 'data_size_results.json'}")
    return all_results


def summarise(all_results: List[Dict], results_dir: Path, dataset_type: str) -> None:
    out_dir = results_dir / "pod_experiment" / dataset_type
    valid   = [r for r in all_results if "error" not in r and r is not None]
    if not valid:
        print("  No valid results to summarise.")
        return

    methods = sorted(set(r["method"] for r in valid))
    sizes   = sorted(set(r["n_train"] for r in valid))

    summary: Dict = {}
    for m in methods:
        summary[m] = {}
        for n in sizes:
            trials = [r for r in valid if r["method"] == m and r["n_train"] == n]
            if not trials:
                continue
            mae_L  = np.array([t["mae_L"]            for t in trials])
            mae_th = np.array([t["mae_theta"]         for t in trials])
            mae_h  = np.array([t.get("mae_h", np.nan) for t in trials])
            CR     = np.array([t.get("CR", np.nan)    for t in trials])
            times  = np.array([t["train_time_s"]      for t in trials])
            summary[m][n] = {
                "mae_L_mean":     float(np.nanmean(mae_L)),
                "mae_L_std":      float(np.nanstd(mae_L)),
                "mae_theta_mean": float(np.nanmean(mae_th)),
                "mae_theta_std":  float(np.nanstd(mae_th)),
                "mae_h_mean":     float(np.nanmean(mae_h)),
                "mae_h_std":      float(np.nanstd(mae_h)),
                "CR_mean":        float(np.nanmean(CR)),
                "CR_std":         float(np.nanstd(CR)),
                "time_mean":      float(times.mean()),
                "time_std":       float(times.std()),
                "n_trials":       len(trials),
            }

    _atomic_write(out_dir / "summary.json", summary)

    print(f"\n{'═'*80}")
    print(f"  POD EXPERIMENT SUMMARY — {dataset_type.upper()}")
    print(f"{'═'*80}")
    hdr = (f"{'Method':<18} {'N':>7}  {'MAE_L mm':>12}  "
           f"{'MAE_h mm':>12}  {'CR (1.5mm)':>11}  {'Time(s)':>10}")
    print(hdr); print("─" * len(hdr))
    for m in methods:
        for n in sizes:
            s = summary[m].get(n)
            if s is None:
                continue
            print(f"{m:<18} {n:>7}  "
                  f"{s['mae_L_mean']:>5.4f}±{s['mae_L_std']:<4.4f}  "
                  f"{s['mae_h_mean']:>5.4f}±{s['mae_h_std']:<4.4f}  "
                  f"{s['CR_mean']:>6.3f}±{s['CR_std']:<5.3f}  "
                  f"{s['time_mean']:>8.1f}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="POD baseline experiment for Bristol PWI crack sizing")
    parser.add_argument(
        "--data_root", type=Path,
        default="/workspaces/surface_spline_prob/src/ut_baseline/1u376lnlam8ac2lf0uato86zrr",
        help="Root directory containing StandardDataSet/ and SpeedVaryDataSet/")
    parser.add_argument(
        "--results_dir", type=Path,
        default=Path("./ut_results_pod_Nfull"),
        help="Output directory (pod_experiment/<dataset>/ will be created inside)")
    parser.add_argument(
        "--dataset", type=str, default="standard",
        choices=["standard", "speed_vary"])
    parser.add_argument(
        "--methods", nargs="+",
        default=["pod_mlp", "pod_linear"],
        help="Which POD variants to run: pod_mlp and/or pod_linear")
    parser.add_argument(
        "--sizes", nargs="*", type=int, default=None,
        help="Training set sizes to sweep (default: [10, 100, 1000, 10000, N_full])")
    parser.add_argument(
        "--n_repeats", type=int, default=5,
        help="Random seeds per (method, size) combination")
    parser.add_argument(
        "--n_modes", type=int, default=64,
        help="POD modes per view (total descriptor = 4 * n_modes, default 64 → 256-D)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke-test mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Determine sizes
    if not args.sizes:
        dm = RawBristolMMap(root=args.data_root, dataset_type=args.dataset,
                            seed=args.seed).load()
        N_full = dm.splits.sim_train.shape[0]
        if args.quick:
            sizes = sorted(set(n for n in [10, 50, 200, 1000, N_full] if n <= N_full))
        else:
            sizes = [N_full]#sorted(set(n for n in [100, 200, 500, 1000, 2000, 10000, N_full] if n <= N_full))
    else:
        sizes = sorted(set(args.sizes))

    # Save config
    cfg = {
        "data_root":  str(args.data_root),
        "results_dir": str(args.results_dir),
        "dataset":    args.dataset,
        "methods":    args.methods,
        "sizes":      sizes,
        "n_repeats":  args.n_repeats,
        "n_modes":    args.n_modes,
        "quick":      args.quick,
        "seed":       args.seed,
        "device":     args.device,
    }
    cfg_path = args.results_dir / "pod_experiment" / args.dataset / "run_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Run
    all_results = run_experiment(
        data_root    = args.data_root,
        results_dir  = args.results_dir,
        dataset_type = args.dataset,
        methods      = args.methods,
        sizes        = sizes,
        n_repeats    = args.n_repeats,
        quick        = args.quick,
        seed         = args.seed,
        device       = args.device,
        n_modes      = args.n_modes,
    )

    summarise(all_results, args.results_dir, args.dataset)

    print("\n" + "═" * 70)
    print("  POD EXPERIMENT COMPLETE")
    print("═" * 70)


if __name__ == "__main__":
    main()
