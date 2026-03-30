# run_data_size_experiment_no_mask_fourier_wavelet.py  (per-view operator rewrite)
# ---------------------------------------------------------------------------
# Data-size scaling experiment.
#
# Key change: spectral_method_no_mask.py now learns one consensus elliptic
# operator AND eigenbasis per view (SS-S_A, SS-L_A, SS-S_B, SS-L_B).
# Each view is projected through its own physics-matched eigenbasis.
# The 4*N_modes = 256-D descriptor and Stage 4 interface are unchanged.
#
# Produces:
# - results_partial.json (checkpoint after every trial)
# - data_size_results.json (raw results)
# - summary.json (aggregated)
# - learning_curves.png
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Local modules ─────────────────────────────────────────────────────────
from baselines import (
    PyleCNNConfig,
    SixdBDropSizer,
    PyleCNN,
    ClassicalMLBaseline,
    crack_metrics,
)
from elliptic_learning import (
    SpectralCrackCharacterizer,
    SpectralMethodConfig,
    PDELearnerConfig,
    EigenConfig,
    RegressorConfig,
)

# ── Additional baselines: Fourier & Haar-wavelet features ───────────────
from dataclasses import dataclass

def _as_nchw(X: np.ndarray) -> np.ndarray:
    """Ensure X is (N,C,H,W) float32."""
    if X.ndim == 4 and X.shape[-1] in (1,2,3,4) and X.shape[1] not in (1,2,3,4):
        # likely (N,H,W,C)
        X = np.transpose(X, (0,3,1,2))
    return X.astype(np.float32, copy=False)

def fourier_features(X: np.ndarray, *, n_keep: int = 8) -> np.ndarray:
    """Low-frequency |FFT| features per channel. Returns (N, C*n_keep*n_keep)."""
    X = _as_nchw(X)
    N, C, H, W = X.shape
    # rfft2: (H, W//2+1)
    F = np.fft.rfft2(X, axes=(-2,-1))
    Mag = np.abs(F).astype(np.float32)
    kk_h = min(n_keep, H)
    kk_w = min(n_keep, Mag.shape[-1])
    Mag = Mag[:, :, :kk_h, :kk_w]
    return Mag.reshape(N, -1)

def _haar_1d_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Haar split along last axis: returns (avg, diff)."""
    a = x[..., 0::2]
    b = x[..., 1::2]
    avg = (a + b) * 0.5
    diff = (a - b) * 0.5
    return avg, diff

def haar_wavelet_features(X: np.ndarray, *, levels: int = 1) -> np.ndarray:
    """Haar 2D wavelet features (LL,LH,HL,HH) for given levels. Returns flattened coeffs."""
    X = _as_nchw(X)
    N, C, H, W = X.shape
    coeffs = []
    cur = X
    for _lvl in range(levels):
        # Row transform (along W)
        avg_w, diff_w = _haar_1d_rows(cur)  # (N,C,H,W/2) each
        # Col transform (along H)
        avg_w_t = np.transpose(avg_w, (0,1,3,2))
        diff_w_t = np.transpose(diff_w, (0,1,3,2))
        avg_h, diff_h = _haar_1d_rows(avg_w_t)
        avg_h2, diff_h2 = _haar_1d_rows(diff_w_t)
        # back to (N,C,H/2,W/2)
        LL = np.transpose(avg_h,  (0,1,3,2))
        HL = np.transpose(diff_h, (0,1,3,2))
        LH = np.transpose(avg_h2, (0,1,3,2))
        HH = np.transpose(diff_h2,(0,1,3,2))
        coeffs.extend([LL, LH, HL, HH])
        cur = LL  # next level on approximation
    feats = np.concatenate([c.reshape(N, -1).astype(np.float32) for c in coeffs], axis=1)
    return feats

@dataclass
class TransformMLBaseline:
    """Simple baseline: transform -> sklearn regressor (RF or Ridge)."""
    transform: str  # 'fourier' or 'haar'
    model: str = 'rf'  # 'rf' or 'ridge'
    n_keep: int = 8  # for fourier
    levels: int = 1  # for haar
    n_estimators: int = 200
    ridge_alpha: float = 1e-3
    random_state: int = 0

    def _feats(self, X: np.ndarray) -> np.ndarray:
        if self.transform == 'fourier':
            return fourier_features(X, n_keep=self.n_keep)
        if self.transform == 'haar':
            return haar_wavelet_features(X, levels=self.levels)
        raise ValueError(f'Unknown transform: {self.transform}')

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        import sklearn
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge

        Xf = self._feats(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y[:, None]
        if self.model == 'rf':
            base = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._model = MultiOutputRegressor(base)
        elif self.model == 'ridge':
            base = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
            self._model = MultiOutputRegressor(base)
        else:
            raise ValueError(f'Unknown model: {self.model}')
        self._model.fit(Xf, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self._feats(X)
        Yp = self._model.predict(Xf)
        return np.asarray(Yp)


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

def _save_checkpoint(all_results: List[Dict], out_dir: Path) -> None:
    # Always keep a running checkpoint + also mirror to "data_size_results.json"
    _atomic_write_json(out_dir / "results_partial.json", all_results)
    _atomic_write_json(out_dir / "data_size_results.json", all_results)

# ======================================================================
# Configuration factories
# ======================================================================

def make_cnn_cfg(quick: bool) -> PyleCNNConfig:
    if quick:
        return PyleCNNConfig(max_epochs=5, patience=5, n_runs=1)
    return PyleCNNConfig(
        lr=1e-3,
        batch_size=128,
        max_epochs=400,
        patience=150,
        dropout=0.10,
        n_runs=10,
    )


def make_spectral_cfg(quick: bool, n_train: int) -> SpectralMethodConfig:
    """
    Build spectral config, scaling Stage-1 sample count to training budget.
    No data preprocessing occurs here; this is just hyperparameter scaling.
    """
    n_stage1 = min(max(n_train, 10), 400)

    if quick:
        return SpectralMethodConfig(
            pde=PDELearnerConfig(steps=200, log_every=100, n_ctrl_u=6, n_ctrl_v=6),
            eigen=EigenConfig(n_modes=16, nq=20),
            regressor=RegressorConfig(mode="mlp", epochs=20, patience=10, hidden=64),
            n_modes=16,
            n_stage1_samples=min(n_stage1, 50),
        )

    

    if n_train < 100:
        pde_steps = 5000
        n_coll = 14
    elif n_train < 500:
        pde_steps = 10000
        n_coll = 18
    else:
        pde_steps = 20000
        n_coll = 20
        

    return SpectralMethodConfig(
        pde=PDELearnerConfig(
            n_ctrl_u=8,
            n_ctrl_v=8,
            degree=3,
            n_coll=n_coll,
            lam_data = 1.0,
            lam_pde=0.2,
            lam_consensus=0.2,
            lam_elliptic=0.5,
            lam_norm=0.1,
            lam_smooth_f=1e-2,
            lam_smooth_op=5e-3,
            lr=1e-2,
            steps=pde_steps,
            log_every=500,
        ),
        eigen=EigenConfig(n_modes=64, nq=40, is_self_adjoint=True),
        regressor=RegressorConfig(
            mode="mlp",
            hidden=256,       # wider head; descriptor is now view-independent
            dropout=0.1,
            lr=1e-3,
            epochs=1000,
            patience=50,
            batch_size=min(256, max(8, n_train)),
        ),
        n_modes=64,
        n_stage1_samples=n_stage1,  # per-view pool (4 operators trained independently)
    )


# ======================================================================
# Raw mmap loader
# ======================================================================

@dataclass(frozen=True)
class SplitIdx:
    sim_train: np.ndarray
    sim_val: np.ndarray
    exp_val: np.ndarray
    exp_test: np.ndarray


def make_splits(n_sim: int, n_exp: int, *, sim_train_frac: float, seed: int) -> SplitIdx:
    rng = np.random.default_rng(seed)

    sim = np.arange(n_sim, dtype=np.int64)
    rng.shuffle(sim)
    n_tr = int(round(sim_train_frac * n_sim))
    sim_train = sim[:n_tr]
    sim_val = sim[n_tr:]

    exp = np.arange(n_exp, dtype=np.int64)
    rng.shuffle(exp)
    n_ev = n_exp // 2
    exp_val = exp[:n_ev]
    exp_test = exp[n_ev:]

    return SplitIdx(sim_train=sim_train, sim_val=sim_val, exp_val=exp_val, exp_test=exp_test)


class RawBristolMMap:
    def __init__(
        self,
        *,
        root: Path,
        dataset_type: str,
        seed: int,
        sim_train_frac: float = 0.85,
    ):
        self.root = Path(root)
        self.dataset_type = dataset_type
        self.seed = seed
        self.sim_train_frac = float(sim_train_frac)

        sub = "StandardDataSet" if dataset_type == "standard" else "SpeedVaryDataSet"
        self.dir = self.root / sub

        self.X_sim = None
        self.Y_sim = None
        self.X_exp = None
        self.Y_exp = None
        self.mesh = None
        self.splits: Optional[SplitIdx] = None

    def load(self) -> "RawBristolMMap":
        self.X_sim = np.load(self.dir / "PWI_S.npy", mmap_mode="r")  # (N,32,32,4)
        self.Y_sim = np.load(self.dir / "Y_S.npy", mmap_mode="r")    # (N,3)
        self.X_exp = np.load(self.dir / "PWI_E.npy", mmap_mode="r")  # (M,32,32,4)
        self.Y_exp = np.load(self.dir / "Y_E.npy", mmap_mode="r")    # (M,3)
        self.mesh = np.load(self.dir / "Mesh.npy", allow_pickle=True)

        assert self.X_sim.shape[0] == self.Y_sim.shape[0]
        assert self.X_exp.shape[0] == self.Y_exp.shape[0]
        assert self.X_sim.shape[1:] == (32, 32, 4)
        assert self.X_exp.shape[1:] == (32, 32, 4)

        self.splits = make_splits(
            n_sim=self.X_sim.shape[0],
            n_exp=self.X_exp.shape[0],
            sim_train_frac=self.sim_train_frac,
            seed=self.seed,
        )
        return self

    @staticmethod
    def materialise(
        Xsrc: np.memmap,
        Ysrc: np.memmap,
        idx: np.ndarray,
        *,
        chunk: int = 2048,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          X: (N,4,32,32) float32  [raw values, only transposed]
          Y: (N,2) float32        [L, theta]
          P: (N,) float32         [position] (NaN if absent)
        """
        idx = np.asarray(idx, dtype=np.int64)
        N = idx.shape[0]

        X_out = np.empty((N, 4, 32, 32), dtype=np.float32)
        Y_out = np.empty((N, 2), dtype=np.float32)
        P_out = np.empty((N,), dtype=np.float32)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            gidx = idx[s:e]

            xb = np.asarray(Xsrc[gidx], dtype=np.float32)  # (B,32,32,4)
            xb = xb.transpose(0, 3, 1, 2)                  # (B,4,32,32)
            yb = np.asarray(Ysrc[gidx], dtype=np.float32)  # (B,3) or (B,2+)

            X_out[s:e] = xb
            Y_out[s:e] = yb[:, :2]
            if yb.shape[1] > 2:
                P_out[s:e] = yb[:, 2]
            else:
                P_out[s:e] = np.nan

        return X_out, Y_out, P_out


def P_or_none(P: np.ndarray) -> Optional[np.ndarray]:
    return None if np.all(~np.isfinite(P)) else P


# ======================================================================
# Subsampling helper
# ======================================================================

def subsample_indices(N: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, size=min(n, N), replace=False))
    return idx.astype(np.int64)


# ======================================================================
# Single-size trial
# ======================================================================

def run_single_size(
    *,
    n_train: int,
    method: str,
    X_train_full: np.ndarray,
    Y_train_full: np.ndarray,
    P_train_full: Optional[np.ndarray],
    X_val: np.ndarray,
    Y_val: np.ndarray,
    P_val: Optional[np.ndarray],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    P_test: Optional[np.ndarray],
    quick: bool,
    seed: int,
    device: str,
    pixel_size_mm: float,
) -> Dict:
    # ── Subsample training data ──────────────────────────────────────
    N_full = X_train_full.shape[0]
    sub_idx = subsample_indices(N_full, n_train, seed)

    X_sub = X_train_full[sub_idx]
    Y_sub = Y_train_full[sub_idx]
    P_sub = P_train_full[sub_idx] if P_train_full is not None else None

    actual_n = X_sub.shape[0]
    print(f"\n  [{method}] n_train = {actual_n}")


    other = None

    # ── Train & predict ──────────────────────────────────────────────
    t0 = time.time()


    if method == "spectral":
        cfg = make_spectral_cfg(quick, actual_n)
        cfg.regressor.device = device
        model = SpectralCrackCharacterizer(cfg=cfg)
        model.fit(
            X_train=X_sub, Y_train=Y_sub,
            X_val=X_val,   Y_val=Y_val,
            P_train=P_sub, P_val=P_val,
            verbose=True,
        )
        Y_pred = model.predict(X_test, Y_test, P_test)
        other = model.get_diagnostics_dict()

    elif method == "spectral_linear":
        cfg = make_spectral_cfg(quick, actual_n)
        cfg.regressor = RegressorConfig(mode="linear", reg_lambda=1e-3)
        cfg.regressor.device = device
        model = SpectralCrackCharacterizer(cfg=cfg)
        model.fit(
            X_train=X_sub, Y_train=Y_sub,
            X_val=X_val,   Y_val=Y_val,
            P_train=P_sub, P_val=P_val,
            verbose=True,
        )
        Y_pred = model.predict(X_test, Y_test, P_test)
        other = model.get_diagnostics_dict()

    elif method == "cnn":
        cnn_cfg = make_cnn_cfg(quick)
        cnn_cfg.device = device
        model = PyleCNN(cfg=cnn_cfg)
        model.fit(X_sub, Y_sub, X_val, Y_val, verbose=False)
        Y_pred = model.predict(X_test)

    elif method == "rf":
        model = ClassicalMLBaseline(variant="rf", n_estimators=200 if not quick else 20)
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)

    elif method == "svr":
        model = ClassicalMLBaseline(variant="svr")
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)

    elif method == "6db":
        model = SixdBDropSizer(pixel_size_mm=pixel_size_mm)
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)


    elif method == "fourier_rf":
        model = TransformMLBaseline(transform="fourier", model="rf",
                                    n_keep=8 if not quick else 4,
                                    n_estimators=200 if not quick else 20,
                                    random_state=seed)
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)

    elif method == "fourier_ridge":
        model = TransformMLBaseline(transform="fourier", model="ridge",
                                    n_keep=8 if not quick else 4,
                                    ridge_alpha=1e-3,
                                    random_state=seed)
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)

    elif method == "wavelet_rf":
        model = TransformMLBaseline(transform="haar", model="rf",
                                    levels=1 if not quick else 1,
                                    n_estimators=200 if not quick else 20,
                                    random_state=seed)
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)

    elif method == "wavelet_ridge":
        model = TransformMLBaseline(transform="haar", model="ridge",
                                    levels=1 if not quick else 1,
                                    ridge_alpha=1e-3,
                                    random_state=seed)
        model.fit(X_sub, Y_sub)
        Y_pred = model.predict(X_test)

    else:
        raise ValueError(f"Unknown method: {method}")

    train_time = time.time() - t0

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = crack_metrics(Y_test, Y_pred)
    result = {
        "method": method,
        "n_train": int(actual_n),
        "mae_L": float(metrics.get("mae_L", metrics.get("L_mae", np.nan))),
        "mae_theta": float(metrics.get("mae_theta", metrics.get("theta_mae", np.nan))),
        "train_time_s": float(round(train_time, 3)),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "manifold_metrics": other
    }

    print(
        f"    MAE(L)={result['mae_L']:.4f} mm  "
        f"MAE(θ)={result['mae_theta']:.4f}°  "
        f"time={train_time:.1f}s"
    )

    return result


# ======================================================================
# Experiment orchestration
# ======================================================================

def run_data_size_experiment(
    *,
    data_root: Path,
    results_dir: Path,
    dataset_type: str,
    methods: List[str],
    sizes: List[int],
    n_repeats: int,
    quick: bool,
    seed: int,
    device: str,
) -> List[Dict]:
    out_dir = results_dir / "data_size_scaling" / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  DATA-SIZE SCALING EXPERIMENT — {dataset_type.upper()}")
    print(f"{'═'*70}")

    dm = RawBristolMMap(root=data_root, dataset_type=dataset_type, seed=seed, sim_train_frac=0.85).load()
    assert dm.splits is not None

    # materialise fixed val/test sets once
    X_train_full, Y_train_full, P_train_full = dm.materialise(dm.X_sim, dm.Y_sim, dm.splits.sim_train, chunk=2048)
    X_val, Y_val, P_val = dm.materialise(dm.X_sim, dm.Y_sim, dm.splits.sim_val, chunk=2048)
    X_test, Y_test, P_test = dm.materialise(dm.X_exp, dm.Y_exp, dm.splits.exp_test, chunk=2048)

    P_train = P_or_none(P_train_full)
    P_v = P_or_none(P_val)
    P_t = P_or_none(P_test)

    N_full = X_train_full.shape[0]
    pixel_size_mm = (21.0 - 13.0) / 32.0

    print(f"  Full training set: {N_full} samples")
    print(f"  Test set:          {X_test.shape[0]} samples")
    print(f"  Sizes to test:     {sizes}")
    print(f"  Methods:           {methods}")
    print(f"  Repeats per size:  {n_repeats}")

    all_results: List[Dict] = []

    for n_train in sizes:
        if n_train > N_full:
            print(f"\n  ⚠ Skipping n_train={n_train} (exceeds full set {N_full})")
            continue

        for rep in range(n_repeats):
            trial_seed = seed + rep * 1000 + int(n_train)
            print(f"\n{'─'*60}")
            print(f"  SIZE={n_train}  REPEAT={rep+1}/{n_repeats}  (seed={trial_seed})")
            print(f"{'─'*60}")

            for method in methods:
                record = None
                try:
                    record = run_single_size(
                        n_train=int(n_train),
                        method=method,
                        X_train_full=X_train_full,
                        Y_train_full=Y_train_full,
                        P_train_full=P_train,
                        X_val=X_val,
                        Y_val=Y_val,
                        P_val=P_v,
                        X_test=X_test,
                        Y_test=Y_test,
                        P_test=P_t,
                        quick=quick,
                        seed=trial_seed,
                        device=device,
                        pixel_size_mm=pixel_size_mm,
                    )
                    record["repeat"] = int(rep)
                    record["seed"] = int(trial_seed)
                    record["dataset"] = dataset_type

                except Exception as e:
                    print(f"    ✗ {method} n={n_train} rep={rep} FAILED: {e}")
                    record = {
                        "method": method,
                        "n_train": int(n_train),
                        "repeat": int(rep),
                        "seed": int(trial_seed),
                        "dataset": dataset_type,
                        "error": str(e),
                    }

                finally:
                    # ALWAYS append + checkpoint after every experiment
                    all_results.append(record)
                    _save_checkpoint(all_results, out_dir)

    results_path = out_dir / "data_size_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results saved to {results_path}")
    return all_results


# ======================================================================
# Summary + plots
# ======================================================================

def summarise_results(all_results: List[Dict], results_dir: Path, dataset_type: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = results_dir / "data_size_scaling" / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = [r for r in all_results if "error" not in r]
    if not valid:
        print("  No valid results to summarise.")
        return

    methods = sorted(set(r["method"] for r in valid))
    sizes = sorted(set(r["n_train"] for r in valid))

    summary: Dict[str, Dict[int, Dict[str, float]]] = {}
    for m in methods:
        summary[m] = {}
        for n in sizes:
            trials = [r for r in valid if r["method"] == m and r["n_train"] == n]
            if not trials:
                continue
            mae_L = np.array([t["mae_L"] for t in trials], dtype=np.float64)
            mae_th = np.array([t["mae_theta"] for t in trials], dtype=np.float64)
            times = np.array([t["train_time_s"] for t in trials], dtype=np.float64)
            summary[m][n] = {
                "mae_L_mean": float(mae_L.mean()),
                "mae_L_std": float(mae_L.std()),
                "mae_theta_mean": float(mae_th.mean()),
                "mae_theta_std": float(mae_th.std()),
                "time_mean": float(times.mean()),
                "time_std": float(times.std()),
                "n_trials": float(len(trials)),
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'═'*80}")
    print(f"  DATA-SIZE SCALING SUMMARY — {dataset_type.upper()}")
    print(f"{'═'*80}")
    header = f"{'Method':<20} {'N':>6}  {'MAE(L) mm':>14}  {'MAE(θ) deg':>14}  {'Time (s)':>10}"
    print(header)
    print("─" * len(header))
    for m in methods:
        for n in sizes:
            if n not in summary.get(m, {}):
                continue
            s = summary[m][n]
            print(
                f"{m:<20} {n:>6}  "
                f"{s['mae_L_mean']:>6.4f} ± {s['mae_L_std']:<5.4f}  "
                f"{s['mae_theta_mean']:>6.3f} ± {s['mae_theta_std']:<5.3f}  "
                f"{s['time_mean']:>8.1f}"
            )

    # Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Data-Size Scaling — {dataset_type}", fontsize=14, y=1.02)

    def plot_metric(ax, key_mean: str, key_std: str, ylabel: str):
        for m in methods:
            if m not in summary:
                continue
            ns = sorted(summary[m].keys())
            vals = [summary[m][n][key_mean] for n in ns]
            stds = [summary[m][n][key_std] for n in ns]
            ax.errorbar(ns, vals, yerr=stds, capsize=3, marker="o", linestyle="-", label=m)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    plot_metric(axes[0], "mae_L_mean", "mae_L_std", "MAE Length (mm)")
    plot_metric(axes[1], "mae_theta_mean", "mae_theta_std", "MAE Angle (°)")
    plot_metric(axes[2], "time_mean", "time_std", "Train Time (s)")
    axes[0].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig_path = out_dir / "learning_curves.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Learning-curve figure saved to {fig_path}")


# ======================================================================
# Default size schedule
# ======================================================================

def default_sizes(N_full: int, quick: bool) -> List[int]:
    if quick:
        candidates = [10, 50, 200, 1000, N_full]
    else:
        candidates = [10, 100, 1000, 10000, N_full]
    return sorted(set(int(n) for n in candidates if int(n) <= int(N_full)))


# ======================================================================
# Entry
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Data-size scaling (raw mmap loader)")
    parser.add_argument("--data_root", type=Path, default = "/workspaces/surface_spline_prob/src/ut_baseline/1u376lnlam8ac2lf0uato86zrr")#required=True)
    parser.add_argument("--results_dir", type=Path, default=Path("./ut_results_segregated_perspective_metric_space_hmm_final"))#required=True)
    parser.add_argument("--datasets", default=["speed_vary","standard"], choices=["standard","speed_vary"])
    parser.add_argument("--methods", nargs="+", default=["spectral","spectral_linear", "cnn", "rf", "svr", "fourier_rf", "wavelet_rf"])
    parser.add_argument("--sizes", nargs="*", type=int, default=None) 
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.results_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(args.datasets,list):
        for dataset in args.datasets:
            # Determine sizes
            if args.sizes is None or len(args.sizes) == 0:
                dm_tmp = RawBristolMMap(root=args.data_root, dataset_type=dataset, seed=args.seed).load()
                assert dm_tmp.splits is not None
                # full train length is sim_train split length after split
                N_full = dm_tmp.splits.sim_train.shape[0]
                sizes = default_sizes(N_full, args.quick)
            else:
                sizes = sorted(set(int(x) for x in args.sizes))
            

            # Save run config
            cfg = {
                "data_root": str(args.data_root),
                "results_dir": str(args.results_dir),
                "dataset": dataset,
                "methods": args.methods,
                "sizes": sizes,
                "n_repeats": args.n_repeats,
                "quick": args.quick,
                "seed": args.seed,
                "device": args.device,
            }
            out_dir = args.results_dir / "data_size_scaling" / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

            all_results = run_data_size_experiment(
                data_root=args.data_root,
                results_dir=args.results_dir,
                dataset_type=dataset,
                methods=args.methods,
                sizes=sizes,
                n_repeats=args.n_repeats,
                quick=args.quick,
                seed=args.seed,
                device=args.device,
            )

            summarise_results(all_results, args.results_dir, dataset)

            print("\n" + "═" * 70)
            print("  DATA-SIZE SCALING EXPERIMENT COMPLETE")
            print("═" * 70)


if __name__ == "__main__":
    main()
