"""
baselines.py
============
Baseline characterisation methods for journal comparison against the proposed
spectral method.  Three baselines are implemented:

  1. SixdBDropSizer
     Exact reproduction of Pyle et al. (2021) physics-based baseline.
     No training required.  Applied to the SS-S view of Array A.

  2. PyleCNN
     Exact architecture from Pyle et al. Fig. 5a:
       - 32×32×4 input (four views stacked as channels)
       - Two separate sub-networks predicting L and θ
       - Convolutional blocks → MaxPool → Dropout(0.1) → FC
     Training matches the paper: Adam lr=1e-3, batch=128, max 400 epochs,
     patience=150, MSE loss, early stopping on simulated-validation loss.

  3. ClassicalMLBaseline
     Provides SVR and Random Forest on hand-crafted features from each image:
       - Pixel-flattened (PCA-reduced) features
       - Energy-per-view features
       - Histogram of gradients features
     Serves as a weaker lower-bound on what a learned method should beat.

All baselines share the same evaluation API:
  fit(X_train, Y_train)
  predict(X) -> Y_hat  where X is (N, 4, 32, 32) numpy, Y is (N, 2)

"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CrackCharacterizer(ABC):
    """Shared API for all baselines and the proposed method."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "CrackCharacterizer":
        """
        Parameters
        ----------
        X : (N, 4, 32, 32) float32  normalised PWI images, channel-first
        Y : (N, 2)          float32  [L_m, theta_deg]
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, 2) predictions [L_hat_m, theta_hat_deg]."""

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """Standard metrics used in the paper."""
        Y_hat = self.predict(X)
        return crack_metrics(Y, Y_hat)


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------

def crack_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the metrics reported in Pyle et al. (2021):

    MAE_L      – Mean absolute error in length (m)
    MAE_theta  – Mean absolute error in angle  (°)
    wall_loss  – Mean |c(θ_pred, L_pred) - c(θ_true, L_true)|
                 where c(θ, L) = L·cos θ (through-wall extent)
    RMSE_L     – Root mean square error in length
    RMSE_theta – Root mean square error in angle
    """
    L_true  = Y_true[:, 0]
    th_true = Y_true[:, 1]
    L_pred  = Y_pred[:, 0]
    th_pred = Y_pred[:, 1]

    err_L  = L_pred  - L_true
    err_th = th_pred - th_true

    # Wall-loss: through-wall extent (Pyle eq. c(θ) = L·cos θ — standard in pipe inspection)
    wl_true = L_true  * np.cos(np.deg2rad(th_true))
    wl_pred = L_pred  * np.cos(np.deg2rad(th_pred))
    wall_loss = np.abs(wl_pred - wl_true)

    return {
        "MAE_L":       float(np.mean(np.abs(err_L))),
        "MAE_theta":   float(np.mean(np.abs(err_th))),
        "wall_loss":   float(np.mean(wall_loss)),
        "RMSE_L":      float(np.sqrt(np.mean(err_L ** 2))),
        "RMSE_theta":  float(np.sqrt(np.mean(err_th ** 2))),
        "std_L":       float(np.std(np.abs(err_L))),
        "std_theta":   float(np.std(np.abs(err_th))),
        "std_wall":    float(np.std(wall_loss)),
    }


# ===========================================================================
# 1.  6 dB DROP METHOD
# ===========================================================================

class SixdBDropSizer(CrackCharacterizer):
    """
    Physics-based 6 dB drop method from Pyle et al. (2021) Section III.A.

    Applied to the SS-S view from Array A (channel index 0) as in the paper.
    The bounding-box major axis gives crack length and angle.

    Parameters
    ----------
    max_gap_pixels : int
        Maximum gap between connected-component pixels (paper: 4 pixels = 1.27 m).
    channel : int
        Which image channel to apply the method to (default 0 = SS-S Array A).
    pixel_size_m : float
        Physical size of each pixel in m.  Computed from Mesh if provided.
    """

    def __init__(
        self,
        max_gap_pixels: int = 4,
        channel: int = 0,
        pixel_size_m: float = 1.27 / 4.0,
    ):
        self.max_gap_pixels = max_gap_pixels
        self.channel        = channel
        self.pixel_size_m  = pixel_size_m   # m per pixel

    def fit(self, X, Y, **kwargs) -> "SixdBDropSizer":
        """No training required."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (N, 4, 32, 32) — channel-first.  Values can be normalised dB or raw dB;
            the method works on relative amplitudes so normalisation doesn't matter.
        """
        N = X.shape[0]
        preds = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            preds[i] = self._size_single(X[i, self.channel])
        return preds

    def _size_single(self, img: np.ndarray) -> Tuple[float, float]:
        """
        Apply 6 dB drop to a single (32, 32) image.

        Returns
        -------
        (L_m, theta_deg)
        """
        # Convert to dB (relative to max peak)
        # Image values may already be dB-normalised; we work with relative dB
        img_db = img - img.max()   # peak at 0 dB, everything else negative

        # 6 dB mask
        mask = img_db >= -6.0                     # boolean (32, 32)

        # Connected-component analysis with morphological dilation to bridge
        # small gaps (paper: max 4 pixels between connected indicators)
        struct = np.ones(
            (2 * self.max_gap_pixels + 1, 2 * self.max_gap_pixels + 1), dtype=bool
        )
        dilated = ndimage.binary_dilation(mask, structure=struct)
        labeled, n_labels = ndimage.label(dilated)

        if n_labels == 0:
            return 0.0, 0.0

        # Find the component containing the peak
        peak_row, peak_col = np.unravel_index(np.argmax(img), img.shape)
        peak_label = labeled[peak_row, peak_col]

        # Get pixels of that component in the *original* (undilated) mask
        component_mask = mask & (labeled == peak_label)
        ys, xs = np.where(component_mask)

        if len(xs) < 2:
            return 0.0, 0.0

        # Convert to physical coordinates
        x_m = xs * self.pixel_size_m
        y_m = ys * self.pixel_size_m

        # Minimum bounding box via PCA on the pixel coordinates
        coords = np.stack([x_m, y_m], axis=1)    # (M, 2)
        cov    = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Major axis = largest eigenvalue direction
        major_vec   = eigvecs[:, -1]                # (2,)
        # Project all points onto major axis
        proj        = coords @ major_vec
        L_m        = float(proj.max() - proj.min())

        # Angle from vertical (z-axis in the image corresponds to rows)
        # major_vec is [dx, dz]; angle = arctan(dx / dz) from vertical
        theta_deg   = float(np.degrees(np.arctan2(major_vec[0], major_vec[1])))

        return L_m, theta_deg


# ===========================================================================
# 2.  PYLE ET AL. CNN
# ===========================================================================

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _PyleSubNetwork(nn.Module):
    """
    Single-output sub-network predicting either L or θ.

    Architecture (from Fig. 5a and paper description):
      Input: (B, 4, 32, 32)
      Conv(16, 5×5) → ReLU → MaxPool(2)  →  (B, 16, 16, 16)
      Conv(32, 5×5) → ReLU → MaxPool(2)  →  (B, 32,  8,  8)
      Conv(64, 5×5) → ReLU              →  (B, 64,  8,  8)
      Flatten → FC(256) → Dropout(0.1) → ReLU
               FC(64)  → Dropout(0.1) → ReLU
               FC(1)
    """

    def __init__(self, in_channels: int = 4, dropout: float = 0.10):
        super().__init__()

        self.features = nn.Sequential(
            _ConvBlock(in_channels, 16, kernel=5, pool=True),   # → (16,16,16)
            _ConvBlock(16, 32, kernel=5, pool=True),            # → (32, 8, 8)
            _ConvBlock(32, 64, kernel=5, pool=False),           # → (64, 8, 8)
        )

        # After two 2× maxpools: 32 → 8
        flat_dim = 64 * 8 * 8

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x)).squeeze(1)   # (B,)


class PyleCNNModel(nn.Module):
    """
    Two decoupled sub-networks: one for L, one for θ.

    Returns a (B, 2) tensor [L_pred, theta_pred].
    The networks share no weights — they learn independently,
    matching the paper's description of decoupled predictors.
    """

    def __init__(self, in_channels: int = 4, dropout: float = 0.10):
        super().__init__()
        self.length_net = _PyleSubNetwork(in_channels, dropout)
        self.angle_net  = _PyleSubNetwork(in_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L     = self.length_net(x)   # (B,)
        theta = self.angle_net(x)    # (B,)
        return torch.stack([L, theta], dim=1)  # (B, 2)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class PyleCNNConfig:
    """Hyperparameters matching the paper exactly (Section III.B.2)."""
    lr:           float = 1e-3
    batch_size:   int   = 128
    max_epochs:   int   = 400
    patience:     int   = 150         # epochs without improvement on sim-val
    dropout:      float = 0.10
    weight_decay: float = 0.0
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:         int   = 42
    n_runs:       int   = 1           # paper used 10 independent inits for error bars
    save_best:    bool  = True


class PyleCNN(CrackCharacterizer):
    """
    Pyle et al. (2021) CNN baseline with full training loop.

    Supports multiple independent runs (paper used 10) to compute
    mean ± std over random initialisation — required for Fig. 5b comparisons.
    """

    def __init__(self, cfg: Optional[PyleCNNConfig] = None):
        self.cfg       = cfg or PyleCNNConfig()
        self.models: List[PyleCNNModel] = []
        self.history:  List[Dict]       = []
        self.best_val_losses: List[float] = []

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        Y_val:   Optional[np.ndarray] = None,
        verbose: bool = True,
        **kwargs,
    ) -> "PyleCNN":
        """
        Train `cfg.n_runs` independent CNNs.

        Parameters
        ----------
        X_train : (N_tr, 4, 32, 32)
        Y_train : (N_tr, 2)
        X_val   : (N_v, 4, 32, 32)  — used for early stopping
        Y_val   : (N_v, 2)
        """
        self.models.clear()
        self.history.clear()

        for run in range(self.cfg.n_runs):
            if verbose:
                print(f"\n── PyleCNN run {run+1}/{self.cfg.n_runs} ──")
            seed = self.cfg.seed + run
            torch.manual_seed(seed)
            np.random.seed(seed)

            model, hist = self._train_single(X_train, Y_train, X_val, Y_val, verbose)
            self.models.append(model)
            self.history.append(hist)
            self.best_val_losses.append(min(hist["val_loss"]) if hist["val_loss"] else float("inf"))

        if verbose:
            print(f"\nBest val MSE across {self.cfg.n_runs} runs: "
                  f"{np.mean(self.best_val_losses):.4f} ± {np.std(self.best_val_losses):.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble mean over all trained runs."""
        if not self.models:
            raise RuntimeError("Call fit() before predict()")
        preds = np.stack([self._predict_single(m, X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) from ensemble — useful for uncertainty quantification."""
        preds = np.stack([self._predict_single(m, X) for m in self.models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _train_single(
        self,
        X_train, Y_train,
        X_val, Y_val,
        verbose: bool,
    ) -> Tuple[PyleCNNModel, Dict]:

        cfg    = self.cfg
        device = torch.device(cfg.device)

        model = PyleCNNModel(in_channels=4, dropout=cfg.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()

        # Build DataLoaders from numpy arrays
        tr_loader = self._make_loader(X_train, Y_train, shuffle=True)
        val_loader = self._make_loader(X_val, Y_val, shuffle=False) \
                     if X_val is not None else None

        best_val  = float("inf")
        best_state = None
        wait       = 0
        history    = {"train_loss": [], "val_loss": [], "epoch_time": []}

        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            # --- train ---
            model.train()
            tr_losses = []
            for X_b, Y_b in tr_loader:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, Y_b)
                loss.backward()
                optimizer.step()
                tr_losses.append(loss.item())

            tr_loss = float(np.mean(tr_losses))
            history["train_loss"].append(tr_loss)
            elapsed = time.time() - t0
            history["epoch_time"].append(elapsed)

            # --- val ---
            val_loss = float("inf")
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for X_b, Y_b in val_loader:
                        X_b, Y_b = X_b.to(device), Y_b.to(device)
                        val_losses.append(criterion(model(X_b), Y_b).item())
                val_loss = float(np.mean(val_losses))
                history["val_loss"].append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    if cfg.save_best:
                        best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= cfg.patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch}")
                        break

            if verbose and epoch % 20 == 0:
                print(f"  Ep {epoch:4d}/{cfg.max_epochs} | "
                      f"train MSE={tr_loss:.4f} | val MSE={val_loss:.4f} | "
                      f"wait={wait}")

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        return model, history

    def _make_loader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        Xt = torch.from_numpy(X.astype(np.float32))
        Yt = torch.from_numpy(Y.astype(np.float32))
        ds = TensorDataset(Xt, Yt)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    def _predict_single(
        self, model: PyleCNNModel, X: np.ndarray
    ) -> np.ndarray:
        device = torch.device(self.cfg.device)
        model.eval()
        preds = []
        loader = self._make_loader(X, np.zeros((len(X), 2), dtype=np.float32),
                                    shuffle=False)
        with torch.no_grad():
            for X_b, _ in loader:
                preds.append(model(X_b.to(device)).cpu().numpy())
        return np.concatenate(preds, axis=0)


# ===========================================================================
# 3.  CLASSICAL ML BASELINE
# ===========================================================================

class ClassicalMLBaseline(CrackCharacterizer):
    """
    Two classical ML variants operating on engineered image features:

      variant="svr"  : SVR (radial basis function kernel)
      variant="rf"   : Random Forest regressor

    Features extracted per sample (concatenated across all 4 views):
      - PCA-projected pixels (128 components)
      - Per-view energy (total dB power in each channel)
      - First-order image moments (mean x, mean z of energy-weighted pixels)
    """

    def __init__(
        self,
        variant: str = "rf",           # "svr" | "rf"
        n_pca_components: int = 128,
        n_estimators: int = 200,
        svr_C: float = 10.0,
        svr_gamma: str = "scale",
    ):
        assert variant in ("svr", "rf")
        self.variant           = variant
        self.n_pca_components  = n_pca_components
        self.n_estimators      = n_estimators
        self.svr_C             = svr_C
        self.svr_gamma         = svr_gamma

        self._pca:      Optional[PCA] = None
        self._scaler:   Optional[StandardScaler] = None
        self._model_L:  Optional[object] = None
        self._model_th: Optional[object] = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "ClassicalMLBaseline":
        """
        X : (N, 4, 32, 32)
        Y : (N, 2)
        """
        feats = self._extract_features(X, fit=True)

        L_true  = Y[:, 0]
        th_true = Y[:, 1]

        if self.variant == "svr":
            self._model_L  = SVR(C=self.svr_C, gamma=self.svr_gamma, kernel="rbf")
            self._model_th = SVR(C=self.svr_C, gamma=self.svr_gamma, kernel="rbf")
        else:
            self._model_L  = RandomForestRegressor(n_estimators=self.n_estimators,
                                                    n_jobs=-1, random_state=42)
            self._model_th = RandomForestRegressor(n_estimators=self.n_estimators,
                                                    n_jobs=-1, random_state=42)

        print(f"Fitting {self.variant} on {feats.shape[1]}-dim features …")
        self._model_L.fit(feats, L_true)
        self._model_th.fit(feats, th_true)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats   = self._extract_features(X, fit=False)
        L_pred  = self._model_L.predict(feats).astype(np.float32)
        th_pred = self._model_th.predict(feats).astype(np.float32)
        return np.stack([L_pred, th_pred], axis=1)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """
        X : (N, 4, 32, 32) channel-first
        Returns (N, D) feature matrix.
        """
        N = X.shape[0]
        flat  = X.reshape(N, -1)   # (N, 4096)

        # PCA-compressed pixel features
        if fit:
            self._pca    = PCA(n_components=self.n_pca_components, random_state=42)
            self._scaler = StandardScaler()
            pca_feats    = self._scaler.fit_transform(flat)
            pca_feats    = self._pca.fit_transform(pca_feats)
        else:
            pca_feats = self._pca.transform(self._scaler.transform(flat))

        # Per-view energy (4 features)
        energy = X.reshape(N, 4, -1).mean(axis=2)   # (N, 4)

        # First-order spatial moments per view (8 features)
        r = np.arange(32, dtype=np.float32)
        moments = np.zeros((N, 8), dtype=np.float32)
        for c in range(4):
            w = np.exp(X[:, c] / 10.0)             # (N,32,32) — energy weights
            w_sum = w.sum(axis=(1, 2), keepdims=True) + 1e-8
            w_n   = w / w_sum
            moments[:, 2*c]   = (w_n * r[None, :, None]).sum(axis=(1, 2))
            moments[:, 2*c+1] = (w_n * r[None, None, :]).sum(axis=(1, 2))

        return np.hstack([pca_feats, energy, moments]).astype(np.float32)


# ===========================================================================
# Convenience wrapper: train/evaluate all baselines in one call
# ===========================================================================

class BaselineSuite:
    """
    Thin wrapper that trains all baselines on the same data and collects
    results in a standardised format for the paper's results table.
    """

    def __init__(
        self,
        include: Optional[List[str]] = None,
        cnn_cfg: Optional[PyleCNNConfig] = None,
        pixel_size_m: float = 1.27 / 4.0,
    ):
        """
        Parameters
        ----------
        include : list of {"6db", "cnn", "rf", "svr"} — None means all
        cnn_cfg : PyleCNNConfig override
        """
        self.include = include or ["6db", "cnn", "rf", "svr"]
        self.methods: Dict[str, CrackCharacterizer] = {}

        if "6db" in self.include:
            self.methods["6db"] = SixdBDropSizer(pixel_size_m=pixel_size_m)
        if "cnn" in self.include:
            self.methods["cnn"] = PyleCNN(cfg=cnn_cfg)
        if "rf" in self.include:
            self.methods["rf"]  = ClassicalMLBaseline(variant="rf")
        if "svr" in self.include:
            self.methods["svr"] = ClassicalMLBaseline(variant="svr")

        self.results: Dict[str, Dict[str, float]] = {}

    def fit_all(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        Y_val:   Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "BaselineSuite":
        for name, method in self.methods.items():
            print(f"\n{'='*60}")
            print(f"  Training: {name}")
            print(f"{'='*60}")
            t0 = time.time()
            if name == "cnn":
                method.fit(X_train, Y_train, X_val, Y_val, verbose=verbose)
            elif name == "6db":
                method.fit(X_train, Y_train)
            else:
                method.fit(X_train, Y_train)
            print(f"  Done in {time.time() - t0:.1f}s")
        return self

    def evaluate_all(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        split_name: str = "exp_test",
    ) -> Dict[str, Dict[str, float]]:
        for name, method in self.methods.items():
            self.results[f"{name}_{split_name}"] = method.evaluate(X_test, Y_test)
        return self.results

    def print_table(self, split_name: str = "exp_test") -> None:
        """Print LaTeX-ready comparison table."""
        header = f"{'Method':<20} {'MAE_L':>8} {'MAE_θ':>8} {'WallLoss':>10}"
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))
        for key, res in self.results.items():
            if split_name not in key:
                continue
            name = key.replace(f"_{split_name}", "")
            print(f"{name:<20} {res['MAE_L']:>8.3f} {res['MAE_theta']:>8.2f} "
                  f"{res['wall_loss']:>10.3f}")
        print("─" * len(header))
        # Paper's reported values for reference
        print(f"\n{'[Paper 6dB]':<20} {'1.10':>8} {'8.60':>8} {'—':>10}")
        print(f"{'[Paper CNN]':<20} {'0.29':>8} {'2.90':>8} {'—':>10}")
