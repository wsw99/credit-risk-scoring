"""Tabular deep learning models for credit risk scoring.

Implements:
  - TabNet (via pytorch_tabnet): attention-based feature selection
  - FT-Transformer: feature tokenizer + transformer encoder
  - EBM (via interpretml): glass-box interpretable boosting
  - NODE (optional): differentiable oblivious decision trees

All models expose a uniform sklearn-style interface (fit / predict_proba).
GPU-aware with automatic fallback to CPU.
"""

from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data preprocessing for DL models
# ---------------------------------------------------------------------------

class TabularPreprocessor:
    """Standardize numeric features and label-encode categoricals for DL models.

    DL models on tabular data require:
      - Standardized numeric features (zero mean, unit variance)
      - Integer-encoded categorical features
      - Missing value imputation
    """

    def __init__(self):
        self.num_scaler = StandardScaler()
        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
    ) -> "TabularPreprocessor":
        self.num_cols = [c for c in num_cols if c in X.columns]
        self.cat_cols = [c for c in cat_cols if c in X.columns]

        # Fit numeric scaler
        if self.num_cols:
            X_num = X[self.num_cols].fillna(X[self.num_cols].median())
            self.num_scaler.fit(X_num)

        # Fit categorical encoders
        for c in self.cat_cols:
            le = LabelEncoder()
            vals = X[c].fillna("__MISSING__").astype(str)
            le.fit(vals)
            self.cat_encoders[c] = le

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return float32 numpy array ready for DL model input."""
        parts = []

        if self.num_cols:
            X_num = X[self.num_cols].fillna(X[self.num_cols].median())
            parts.append(self.num_scaler.transform(X_num).astype(np.float32))

        if self.cat_cols:
            X_cat = np.zeros((len(X), len(self.cat_cols)), dtype=np.float32)
            for i, c in enumerate(self.cat_cols):
                vals = X[c].fillna("__MISSING__").astype(str)
                encoded = np.zeros(len(X), dtype=np.float32)
                for j, v in enumerate(vals):
                    le = self.cat_encoders[c]
                    if v in le.classes_:
                        encoded[j] = float(le.transform([v])[0])
                    else:
                        encoded[j] = float(len(le.classes_))  # unseen → last+1
                X_cat[:, i] = encoded
            parts.append(X_cat)

        if len(parts) == 1:
            return parts[0]
        return np.column_stack(parts).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> np.ndarray:
        self.fit(X, num_cols, cat_cols)
        return self.transform(X)

    def input_dim(self) -> int:
        return len(self.num_cols) + len(self.cat_cols)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# FT-Transformer (clean PyTorch implementation)
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Map each scalar feature to a d_token-dimensional embedding."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        # out: (B, F, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular data.

    Architecture (simplified from Gorishniy et al. NeurIPS 2021):
      1. FeatureTokenizer — each feature → d_token embedding
      2. Append [CLS] token
      3. Transformer encoder layers
      4. MLP head on [CLS] output

    Args:
        n_features: number of input features
        d_token: embedding dimension per feature
        n_blocks: number of transformer layers
        n_heads: attention heads
        attn_dropout: attention dropout
        ffn_dropout: feedforward dropout
        d_ffn_factor: multiplier for FFN hidden dim (= factor * d_token)
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        d_ffn_factor: float = 4.0/3.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * d_ffn_factor),
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        self.cls_ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token * 2, d_token),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Tokenize
        tokens = self.tokenizer(x)  # (B, F, d_token)
        # Append CLS
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+F, d_token)
        # Transformer
        tokens = self.transformer(tokens)
        # CLS output
        cls_out = tokens[:, 0, :]  # (B, d_token)
        cls_out = self.cls_ffn(cls_out)
        return self.head(cls_out).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# DL trainer
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict] = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class DeepModelWrapper:
    """Wrap a PyTorch model with sklearn-style fit/predict_proba interface.

    Handles training loop, early stopping, device management, and
    probability calibration via sigmoid.
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: TabularPreprocessor,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        epochs: int = 100,
        patience: int = 20,
        device: Optional[str] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_auc": []}
        self.fit_time = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        t0 = time.time()

        # Imbalanced class weight via pos_weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        early_stop = EarlyStopping(patience=self.patience)

        X_tr = torch.tensor(X_train.astype(np.float32))
        y_tr = torch.tensor(y_train.astype(np.float32))
        ds_train = TensorDataset(X_tr, y_tr)
        dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if X_val is not None:
            X_va_t = torch.tensor(X_val.astype(np.float32))  # keep on CPU, batch to GPU
            ds_val = TensorDataset(X_va_t)
            dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in dl_train:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            avg_loss = total_loss / len(X_train)
            self.history["train_loss"].append(avg_loss)

            if X_val is not None:
                self.model.eval()
                val_logits_list = []
                with torch.no_grad():
                    for (batch_x,) in dl_val:
                        batch_x = batch_x.to(self.device)
                        val_logits_list.append(self.model(batch_x).cpu())
                val_preds = torch.sigmoid(torch.cat(val_logits_list)).numpy()
                val_auc = roc_auc_score(y_val, val_preds)
                self.history["val_auc"].append(val_auc)
                score = val_auc
            else:
                score = -avg_loss

            scheduler.step()
            if early_stop(score, self.model):
                break

        early_stop.restore(self.model)
        self.fit_time = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X.astype(np.float32))
        dl = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        logits_list = []
        with torch.no_grad():
            for (batch_x,) in dl:
                logits_list.append(self.model(batch_x.to(self.device)).cpu())
        return torch.sigmoid(torch.cat(logits_list)).numpy()

    def save(self, path: str | Path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "preprocessor": self.preprocessor,
            "history": self.history,
            "config": {
                "lr": self.lr, "weight_decay": self.weight_decay,
                "batch_size": self.batch_size, "epochs": self.epochs,
                "patience": self.patience,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path, model: nn.Module) -> "DeepModelWrapper":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        wrapper = cls.__new__(cls)
        wrapper.model = model
        wrapper.preprocessor = ckpt["preprocessor"]
        wrapper.history = ckpt.get("history", {})
        cfg = ckpt["config"]
        wrapper.lr = cfg["lr"]
        wrapper.weight_decay = cfg["weight_decay"]
        wrapper.batch_size = cfg["batch_size"]
        wrapper.epochs = cfg["epochs"]
        wrapper.patience = cfg["patience"]
        wrapper.device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper.fit_time = 0.0
        wrapper.model.to(wrapper.device)
        return wrapper


# ---------------------------------------------------------------------------
# TabNet wrapper (via pytorch_tabnet)
# ---------------------------------------------------------------------------

class TabNetWrapper:
    """Wrapper around pytorch_tabnet.TabNetClassifier.

    Provides sklearn-style fit/predict_proba with sensible credit-scoring defaults.
    """

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        lr: float = 2e-2,
        batch_size: int = 2048,
        virtual_batch_size: int = 256,
        epochs: int = 100,
        patience: int = 20,
        device: Optional[str] = None,
    ):
        self.config = {k: v for k, v in locals().items() if k not in ("self", "device")}
        self.config["device"] = device
        self._model = None
        self._preprocessor = TabularPreprocessor()
        self.fit_time = 0.0
        self.history: Dict[str, List[float]] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ):
        from pytorch_tabnet.tab_model import TabNetClassifier

        if num_cols is None:
            num_cols = [c for c in X_train.columns if X_train[c].dtype in ("float64", "float32", "int64", "int32")]
        if cat_cols is None:
            cat_cols = [c for c in X_train.columns if X_train[c].dtype in ("object", "category")]

        self._preprocessor.fit(X_train, num_cols, cat_cols)
        X_tr = self._preprocessor.transform(X_train)
        y_tr = y_train.values.astype(int)

        X_va_np = self._preprocessor.transform(X_val) if X_val is not None else None
        y_va_np = y_val.values.astype(int) if y_val is not None else None

        cat_idxs = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        cat_dims = [int(X_train[c].nunique()) + 2 for c in cat_cols] if cat_cols else []

        device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TabNetClassifier(
            n_d=self.config["n_d"],
            n_a=self.config["n_a"],
            n_steps=self.config["n_steps"],
            gamma=self.config["gamma"],
            n_independent=self.config["n_independent"],
            n_shared=self.config["n_shared"],
            lambda_sparse=self.config["lambda_sparse"],
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.config["lr"]),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params=dict(T_max=self.config["epochs"]),
            mask_type="sparsemax",
            device_name=device,
            verbose=0,
        )

        t0 = time.time()
        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_va_np, y_va_np)] if X_va_np is not None else None,
            eval_metric=["auc"],
            max_epochs=self.config["epochs"],
            patience=self.config["patience"],
            batch_size=self.config["batch_size"],
            virtual_batch_size=self.config["virtual_batch_size"],
        )
        self.fit_time = time.time() - t0
        self.history = self._model.history.history if hasattr(self._model, "history") else {}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_np = self._preprocessor.transform(X)
        return self._model.predict_proba(X_np)[:, 1]

    def save(self, path: str | Path):
        path = Path(path)
        self._model.save_model(str(path.with_suffix("")))
        self._preprocessor.save(str(path.parent / f"{path.stem}_preprocessor.pkl"))
        with open(path, "wb") as f:
            pickle.dump({"config": self.config, "fit_time": self.fit_time, "history": self.history}, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabNetWrapper":
        from pytorch_tabnet.tab_model import TabNetClassifier

        path = Path(path)
        with open(path, "rb") as f:
            meta = pickle.load(f)

        wrapper = cls.__new__(cls)
        wrapper.config = meta["config"]
        wrapper.fit_time = meta.get("fit_time", 0.0)
        wrapper.history = meta.get("history", {})
        wrapper._model = TabNetClassifier()
        wrapper._model.load_model(str(path.with_suffix("")))
        pp_path = path.parent / f"{path.stem}_preprocessor.pkl"
        wrapper._preprocessor = TabularPreprocessor.load(pp_path)
        return wrapper


# ---------------------------------------------------------------------------
# EBM wrapper (via interpretml)
# ---------------------------------------------------------------------------

class ExplainableBoostingMachine:
    """Wrapper around interpret.glassbox.ExplainableBoostingClassifier.

    EBM is a glass-box model: accuracy close to LightGBM + full interpretability
    via per-feature additive contribution functions.
    """

    def __init__(
        self,
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        interactions: Union[int, float] = 10,
        learning_rate: float = 0.01,
        max_rounds: int = 5000,
        early_stopping_rounds: int = 50,
        inner_bags: int = 25,
        outer_bags: int = 1,
        random_state: int = 42,
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self._model = None
        self._feature_names: List[str] = []
        self.fit_time = 0.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ):
        from interpret.glassbox import ExplainableBoostingClassifier

        self._feature_names = list(X_train.columns)
        self._model = ExplainableBoostingClassifier(**self.config)

        t0 = time.time()
        self._model.fit(X_train, y_train)
        self.fit_time = time.time() - t0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def explain_global(self, X: pd.DataFrame) -> "pd.DataFrame":
        """Return per-feature importance from EBM's native explainer."""
        ebm_global = self._model.explain_global()
        rows = []
        for feat_name, importance in zip(ebm_global.data()["names"], ebm_global.data()["scores"]):
            rows.append({"feature": feat_name, "importance": importance})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    def explain_local(self, X: pd.DataFrame) -> "pd.DataFrame":
        """Return per-sample per-feature additive contribution."""
        ebm_local = self._model.explain_local(X, y=None)
        contribs = {}
        for i, name in enumerate(self._feature_names):
            scores = ebm_local.data(i)["scores"]
            contribs[name] = scores
        return pd.DataFrame(contribs, index=X.index)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ExplainableBoostingMachine":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Model-specific creation functions
# ---------------------------------------------------------------------------

def create_ft_transformer(
    n_features: int,
    d_token: int = 64,
    n_blocks: int = 3,
    n_heads: int = 8,
    attn_dropout: float = 0.15,
    ffn_dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    epochs: int = 150,
    patience: int = 25,
) -> DeepModelWrapper:
    """Create an FT-Transformer model with sensible defaults for credit scoring."""
    model = FTTransformer(
        n_features=n_features,
        d_token=d_token,
        n_blocks=n_blocks,
        n_heads=n_heads,
        attn_dropout=attn_dropout,
        ffn_dropout=ffn_dropout,
    )
    return DeepModelWrapper(
        model, TabularPreprocessor(),
        lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, epochs=epochs, patience=patience,
    )


def create_tabnet(
    n_d: int = 32,
    n_a: int = 32,
    n_steps: int = 5,
    gamma: float = 1.5,
    lr: float = 2e-2,
    batch_size: int = 2048,
    epochs: int = 100,
    patience: int = 20,
) -> TabNetWrapper:
    """Create a TabNet model with sensible defaults."""
    return TabNetWrapper(
        n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
        lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
    )


def create_ebm(
    learning_rate: float = 0.01,
    max_rounds: int = 5000,
    early_stopping_rounds: int = 50,
) -> ExplainableBoostingMachine:
    """Create an EBM model with sensible defaults."""
    return ExplainableBoostingMachine(
        learning_rate=learning_rate,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )


# ---------------------------------------------------------------------------
# Utility: unified predict for any model type
# ---------------------------------------------------------------------------

def predict_proba_dl(model: Any, X) -> np.ndarray:
    """Unified predict_proba for all model types (tree, DL, EBM, TabNet).

    Returns 1-D array of positive-class probabilities.
    """
    if isinstance(X, np.ndarray):
        pass  # keep as array
    elif hasattr(X, "values"):
        # pandas DataFrame
        pass

    result = model.predict_proba(X)
    if isinstance(result, np.ndarray) and result.ndim == 1:
        return result
    if isinstance(result, np.ndarray) and result.ndim == 2:
        return result[:, 1]
    try:
        if hasattr(result, "ndim") and result.ndim == 1:
            return result
        return result[:, 1]
    except Exception:
        return result
