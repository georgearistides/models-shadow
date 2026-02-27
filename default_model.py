"""
DefaultModel — WoE Logistic Regression with Isotonic Calibration
=================================================================
Replaces the 3-sub-model blend (WoE + Rule + XGBoost) with WoE alone.
WoE standalone AUC 0.6559 vs blend 0.6597 — only 0.0038 difference.

Weight of Evidence (WoE) transforms each feature into risk-ordered bins,
then fits a logistic regression on the transformed values. The result is
fully transparent: each feature's contribution = WoE value * coefficient.

Calibration uses 5-fold cross-validated isotonic regression on out-of-fold
predictions (fixes the v1.1-v1.4 in-sample calibration overfitting bug).

Usage:
    from default_model import DefaultModel

    model = DefaultModel()
    model.fit(train_df, y_col="is_bad")

    pd_value = model.predict_pd({"fico": 620, "qi_encoded": 4, ...})
    pd_series = model.predict_pd_batch(test_df)

    model.save("woe_model.joblib")
    model2 = DefaultModel.load("woe_model.joblib")
"""

import logging
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# Expert FICO bins (domain knowledge > data quantiles)
EXPERT_FICO_BINS = [0, 570, 600, 620, 650, 670, 700, 740, 800, 900]

# WoE feature candidates (order matters for IV selection)
WOE_FEATURE_CANDIDATES = [
    "fico", "d30", "d60", "inq6", "revutil", "pastdue",
    "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
    "mopmt", "crhist",
    "qi_encoded", "fico_qi_interaction",
    "paid_ratio", "delin_rate", "bureau_health", "cash_stress",
    "is_repeat", "has_prior_bad",
]

# Minimum Information Value to retain a feature
MIN_IV = 0.02
MAX_FEATURES = 15
N_WOE_BINS = 6  # quantile bins for non-FICO features


class DefaultModel:
    """WoE Logistic Regression with isotonic calibration."""

    def __init__(self):
        self.woe_maps = {}       # feature -> {bin_label: woe_value}
        self.woe_bins = {}       # feature -> bin edges or categories
        self.selected_features = []
        self.iv_scores = {}      # feature -> information value
        self.lr_model = None     # LogisticRegressionCV
        self.calibrator = None   # IsotonicRegression
        self.train_medians = {}  # for imputation
        self.version = "6.0"

    def _compute_woe_bins(self, series, y, feature_name):
        """Compute WoE bins for a single feature.

        Returns: (bin_edges_or_categories, woe_map, iv)
        """
        if feature_name == "fico":
            bins = EXPERT_FICO_BINS
            binned = pd.cut(series, bins=bins, labels=False, include_lowest=True)
        elif series.nunique() <= 10:
            # Categorical or low-cardinality: use natural values
            binned = series.fillna(-999)
            bins = "categorical"
        else:
            # Quantile bins — use qcut to get actual bin edges, deduplicate
            try:
                binned, bin_edges = pd.qcut(series, q=N_WOE_BINS, labels=False,
                                            duplicates="drop", retbins=True)
            except ValueError:
                binned, bin_edges = pd.cut(series, bins=N_WOE_BINS, labels=False,
                                           include_lowest=True, retbins=True)
            # Ensure unique edges and extend range for unseen values
            bin_edges = np.unique(bin_edges)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            bins = bin_edges

        # Compute WoE per bin
        woe_map = {}
        total_goods = (y == 0).sum()
        total_bads = (y == 1).sum()
        iv = 0.0

        for bin_val in sorted(binned.dropna().unique()):
            mask = binned == bin_val
            goods_in_bin = max(((y == 0) & mask).sum(), 0.5)  # Laplace smoothing
            bads_in_bin = max(((y == 1) & mask).sum(), 0.5)

            pct_goods = goods_in_bin / total_goods
            pct_bads = bads_in_bin / total_bads

            woe = np.log(pct_goods / pct_bads)
            woe_map[bin_val] = woe
            iv += (pct_goods - pct_bads) * woe

        return bins, woe_map, iv

    def _apply_woe(self, series, feature_name):
        """Transform a feature series using pre-computed WoE bins."""
        bins = self.woe_bins[feature_name]
        woe_map = self.woe_maps[feature_name]

        # Fill missing with training median
        series = series.fillna(self.train_medians.get(feature_name, 0))

        if feature_name == "fico":
            binned = pd.cut(series, bins=bins, labels=False, include_lowest=True)
        elif isinstance(bins, str) and bins == "categorical":
            binned = series.fillna(-999)
        else:
            binned = pd.cut(series, bins=bins, labels=False, duplicates="drop")

        # Map bins to WoE values, default to 0 for unseen bins
        return binned.map(woe_map).fillna(0.0)

    def fit(self, df, y_col="is_bad"):
        """Fit the WoE model: bin features, select by IV, fit logistic regression, calibrate.

        Args:
            df: Training DataFrame with features and target.
            y_col: Name of binary target column.
        """
        y = df[y_col].values
        available = [f for f in WOE_FEATURE_CANDIDATES if f in df.columns]

        # Store medians for imputation
        for feat in available:
            if df[feat].dtype in [np.float64, np.int64, float, int]:
                self.train_medians[feat] = float(df[feat].median())

        # Step 1: Compute WoE bins and IV for each candidate feature
        logger.info(f"Computing WoE for {len(available)} candidate features...")
        for feat in available:
            series = df[feat].fillna(self.train_medians.get(feat, 0))
            bins, woe_map, iv = self._compute_woe_bins(series, y, feat)
            self.woe_bins[feat] = bins
            self.woe_maps[feat] = woe_map
            self.iv_scores[feat] = iv

        # Step 2: Select features by IV (>= MIN_IV, top MAX_FEATURES)
        ranked = sorted(self.iv_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [
            f for f, iv in ranked if iv >= MIN_IV
        ][:MAX_FEATURES]

        logger.info(f"Selected {len(self.selected_features)} features by IV >= {MIN_IV}:")
        for f in self.selected_features:
            logger.info(f"  {f}: IV={self.iv_scores[f]:.4f}")

        # Step 3: Transform training data to WoE values
        X_woe = pd.DataFrame()
        for feat in self.selected_features:
            X_woe[feat] = self._apply_woe(df[feat], feat)

        # Step 4: Fit logistic regression with CV
        self.lr_model = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=3,
            scoring="roc_auc",
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.lr_model.fit(X_woe.values, y)

        train_proba = self.lr_model.predict_proba(X_woe.values)[:, 1]
        train_auc = roc_auc_score(y, train_proba)
        logger.info(f"Train AUC (uncalibrated): {train_auc:.4f}")

        # Step 5: Calibrate with 5-fold cross-validated isotonic regression
        # (Fixes the v1.1-v1.4 in-sample calibration overfitting bug)
        oof_preds = np.zeros(len(y))
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_woe, y)):
            X_tr, X_val = X_woe.values[tr_idx], X_woe.values[val_idx]
            y_tr = y[tr_idx]

            fold_lr = LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
                cv=3, scoring="roc_auc", penalty="l2",
                solver="lbfgs", max_iter=1000, random_state=42,
            )
            fold_lr.fit(X_tr, y_tr)
            oof_preds[val_idx] = fold_lr.predict_proba(X_val)[:, 1]

        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(oof_preds, y)

        calibrated = self.calibrator.predict(train_proba)
        logger.info(f"Calibration: mean predicted={calibrated.mean():.4f}, actual={y.mean():.4f}")

        return self

    def predict_pd(self, row: dict) -> float:
        """Predict calibrated PD for a single application."""
        X_woe = np.array([
            self.woe_maps[feat].get(
                self._bin_value(row.get(feat, self.train_medians.get(feat, 0)), feat),
                0.0
            )
            for feat in self.selected_features
        ]).reshape(1, -1)

        raw_pd = self.lr_model.predict_proba(X_woe)[:, 1][0]
        return float(self.calibrator.predict([raw_pd])[0])

    def _bin_value(self, value, feature_name):
        """Bin a single value for WoE lookup."""
        value = float(value) if value is not None else self.train_medians.get(feature_name, 0)
        bins = self.woe_bins[feature_name]

        if feature_name == "fico":
            binned = pd.cut([value], bins=bins, labels=False, include_lowest=True)
            return binned[0] if not pd.isna(binned[0]) else 0
        elif isinstance(bins, str) and bins == "categorical":
            return int(value) if not np.isnan(value) else -999
        else:
            binned = pd.cut([value], bins=bins, labels=False, duplicates="drop")
            return binned[0] if not pd.isna(binned[0]) else 0

    def predict_pd_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict calibrated PD for a DataFrame. Returns Series of PD values."""
        X_woe = pd.DataFrame()
        for feat in self.selected_features:
            col = df[feat] if feat in df.columns else pd.Series(
                self.train_medians.get(feat, 0), index=df.index
            )
            X_woe[feat] = self._apply_woe(col, feat)

        raw_pds = self.lr_model.predict_proba(X_woe.values)[:, 1]
        calibrated_pds = self.calibrator.predict(raw_pds)
        return pd.Series(calibrated_pds, index=df.index)

    def evaluate(self, df, y_col="is_bad"):
        """Evaluate model on a test set. Returns dict of metrics."""
        pds = self.predict_pd_batch(df)
        y = df[y_col].values
        auc = roc_auc_score(y, pds)

        return {
            "auc": round(auc, 4),
            "gini": round(2 * auc - 1, 4),
            "mean_pd": round(float(pds.mean()), 4),
            "actual_bad_rate": round(float(y.mean()), 4),
            "pred_actual_ratio": round(float(pds.mean() / max(y.mean(), 0.001)), 2),
            "n": len(y),
            "n_bad": int(y.sum()),
            "selected_features": self.selected_features,
            "iv_scores": {f: round(self.iv_scores[f], 4) for f in self.selected_features},
        }

    def save(self, path: str):
        """Save model to joblib."""
        state = {
            "version": self.version,
            "woe_maps": self.woe_maps,
            "woe_bins": self.woe_bins,
            "selected_features": self.selected_features,
            "iv_scores": self.iv_scores,
            "lr_model": self.lr_model,
            "calibrator": self.calibrator,
            "train_medians": self.train_medians,
        }
        joblib.dump(state, path)
        logger.info(f"Saved DefaultModel to {path}")

    @classmethod
    def load(cls, path: str) -> "DefaultModel":
        """Load model from joblib."""
        state = joblib.load(path)
        model = cls()
        model.version = state.get("version", "unknown")
        model.woe_maps = state["woe_maps"]
        model.woe_bins = state["woe_bins"]
        model.selected_features = state["selected_features"]
        model.iv_scores = state["iv_scores"]
        model.lr_model = state["lr_model"]
        model.calibrator = state["calibrator"]
        model.train_medians = state["train_medians"]
        return model
