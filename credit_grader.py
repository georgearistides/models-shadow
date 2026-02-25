"""CreditGrader — Maps calibrated probability of default (PD) to letter grades.

Supports four grading schemes:
  - 5grade: A/B/C/D/F based on PD quantile boundaries
  - 6grade: A/B/C/D/E/F based on PD quantile boundaries
  - 7grade: A/B/C/D/E/F/G based on fixed PD thresholds (production-style)
  - optimal_iv: A/B/C/D/E with boundaries that maximize total Information Value
                (requires y_true labels during fit)

Partner overrides:
  - PAYSAFE: 1-grade downgrade (A->B, B->C, ..., F stays F, G stays G)

Usage:
    cg = CreditGrader(scheme="optimal_iv")
    cg.fit(train_pds, y_true=train_labels)      # optimal_iv needs labels
    result = cg.grade(0.15)
    grades_df = cg.grade_batch(pd_array)
    metrics = cg.evaluate(pds, y_true)
    cg.save("credit_grader.joblib")
    cg2 = CreditGrader.load("credit_grader.joblib")
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------

SCHEME_CONFIG = {
    "5grade": {
        "type": "quantile",
        "grades": ["A", "B", "C", "D", "F"],
        "quantiles": [20, 40, 60, 80],   # upper boundary percentiles
    },
    "6grade": {
        "type": "quantile",
        "grades": ["A", "B", "C", "D", "E", "F"],
        "quantiles": [15, 30, 50, 70, 85],
    },
    "7grade": {
        "type": "fixed",
        "grades": ["A", "B", "C", "D", "E", "F", "G"],
        "boundaries": [0.04, 0.08, 0.12, 0.18, 0.25, 0.35],
    },
    "optimal_iv": {
        "type": "optimal_iv",
        "grades": ["A", "B", "C", "D", "E"],
        "n_grades": 5,
    },
}


class CreditGrader:
    """Maps calibrated PD to letter grade with optional partner overrides."""

    VERSION = "1.1.0"

    def __init__(self, scheme="5grade"):
        """
        Args:
            scheme: One of "5grade", "6grade", "7grade", "optimal_iv".
        """
        if scheme not in SCHEME_CONFIG:
            raise ValueError(
                f"Unknown scheme '{scheme}'. Choose from: {list(SCHEME_CONFIG.keys())}"
            )
        self.scheme = scheme
        self.boundaries = None          # sorted array of PD cutoffs
        self.grades = list(SCHEME_CONFIG[scheme]["grades"])
        self._fitted = False
        self._iv_score = None           # Total IV (for optimal_iv scheme)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, pds, scheme=None, y_true=None):
        """Compute grade boundaries from a PD distribution.

        Args:
            pds: array-like of calibrated PDs (from training set).
            scheme: Optionally override the scheme set at init time.
            y_true: array-like of binary labels (required for optimal_iv scheme).

        Returns:
            self (for chaining).
        """
        if scheme is not None:
            self.set_scheme(scheme)

        pds = np.asarray(pds, dtype=float)
        finite_mask = np.isfinite(pds)

        if y_true is not None:
            y_true = np.asarray(y_true, dtype=int)
            # Keep only finite PDs and their matching labels
            pds = pds[finite_mask]
            y_true = y_true[finite_mask]
        else:
            pds = pds[finite_mask]

        if len(pds) == 0:
            raise ValueError("No finite PD values provided.")

        cfg = SCHEME_CONFIG[self.scheme]

        if cfg["type"] == "optimal_iv":
            if y_true is None:
                raise ValueError(
                    "optimal_iv scheme requires y_true labels. "
                    "Pass y_true=... to .fit()."
                )
            n_grades = cfg["n_grades"]
            self.boundaries, self._iv_score = self._find_optimal_iv_boundaries(
                pds, y_true, n_grades
            )
            self.grades = list(cfg["grades"][:len(self.boundaries) + 1])
            logger.info(f"Optimal-IV fit: {len(self.grades)} grades, "
                        f"total IV={self._iv_score:.4f}, "
                        f"boundaries={[round(b, 4) for b in self.boundaries]}")
        elif cfg["type"] == "quantile":
            self.boundaries = np.percentile(pds, cfg["quantiles"])
        elif cfg["type"] == "fixed":
            self.boundaries = np.array(cfg["boundaries"], dtype=float)
        else:
            raise ValueError(f"Unknown scheme type: {cfg['type']}")

        # Ensure strictly increasing (deduplicate if quantiles collapse)
        self.boundaries = np.unique(self.boundaries)

        # If quantile collapse reduced the number of boundaries, rebuild
        # grade labels to match (keep first N+1 grades for N boundaries).
        if cfg["type"] != "optimal_iv":
            expected_n_boundaries = len(cfg.get("quantiles", cfg.get("boundaries")))
            actual_n_boundaries = len(self.boundaries)
            if actual_n_boundaries < expected_n_boundaries:
                self.grades = list(SCHEME_CONFIG[self.scheme]["grades"])
                n_grades_needed = actual_n_boundaries + 1
                if n_grades_needed < len(self.grades):
                    indices = np.linspace(
                        0, len(SCHEME_CONFIG[self.scheme]["grades"]) - 1,
                        n_grades_needed
                    ).astype(int)
                    self.grades = [SCHEME_CONFIG[self.scheme]["grades"][i] for i in indices]
            else:
                self.grades = list(SCHEME_CONFIG[self.scheme]["grades"])

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Optimal-IV boundary search
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iv_for_boundaries(pds, y_true, boundaries):
        """Compute total Information Value for a set of grade boundaries.

        IV = sum over bins of: (pct_good_i - pct_bad_i) * WoE_i
        where WoE_i = ln(pct_good_i / pct_bad_i)
        """
        total_good = (y_true == 0).sum()
        total_bad = (y_true == 1).sum()
        if total_good == 0 or total_bad == 0:
            return 0.0

        # Create bin assignments
        bin_indices = np.searchsorted(boundaries, pds, side="right")

        iv = 0.0
        for b in range(len(boundaries) + 1):
            mask = bin_indices == b
            n_good = ((y_true == 0) & mask).sum()
            n_bad = ((y_true == 1) & mask).sum()

            # Laplace smoothing to avoid log(0)
            pct_good = max(n_good, 0.5) / total_good
            pct_bad = max(n_bad, 0.5) / total_bad

            woe = np.log(pct_good / pct_bad)
            iv += (pct_good - pct_bad) * woe

        return iv

    @staticmethod
    def _find_optimal_iv_boundaries(pds, y_true, n_grades, n_candidates=200):
        """Find PD boundaries that maximize total Information Value.

        Uses a grid search over candidate percentile-based boundaries,
        then refines with local perturbation.

        Args:
            pds: array of PD values
            y_true: array of binary labels
            n_grades: number of grades (boundaries = n_grades - 1)
            n_candidates: number of random candidate boundary sets to evaluate

        Returns:
            (best_boundaries, best_iv)
        """
        n_boundaries = n_grades - 1

        # Generate candidate boundary sets from percentiles
        # Strategy: evaluate many percentile combinations and pick the best
        best_iv = -1
        best_boundaries = None

        # 1. Try equal-frequency quantile boundaries as baseline
        for q_offset in range(0, 5):
            try:
                quantiles = np.linspace(
                    100 / (n_grades + q_offset),
                    100 - 100 / (n_grades + q_offset),
                    n_boundaries,
                )
                boundaries = np.percentile(pds, quantiles)
                boundaries = np.unique(boundaries)
                if len(boundaries) == n_boundaries:
                    iv = CreditGrader._compute_iv_for_boundaries(pds, y_true, boundaries)
                    if iv > best_iv:
                        best_iv = iv
                        best_boundaries = boundaries.copy()
            except Exception:
                continue

        # 2. Systematic grid search over percentile combinations
        # For 5 grades (4 boundaries), search over a grid of percentile combos
        pds_sorted = np.sort(pds)
        percentile_grid = np.arange(5, 96, 2)  # 5, 7, 9, ..., 95

        if n_boundaries <= 4:
            # For small n_boundaries, do exhaustive-ish search
            from itertools import combinations
            # Sample a subset of percentile combinations for tractability
            all_combos = list(combinations(percentile_grid, n_boundaries))
            # If too many, sample
            if len(all_combos) > n_candidates:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(all_combos), n_candidates, replace=False)
                combos_to_try = [all_combos[i] for i in indices]
            else:
                combos_to_try = all_combos

            for combo in combos_to_try:
                boundaries = np.percentile(pds, list(combo))
                boundaries = np.unique(boundaries)
                if len(boundaries) != n_boundaries:
                    continue
                iv = CreditGrader._compute_iv_for_boundaries(pds, y_true, boundaries)
                if iv > best_iv:
                    best_iv = iv
                    best_boundaries = boundaries.copy()

        # 3. Local refinement around best boundaries
        if best_boundaries is not None:
            for iteration in range(50):
                improved = False
                for i in range(n_boundaries):
                    # Try shifting boundary i up and down
                    current_pct = (pds <= best_boundaries[i]).mean() * 100
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_pct = np.clip(current_pct + delta, 3, 97)
                        trial = best_boundaries.copy()
                        trial[i] = np.percentile(pds, new_pct)
                        trial = np.sort(np.unique(trial))
                        if len(trial) != n_boundaries:
                            continue
                        iv = CreditGrader._compute_iv_for_boundaries(pds, y_true, trial)
                        if iv > best_iv:
                            best_iv = iv
                            best_boundaries = trial.copy()
                            improved = True
                if not improved:
                    break

        if best_boundaries is None:
            # Fallback: equal-frequency
            quantiles = np.linspace(100 / n_grades, 100 - 100 / n_grades, n_boundaries)
            best_boundaries = np.percentile(pds, quantiles)
            best_iv = CreditGrader._compute_iv_for_boundaries(pds, y_true, best_boundaries)

        return best_boundaries, best_iv

    # ------------------------------------------------------------------
    # Scheme switching
    # ------------------------------------------------------------------

    def set_scheme(self, scheme):
        """Switch grading scheme. Resets fitted state."""
        if scheme not in SCHEME_CONFIG:
            raise ValueError(
                f"Unknown scheme '{scheme}'. Choose from: {list(SCHEME_CONFIG.keys())}"
            )
        self.scheme = scheme
        self.grades = list(SCHEME_CONFIG[scheme]["grades"])
        self.boundaries = None
        self._fitted = False
        self._iv_score = None

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("CreditGrader is not fitted. Call .fit(pds) first.")

    def _raw_grade_index(self, pd_value):
        """Return the 0-based grade index for a single PD value."""
        # np.searchsorted with side='right': values exactly on a boundary
        # fall into the higher (worse) grade.
        idx = int(np.searchsorted(self.boundaries, pd_value, side="right"))
        return min(idx, len(self.grades) - 1)

    def _apply_partner_override(self, grade_index, partner):
        """Downgrade by 1 for PAYSAFE; cap at worst grade."""
        if partner is not None and str(partner).upper() == "PAYSAFE":
            return min(grade_index + 1, len(self.grades) - 1)
        return grade_index

    def _pd_range_for_index(self, idx):
        """Return (low, high) PD range for a grade index."""
        low = 0.0 if idx == 0 else float(self.boundaries[idx - 1])
        high = float(self.boundaries[idx]) if idx < len(self.boundaries) else 1.0
        return [round(low, 6), round(high, 6)]

    def grade(self, pd_value, partner=None):
        """Grade a single PD value.

        Args:
            pd_value: Calibrated probability of default.
            partner: Optional partner name (e.g. "PAYSAFE" for override).

        Returns:
            dict with keys: grade, pd_range, pricing_tier, grade_index.
        """
        self._check_fitted()
        pd_value = float(pd_value)

        idx = self._raw_grade_index(pd_value)
        idx = self._apply_partner_override(idx, partner)

        return {
            "grade": self.grades[idx],
            "pd_range": self._pd_range_for_index(idx),
            "pricing_tier": idx + 1,
            "grade_index": idx,
        }

    def grade_batch(self, pds, partners=None):
        """Grade an array of PDs.

        Args:
            pds: array-like of PD values.
            partners: None, a single string, or array-like of partner names
                      matching length of pds.

        Returns:
            DataFrame with columns: grade, pd_range_low, pd_range_high,
            pricing_tier, grade_index.
        """
        self._check_fitted()
        pds = np.asarray(pds, dtype=float)
        n = len(pds)

        # Vectorised raw index assignment
        raw_indices = np.searchsorted(self.boundaries, pds, side="right")
        raw_indices = np.minimum(raw_indices, len(self.grades) - 1)

        # Partner overrides
        if partners is not None:
            if isinstance(partners, str):
                partners = np.array([partners] * n)
            else:
                partners = np.asarray(partners)

            is_paysafe = np.char.upper(partners.astype(str)) == "PAYSAFE"
            raw_indices = np.where(
                is_paysafe,
                np.minimum(raw_indices + 1, len(self.grades) - 1),
                raw_indices,
            )

        grade_arr = np.array(self.grades)

        # Build PD range arrays
        lows = np.where(raw_indices == 0, 0.0,
                        self.boundaries[np.minimum(raw_indices - 1,
                                                    len(self.boundaries) - 1)])
        # For index 0, low is 0 — already handled above
        highs = np.where(
            raw_indices < len(self.boundaries),
            self.boundaries[raw_indices.clip(0, len(self.boundaries) - 1)],
            1.0,
        )

        return pd.DataFrame({
            "grade": grade_arr[raw_indices],
            "pd_range_low": np.round(lows, 6),
            "pd_range_high": np.round(highs, 6),
            "pricing_tier": raw_indices + 1,
            "grade_index": raw_indices,
        })

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, pds, y_true, partners=None):
        """Evaluate grading quality.

        Args:
            pds: array-like of PD values.
            y_true: array-like of binary default labels (0/1).
            partners: Optional partner array for overrides.

        Returns:
            dict with:
                monotonic: bool — are bad rates monotonically increasing by grade?
                grade_bad_rates: dict of grade -> observed bad rate
                grade_counts: dict of grade -> count
                auc: AUC of grade index as a score
        """
        self._check_fitted()
        pds = np.asarray(pds, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        grades_df = self.grade_batch(pds, partners=partners)
        grade_indices = grades_df["grade_index"].values

        # Bad rate per grade
        eval_df = pd.DataFrame({
            "grade": grades_df["grade"].values,
            "grade_index": grade_indices,
            "y": y_true,
        })
        stats = eval_df.groupby("grade").agg(
            bad_rate=("y", "mean"),
            count=("y", "count"),
            bads=("y", "sum"),
        ).reindex(self.grades).dropna()

        grade_bad_rates = stats["bad_rate"].round(4).to_dict()
        grade_counts = stats["count"].astype(int).to_dict()
        grade_bads = stats["bads"].astype(int).to_dict()

        # Monotonicity check
        rates_ordered = [grade_bad_rates.get(g, 0) for g in self.grades if g in grade_bad_rates]
        is_monotonic = all(
            rates_ordered[i] <= rates_ordered[i + 1]
            for i in range(len(rates_ordered) - 1)
        )

        # AUC: use grade_index as a risk score (higher index = higher risk)
        try:
            auc = round(roc_auc_score(y_true, grade_indices), 4)
        except ValueError:
            auc = None

        return {
            "monotonic": is_monotonic,
            "grade_bad_rates": grade_bad_rates,
            "grade_counts": grade_counts,
            "grade_bads": grade_bads,
            "auc": auc,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path):
        """Save grader to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": self.VERSION,
            "scheme": self.scheme,
            "boundaries": self.boundaries,
            "grades": self.grades,
            "_fitted": self._fitted,
            "iv_score": self._iv_score,
        }
        joblib.dump(state, path)
        return str(path)

    @classmethod
    def load(cls, path):
        """Load a saved CreditGrader from disk."""
        state = joblib.load(path)

        # Version check (warn, don't error, for backward compatibility)
        saved_version = state.get("version", state.get("_version", "unknown"))
        if saved_version != cls.VERSION:
            logger.warning(
                f"CreditGrader version mismatch: saved={saved_version}, "
                f"current={cls.VERSION}. Model behavior may differ."
            )

        # Handle loading schemes that may not exist in older configs
        scheme = state.get("scheme", "5grade")
        if scheme not in SCHEME_CONFIG:
            # Backward compat: load as custom scheme
            logger.warning(f"Scheme '{scheme}' not in SCHEME_CONFIG; loading as-is.")
            obj = cls.__new__(cls)
            obj.scheme = scheme
        else:
            obj = cls(scheme=scheme)
        obj.boundaries = state["boundaries"]
        obj.grades = state["grades"]
        obj._fitted = state["_fitted"]
        obj._iv_score = state.get("iv_score")
        return obj

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self):
        if self._fitted:
            bnd = ", ".join(f"{b:.4f}" for b in self.boundaries)
            return (
                f"CreditGrader(scheme='{self.scheme}', grades={self.grades}, "
                f"boundaries=[{bnd}])"
            )
        return f"CreditGrader(scheme='{self.scheme}', fitted=False)"


class BoundaryGrader:
    """Maps a numeric value to a grade label using sorted PD boundaries.

    Uses ``np.searchsorted`` for O(n log k) vectorised grading instead of
    row-wise ``.apply()``.

    Parameters
    ----------
    name : str
        Human-readable name for this grader (e.g. ``"pd_grade"``).
    boundaries : list[float]
        Sorted PD cut-points.  For *k* boundaries there are *k+1* grades.
    grades : list[str]
        Grade labels, length must equal ``len(boundaries) + 1``.
    """

    def __init__(self, name, boundaries, grades):
        self.name = name
        self.boundaries = np.asarray(sorted(boundaries), dtype=float)
        self.grades = list(grades)
        if len(self.grades) != len(self.boundaries) + 1:
            raise ValueError(
                f"Need exactly {len(self.boundaries) + 1} grade labels for "
                f"{len(self.boundaries)} boundaries, got {len(self.grades)}."
            )

    def grade(self, value):
        """Grade a single numeric value."""
        if pd.isna(value):
            return self.grades[-1]
        idx = int(np.searchsorted(self.boundaries, value, side="right"))
        return self.grades[min(idx, len(self.grades) - 1)]

    def grade_series(self, series):
        """Vectorised grading of a pandas Series using ``np.searchsorted``.

        Returns a Series of grade labels with the same index as *series*.
        Missing / NaN values are assigned the worst (last) grade.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        if series.dtype == object:
            series = pd.to_numeric(series, errors="coerce")

        values = series.values.astype(float)
        grade_arr = np.array(self.grades)

        # searchsorted gives the bin index for each value
        indices = np.searchsorted(self.boundaries, values, side="right")
        indices = np.minimum(indices, len(self.grades) - 1)

        # Assign worst grade to NaN values
        nan_mask = np.isnan(values)
        indices[nan_mask] = len(self.grades) - 1

        return pd.Series(grade_arr[indices], index=series.index)

    def __repr__(self):
        return (
            f"BoundaryGrader(name='{self.name}', "
            f"boundaries={list(self.boundaries)}, grades={self.grades})"
        )
