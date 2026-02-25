"""
FraudGate — Rule-based fraud scoring system for subprime SMB lending.

Importable class that scores loan applications using a point-based rule engine
derived from validated fraud signal analysis (AUC 0.7316 on test set).

Usage:
    from scripts.models.fraud_gate import FraudGate

    fg = FraudGate()
    fg.fit(train_df)  # optional — calibrate thresholds from data

    # Single prediction
    result = fg.predict({"fico": 620, "qi": "MISSING", ...})
    # -> {"score": 95, "normalized_score": 0.61, "decision": "decline", "reasons": [...]}

    # Batch prediction
    results_df = fg.predict_batch(df)

    # Evaluate
    metrics = fg.evaluate(test_df, y_col="is_bad")

    # Serialize
    fg.save("fraud_gate.joblib")
    fg2 = FraudGate.load("fraud_gate.joblib")
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from scripts.models.model_utils import (
    _is_missing, _safe_float, _safe_str,
    BAD_STATES, GOOD_STATES, EXCLUDE_STATES,
)
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# With inverted rules removed (2026-02-25), effective category maxima are:
# Cat1=30 Cat2=0(phone trust+prove removed, taktile disabled) Cat3=35 Cat4=25 Cat5=15 Cat6=20 => 125
# Cat7 (Entity Graph) adds up to 40 when graph lookup is loaded.
# MAX_RAW_SCORE kept at 145 for backward compatibility with serialized
# models / threshold calibration. Normalized scores will be slightly
# lower on average, which is correct (fewer false positives).
MAX_RAW_SCORE = 145
MAX_RAW_SCORE_WITH_GRAPH = 185  # 145 + 40 (Cat7 max: 25+10+5)

# BAD_STATES, GOOD_STATES, EXCLUDE_STATES imported from model_utils

THRESHOLDS = {
    "conservative": {"review": 20, "reject": 55},
    "moderate": {"review": 25, "reject": 70},
    "aggressive": {"review": 35, "reject": 80},
}
DEFAULT_THRESHOLD = "moderate"

# Column name mapping: master_features long names -> short names
_COLUMN_MAP = {
    "experian_FICO_SCORE": "fico",
    "shop_qi": "qi",
    "experian_delinquencies_thirty_day_count": "d30",
    "experian_delinquencies_sixty_day_count": "d60",
    "experian_inquiries_last_six_months": "inq6",
    "experian_revolving_account_credit_available_percentage": "revutil",
    "experian_balance_total_past_due_amounts": "pastdue",
    "experian_balance_total_installment_accounts": "instbal",
    "experian_balance_total_revolving_accounts": "revbal",
    "experian_tradelines_total_items": "tl_total",
    "experian_tradelines_total_items_paid": "tl_paid",
    "experian_tradelines_total_items_currently_delinquent": "tl_delin",
    "experian_payment_amount_monthly_total": "mopmt",
    "experian_tradelines_oldest_date": "crhist_date",
    "experian_delinquencies_ninety_to_one_hundred_eighty_days": "d90",
}


# ---------------------------------------------------------------------------
# FraudGate class
# ---------------------------------------------------------------------------


class FraudGate:
    """Rule-based fraud scoring system.

    Scores loan applications using a weighted point system across seven
    categories: business verification, identity verification, financial
    stress, bureau red flags, coverage/missingness, interaction flags,
    and entity graph (optional).

    Parameters
    ----------
    threshold_profile : str
        One of "conservative", "moderate", "aggressive". Controls the
        review and reject score boundaries.
    custom_thresholds : dict, optional
        Override with {"review": int, "reject": int}.
    """

    VERSION = "1.2.0"  # v1.2.0: Remove 6 inverted/zero-lift rules (LFLT, mid-FICO QI, phone trust, Prove not verified, QI+600-649 interaction, identity gap condition)

    def __init__(
        self,
        threshold_profile: str = DEFAULT_THRESHOLD,
        custom_thresholds: Optional[Dict[str, int]] = None,
    ):
        if custom_thresholds is not None:
            self.thresholds = custom_thresholds
        elif threshold_profile in THRESHOLDS:
            self.thresholds = THRESHOLDS[threshold_profile].copy()
        else:
            raise ValueError(
                f"Unknown threshold_profile '{threshold_profile}'. "
                f"Choose from: {list(THRESHOLDS.keys())}"
            )
        self.threshold_profile = threshold_profile

        # Populated by .fit() — optional calibration data
        self._fitted = False
        self._train_score_percentiles: Optional[Dict[int, float]] = None
        self._train_score_stats: Optional[Dict[str, float]] = None

        # Entity graph lookup — optional, loaded via load_graph_lookup()
        # Dict mapping key -> {"has_prior_bad": bool, "connected_count": int, ...}
        self._graph_lookup: Optional[Dict[str, Dict[str, Any]]] = None
        # Which column to use for graph lookup (shop_id or application_id)
        self._graph_lookup_key: str = "shop_id"

    # ------------------------------------------------------------------
    # Entity graph lookup
    # ------------------------------------------------------------------

    @property
    def graph_loaded(self) -> bool:
        """Whether an entity graph lookup is loaded."""
        return self._graph_lookup is not None and len(self._graph_lookup) > 0

    @staticmethod
    def build_graph_lookup(
        entity_df: pd.DataFrame, key_col: str = "application_id"
    ) -> Dict[str, Dict[str, Any]]:
        """Build graph lookup dict from entity graph DataFrame.

        Parameters
        ----------
        entity_df : pd.DataFrame
            Entity graph data with columns: has_prior_bad, prior_shops_ssn,
            is_repeat, is_cross_entity, and the key column.
        key_col : str
            Column to use as the lookup key (default: 'application_id').

        Returns
        -------
        dict mapping key_col values -> entity signal dict
        """
        lookup: Dict[str, Dict[str, Any]] = {}
        for _, row in entity_df.iterrows():
            key = str(row[key_col])
            lookup[key] = {
                "has_prior_bad": bool(row.get("has_prior_bad", 0)),
                "connected_count": int(row.get("prior_shops_ssn", 0)),
                "is_repeat": bool(row.get("is_repeat", 0)),
                "is_cross_entity": bool(row.get("is_cross_entity", 0)),
            }
        return lookup

    def load_graph_lookup(
        self, path: Union[str, Path], key_col: str = "shop_id"
    ) -> "FraudGate":
        """Load entity graph lookup from a JSON file.

        The lookup maps key -> {"has_prior_bad": bool, "connected_count": int, ...}.
        When loaded, Category 7 (Entity Graph) rules become active in scoring.

        Parameters
        ----------
        path : str or Path
            Path to the graph lookup JSON file (created by build_graph_lookup.py).
        key_col : str
            Which column the lookup keys correspond to (default: 'shop_id').

        Returns
        -------
        self
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Graph lookup file not found: {path}")
        with open(path, "r") as f:
            self._graph_lookup = json.load(f)
        self._graph_lookup_key = key_col
        logger.info(
            f"Loaded entity graph lookup: {len(self._graph_lookup)} entries "
            f"(key={key_col}), "
            f"{sum(1 for v in self._graph_lookup.values() if v.get('has_prior_bad'))} "
            f"with prior bad"
        )
        return self

    def set_graph_lookup(
        self,
        lookup: Dict[str, Dict[str, Any]],
        key_col: str = "shop_id",
    ) -> "FraudGate":
        """Set the entity graph lookup directly from a dict.

        Parameters
        ----------
        lookup : dict
            Mapping of key -> {"has_prior_bad": bool, "connected_count": int, ...}.
        key_col : str
            Which column the lookup keys correspond to (default: 'shop_id').

        Returns
        -------
        self
        """
        self._graph_lookup = lookup
        self._graph_lookup_key = key_col
        return self

    # ------------------------------------------------------------------
    # Feature normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """Map long column names to short names.

        Accepts both master_features (long Experian names) and
        default_model (short names). Returns a dict with short names.
        """
        out = dict(features)  # shallow copy
        for long_name, short_name in _COLUMN_MAP.items():
            if long_name in out and short_name not in out:
                out[short_name] = out[long_name]
        return out

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_derived(f: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived features needed by the rules.

        Parameters
        ----------
        f : dict
            Feature dict with short names.

        Returns
        -------
        dict with additional keys: prove_missing, plaid_missing,
        taktile_missing, giact_missing, n_providers, qi_missing,
        cash_stress, contradiction_score, delin_rate.
        """
        d = dict(f)

        # --- Provider missingness ---
        d["prove_missing"] = _is_missing(f.get("prove_phone_trust_score"))
        d["plaid_missing"] = _is_missing(f.get("plaid_balance"))
        d["taktile_missing"] = _is_missing(f.get("taktile_status"))
        d["giact_missing"] = _is_missing(f.get("giact_bankruptcy_count"))

        n_missing = sum([
            d["prove_missing"], d["plaid_missing"],
            d["taktile_missing"], d["giact_missing"],
        ])
        d["n_providers"] = 4 - n_missing

        # --- QI missing ---
        qi_val = _safe_str(f.get("qi"))
        d["qi_missing"] = (qi_val is None) or (qi_val == "MISSING")

        # --- Cash stress composite ---
        # Missing FICO should be treated as risky (pessimistic default)
        fico = _safe_float(f.get("fico"), default=500)
        pastdue = _safe_float(f.get("pastdue"), default=0)
        revutil = _safe_float(f.get("revutil"), default=0)
        d60 = _safe_float(f.get("d60"), default=0)

        d["cash_stress"] = (
            int(fico < 580) * 2
            + int(pastdue > 0)
            + int(revutil > 80)
            + int(d60 > 0)
        )

        # --- Contradiction score ---
        contradictions = 0
        prove_verified = f.get("prove_verified")
        plaid_balance = f.get("plaid_balance")
        if (
            not d["prove_missing"]
            and not _is_missing(prove_verified)
            and _safe_float(prove_verified) == 1
            and not d["plaid_missing"]
            and _safe_float(plaid_balance) < 0
        ):
            contradictions += 1  # Verified identity but negative bank balance

        d["contradiction_score"] = contradictions

        # --- Delinquency rate ---
        tl_total = _safe_float(f.get("tl_total"), default=0)
        tl_delin = _safe_float(f.get("tl_delin"), default=0)
        if tl_total > 0 and not _is_missing(f.get("tl_delin")):
            d["delin_rate"] = tl_delin / max(tl_total, 1)
        else:
            d["delin_rate"] = 0.0

        return d

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------

    def _apply_rules(self, f: Dict[str, Any]) -> tuple:
        """Apply all fraud rules. Returns (raw_score, reasons_list).

        Each category has a cap to prevent a single dimension from
        dominating the total score.
        """
        reasons: List[str] = []

        # Helper to read values
        def val(key, default=0.0):
            return _safe_float(f.get(key), default)

        def sval(key):
            return _safe_str(f.get(key))

        # ============================================================
        # Category 1: Business Verification (max 30 pts)
        # ============================================================
        # v1.1.0: Tiered missing-QI penalties based on FICO and partner.
        # Previously: blanket +30 for any missing QI.
        # Now: HONEYBOOK+missing QI = +30 (74.4% bad rate),
        #       FICO<600 = +30, 600-649 = +25, 650-699 = +20, 700+ = +10
        cat1 = 0
        qi_val = sval("qi")
        qi_missing = f.get("qi_missing", False)
        fico_for_qi = val("fico", 500)
        partner_val_for_qi = sval("partner")

        if qi_missing:
            if partner_val_for_qi == "HONEYBOOK":
                # HONEYBOOK + missing QI = 74.4% bad rate — maximum penalty
                cat1 += 30
                reasons.append("MISSING_QI_HONEYBOOK: Missing QI + HONEYBOOK partner (+30)")
            elif fico_for_qi < 600:
                cat1 += 30
                reasons.append("MISSING_QI_LOW_FICO: Missing QI + FICO<600 (+30)")
            # NOTE: Missing QI + FICO 600-649 rule removed (2026-02-25).
            # Bad rate 2.78% is far below base rate 6.63% — inverted signal.
            # Range is now 600-699 (was 650-699 before the 600-649 tier
            # was removed). Points reduced from 25 to 20 accordingly.
            elif fico_for_qi < 700:
                cat1 += 20
                reasons.append("MISSING_QI_MID_FICO: Missing QI + FICO 600-699 (+20)")
            else:
                cat1 += 10
                reasons.append("MISSING_QI_HIGH_FICO: Missing QI + FICO 700+ (+10)")
        # NOTE: LFLT rule removed (2026-02-25). LFLT bad rate 4.61% is
        # BELOW base rate 6.63% — the rule was inverted (penalizing a
        # protective signal). Old serialized models still load fine; the
        # rule simply never fires.

        bankruptcy_count = val("giact_bankruptcy_count", 0)
        if not f.get("giact_missing", True) and bankruptcy_count > 0:
            pts = min(15, 30 - cat1)  # respect category cap
            if pts > 0:
                cat1 += pts
                reasons.append(f"GIACT bankruptcy count={int(bankruptcy_count)} (+{pts})")

        cat1 = min(cat1, 30)

        # ============================================================
        # Category 2: Identity Verification (max 35 pts)
        # ============================================================
        cat2 = 0

        taktile_status = sval("taktile_status")
        taktile_output = sval("taktile_output")

        # NOTE: Taktile rules DISABLED (2026-02-25).
        # Root cause: taktile_output contains the attribute name "Fraud Check"
        # for a standard evaluation test that nearly always PASSES. The regex
        # r"fraud|suspicious|high.?risk" matched this field name, not an actual
        # fraud indicator. Combined with taktile coverage jumping from 0% to
        # 99%+ in Jul-Sep 2025 (severe temporal confounding), ALL loans from
        # Sep 2025 onward triggered +25 pts, collapsing decisions to 0% approve.
        # Taktile sigma scores were also confirmed AUC ~0.50 (no signal).
        # Re-enable only after mid-2026 with sufficient seasoned data and a
        # properly structured fraud indicator field from Taktile.
        #
        # if not f.get("taktile_missing", True):
        #     if taktile_status is not None and re.search(
        #         r"fail|reject|denied", taktile_status, re.IGNORECASE
        #     ):
        #         cat2 += 20
        #         reasons.append(f"Taktile status FAIL ({taktile_status}) (+20)")
        #     if taktile_output is not None and re.search(
        #         r"fraud|suspicious|high.?risk", taktile_output, re.IGNORECASE
        #     ):
        #         pts = min(25, 35 - cat2)
        #         if pts > 0:
        #             cat2 += pts
        #             reasons.append(f"Taktile fraud flag (+{pts})")

        # NOTE: Low phone trust rule removed (2026-02-25).
        # Bad rate 6.03% is at base rate 6.63% — no signal. The phone
        # trust score does not discriminate between good and bad loans.
        #
        # NOTE: "Prove not verified" rule also removed (2026-02-25).
            # Bad rate 2.99% is below base rate 6.63% — Prove "not verified"
            # is actually protective, not risky.

        cat2 = min(cat2, 35)

        # ============================================================
        # Category 3: Financial Stress (max 35 pts)
        # ============================================================
        cat3 = 0
        # Missing FICO should be treated as risky (pessimistic default)
        fico = val("fico", 500)

        if fico < 550:
            cat3 += 15
            reasons.append(f"FICO < 550 ({int(fico)}) (+15)")
        elif fico < 600:
            cat3 += 8
            reasons.append(f"FICO 550-600 ({int(fico)}) (+8)")

        # Negative bank balance
        if not f.get("plaid_missing", True):
            plaid_balance = val("plaid_balance", 0)
            if plaid_balance < 0:
                pts = min(15, 35 - cat3)
                if pts > 0:
                    cat3 += pts
                    reasons.append(f"Negative bank balance=${plaid_balance:,.0f} (+{pts})")

            # High NSF count
            nsf_count = val("plaid_recentNSFs", 0)
            if nsf_count >= 3:
                pts = min(10, 35 - cat3)
                if pts > 0:
                    cat3 += pts
                    reasons.append(f"High NSFs={int(nsf_count)} (+{pts})")

        # High past due
        pastdue = val("pastdue", 0)
        if pastdue > 500:
            pts = min(8, 35 - cat3)
            if pts > 0:
                cat3 += pts
                reasons.append(f"High past due=${pastdue:,.0f} (+{pts})")

        # 3.6: Extended negative balance history (from Plaid transactions)
        # negative_balance_days_90d > 5 means repeated negative balances, not just a snapshot
        neg_days = f.get("negative_balance_days_90d")
        if neg_days is not None and not _is_missing(neg_days) and _safe_float(neg_days, 0) > 5:
            pts = min(8, 35 - cat3)
            if pts > 0:
                cat3 += pts
                reasons.append(f"Extended negative balance history ({int(_safe_float(neg_days, 0))} days in 90d) (+{pts})")

        cat3 = min(cat3, 35)

        # ============================================================
        # Category 4: Bureau Red Flags (max 25 pts)
        # ============================================================
        cat4 = 0

        d60_val = val("d60", 0)
        if d60_val > 0:
            cat4 += 10
            reasons.append(f"Severe delinquencies (60d+)={int(d60_val)} (+10)")

        inq6_val = val("inq6", 0)
        if inq6_val >= 6:
            pts = min(8, 25 - cat4)
            if pts > 0:
                cat4 += pts
                reasons.append(f"High inquiries (6mo)={int(inq6_val)} (+{pts})")

        tl_total = val("tl_total", 5)
        if tl_total <= 1:
            pts = min(12, 25 - cat4)
            if pts > 0:
                cat4 += pts
                reasons.append(f"Zero/minimal tradelines={int(tl_total)} (+{pts})")

        delin_rate = f.get("delin_rate", 0)
        if delin_rate > 0.5:
            pts = min(10, 25 - cat4)
            if pts > 0:
                cat4 += pts
                reasons.append(f"High delinquency rate={delin_rate:.1%} (+{pts})")

        cat4 = min(cat4, 25)

        # ============================================================
        # Category 5: Coverage / Missingness (max 15 pts)
        # ============================================================
        cat5 = 0
        n_providers = f.get("n_providers", 0)

        if n_providers == 0:
            cat5 += 15
            reasons.append("All providers missing (+15)")
        elif n_providers <= 1:
            # Not additive with "all missing"
            cat5 += 8
            reasons.append(f"Minimal provider coverage ({n_providers}/4) (+8)")
        else:
            # Individual provider missing penalties
            if f.get("plaid_missing", True):
                pts = min(3, 15 - cat5)
                if pts > 0:
                    cat5 += pts
                    reasons.append("Plaid missing (+3)")
            if f.get("prove_missing", True):
                pts = min(3, 15 - cat5)
                if pts > 0:
                    cat5 += pts
                    reasons.append("Prove missing (+3)")

        cat5 = min(cat5, 15)

        # ============================================================
        # Category 6: Interaction Flags (max 20 pts)
        # ============================================================
        cat6 = 0

        # Missing QI + Low FICO interaction (v1.1.0: tiered)
        # Previously: flat +20 for missing QI + FICO<600.
        # Now: +20 for <600, +12 for 600-649, +0 for 650+ (already penalized in Cat1)
        if qi_missing and fico < 600:
            cat6 += 20
            reasons.append(f"INTERACTION_QI_LOW_FICO: Missing QI + FICO<600 ({int(fico)}) (+20)")
        # NOTE: QI + FICO 600-649 interaction rule removed (2026-02-25).
        # Bad rate 2.78% — same inverted segment as Cat1 rule 1.3.

        # HONEYBOOK + Missing QI interaction (additional on top of Cat1)
        partner = sval("partner")
        if qi_missing and partner == "HONEYBOOK" and fico < 650:
            pts = min(8, 20 - cat6)
            if pts > 0:
                cat6 += pts
                reasons.append(f"INTERACTION_HONEYBOOK_QI: HONEYBOOK + Missing QI + low FICO ({int(fico)}) (+{pts})")

        # PAYSAFE + Low FICO
        if partner == "PAYSAFE" and fico < 600:
            pts = min(12, 20 - cat6)
            if pts > 0:
                cat6 += pts
                reasons.append(f"PAYSAFE + Low FICO ({int(fico)}) (+{pts})")

        # Bureau Stress (formerly "Bureau Stress + Identity Gap")
        # NOTE: n_providers condition removed (2026-02-25). The identity
        # gap component (n_providers <= 1) inverted the signal due to
        # temporal confounding — older loans had fewer providers but
        # also lower bad rates. Bureau stress alone (cash_stress >= 3)
        # still has signal.
        cash_stress = f.get("cash_stress", 0)
        if cash_stress >= 3:
            pts = min(10, 20 - cat6)
            if pts > 0:
                cat6 += pts
                reasons.append(f"Bureau stress ({cash_stress}) (+{pts})")

        # Provider Disagreement
        contradiction = f.get("contradiction_score", 0)
        if contradiction > 0:
            pts = min(8, 20 - cat6)
            if pts > 0:
                cat6 += pts
                reasons.append(f"Provider disagreement ({contradiction} contradictions) (+{pts})")

        cat6 = min(cat6, 20)

        # ============================================================
        # Category 7: Entity Graph (max 40 pts) — OPTIONAL
        # ============================================================
        # Only fires when a graph lookup is loaded. Otherwise cat7 = 0
        # and scoring is identical to pre-graph behavior.
        cat7 = 0

        if self._graph_lookup is not None:
            # Look up by the configured key column, falling back to shop_id
            lookup_key_val = f.get(self._graph_lookup_key) or f.get("shop_id")
            if lookup_key_val is not None:
                lookup_key_str = str(lookup_key_val)
                if not (isinstance(lookup_key_val, float) and math.isnan(lookup_key_val)):
                    entry = self._graph_lookup.get(lookup_key_str)
                    if entry is not None:
                        # Rule 7.1: Prior default by connected entity (+25)
                        if entry.get("has_prior_bad", False):
                            cat7 += 25
                            reasons.append("Prior default by connected entity (+25)")

                        # Rule 7.2: Multiple connected entities (3+) (+10)
                        connected_count = entry.get("connected_count", 0)
                        if connected_count >= 3:
                            pts = min(10, 40 - cat7)
                            if pts > 0:
                                cat7 += pts
                                reasons.append(
                                    f"Multiple connected entities ({connected_count}) (+{pts})"
                                )

                        # Rule 7.3: Cross-entity borrower (+5)
                        if entry.get("is_cross_entity", False):
                            pts = min(5, 40 - cat7)
                            if pts > 0:
                                cat7 += pts
                                reasons.append(f"Cross-entity borrower (+{pts})")

        cat7 = min(cat7, 40)

        # ============================================================
        # Total
        # ============================================================
        raw_score = cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7
        return raw_score, reasons

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(self, score: int) -> str:
        """Map raw score to decision string."""
        if score >= self.thresholds["reject"]:
            return "decline"
        elif score >= self.thresholds["review"]:
            return "review"
        else:
            return "pass"

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FraudGate":
        """Optionally calibrate from training data.

        Computes score distribution statistics and percentile-based
        threshold suggestions. Does NOT change thresholds automatically
        (call .set_thresholds() to apply).

        Parameters
        ----------
        df : pd.DataFrame
            Training data (must have at least the feature columns).

        Returns
        -------
        self
        """
        scores = self.predict_batch(df)["score"].values
        self._train_score_stats = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        }
        self._train_score_percentiles = {
            int(np.percentile(scores, p)): p
            for p in range(0, 101, 5)
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Public API: predict (single)
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single loan application.

        Parameters
        ----------
        features : dict
            Feature dictionary (accepts both long and short column names).

        Returns
        -------
        dict with keys: score, normalized_score, decision, reasons
        """
        # Normalize categorical inputs to uppercase for consistent matching
        features = dict(features)
        if isinstance(features.get("qi"), str):
            features["qi"] = features["qi"].upper()
        if isinstance(features.get("shop_qi"), str):
            features["shop_qi"] = features["shop_qi"].upper()
        if isinstance(features.get("partner"), str):
            features["partner"] = features["partner"].upper()

        f = self._normalize_features(features)
        f = self._compute_derived(f)
        raw_score, reasons = self._apply_rules(f)

        max_score = MAX_RAW_SCORE_WITH_GRAPH if self.graph_loaded else MAX_RAW_SCORE
        return {
            "score": raw_score,
            "normalized_score": round(raw_score / max_score, 4),
            "decision": self._decide(raw_score),
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # Public API: predict_batch
    # ------------------------------------------------------------------

    def predict_batch_slow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame row-by-row (original slow implementation).

        Kept for verification/testing. Use predict_batch() for production.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with feature columns.

        Returns
        -------
        pd.DataFrame with columns: score, normalized_score, decision, reasons
        """
        records = df.to_dict(orient="records")
        results = [self.predict(r) for r in records]
        return pd.DataFrame(results, index=df.index)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of loan applications (vectorized).

        Applies all 22 fraud rules using vectorized numpy/pandas operations
        for 10-50x speedup over the row-by-row version.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with feature columns.

        Returns
        -------
        pd.DataFrame with columns: score, normalized_score, decision, reasons
        """
        n = len(df)
        if n == 0:
            return pd.DataFrame(
                columns=["score", "normalized_score", "decision", "reasons"]
            )

        # =================================================================
        # Step 0: Normalize categorical inputs to uppercase
        # =================================================================
        df = df.copy()
        # IMPORTANT: Must preserve NaN/None as actual NaN, not convert to
        # string "NONE" or "NAN". The pattern .astype(str).str.upper() turns
        # None -> "NONE" and NaN -> "NAN", which breaks downstream missing-
        # value checks. Instead, apply .upper() only to non-null values.
        for col in ("qi", "shop_qi", "partner"):
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.upper()

        # =================================================================
        # Step 1: Normalize column names (long -> short)
        # =================================================================
        rename_map = {
            long: short
            for long, short in _COLUMN_MAP.items()
            if long in df.columns and short not in df.columns
        }
        w = df.rename(columns=rename_map)
        cols = set(w.columns)

        # =================================================================
        # Step 2: Extract and coerce feature vectors
        # =================================================================
        def _get_float(col, default=0.0):
            """Get column as float64 numpy array with default for missing."""
            if col not in cols:
                return np.full(n, default, dtype=np.float64)
            return pd.to_numeric(w[col], errors="coerce").fillna(default).values.astype(np.float64)

        def _is_na_vec(col):
            """Boolean mask: True where value is None/NaN/empty-string."""
            if col not in cols:
                return np.ones(n, dtype=bool)
            arr = w[col].values
            # Fast path: numeric types only have NaN/None
            if arr.dtype.kind in ("f", "i", "u"):
                return pd.isna(arr)
            # Object dtype: check None, NaN, and empty/whitespace strings
            result = np.empty(n, dtype=bool)
            for i in range(n):
                v = arr[i]
                if v is None:
                    result[i] = True
                elif isinstance(v, float) and math.isnan(v):
                    result[i] = True
                elif isinstance(v, str) and v.strip() == "":
                    result[i] = True
                else:
                    result[i] = False
            return result

        # --- Core numeric features ---
        fico = _get_float("fico", 500.0)
        pastdue = _get_float("pastdue", 0.0)
        revutil = _get_float("revutil", 0.0)
        d60 = _get_float("d60", 0.0)
        inq6 = _get_float("inq6", 0.0)
        tl_total = _get_float("tl_total", 5.0)
        tl_delin = _get_float("tl_delin", 0.0)
        plaid_balance = _get_float("plaid_balance", 0.0)
        nsf_count = _get_float("plaid_recentNSFs", 0.0)
        neg_bal_days = _get_float("negative_balance_days_90d", 0.0)
        phone_trust = _get_float("prove_phone_trust_score", 1000.0)
        bankruptcy_count = _get_float("giact_bankruptcy_count", 0.0)
        prove_verified_raw = _get_float("prove_verified", np.nan)

        # --- Raw string arrays (object dtype, direct from DataFrame) ---
        def _get_obj_arr(col):
            if col not in cols:
                return np.array([None] * n, dtype=object)
            return w[col].values

        qi_arr = _get_obj_arr("qi")
        partner_arr = _get_obj_arr("partner")
        taktile_status_arr = _get_obj_arr("taktile_status")
        taktile_output_arr = _get_obj_arr("taktile_output")

        # --- Provider missingness (computed once) ---
        prove_missing = _is_na_vec("prove_phone_trust_score")
        plaid_missing = _is_na_vec("plaid_balance")
        taktile_missing = _is_na_vec("taktile_status")
        giact_missing = _is_na_vec("giact_bankruptcy_count")
        prove_verified_missing = _is_na_vec("prove_verified")

        n_providers = (
            4
            - prove_missing.astype(np.int32)
            - plaid_missing.astype(np.int32)
            - taktile_missing.astype(np.int32)
            - giact_missing.astype(np.int32)
        )

        # --- QI missing + QI is LFLT (single pass over qi_arr) ---
        qi_missing = np.empty(n, dtype=bool)
        qi_is_lflt = np.empty(n, dtype=bool)
        for i in range(n):
            v = qi_arr[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                qi_missing[i] = True
                qi_is_lflt[i] = False
            else:
                sv = str(v).strip()
                qi_missing[i] = (sv == "" or sv == "MISSING")
                qi_is_lflt[i] = (sv == "LFLT")

        # --- Partner detection (single pass) ---
        partner_is_paysafe = np.empty(n, dtype=bool)
        partner_is_honeybook = np.empty(n, dtype=bool)
        for i in range(n):
            v = partner_arr[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                partner_is_paysafe[i] = False
                partner_is_honeybook[i] = False
            else:
                sv = str(v).strip()
                partner_is_paysafe[i] = sv == "PAYSAFE"
                partner_is_honeybook[i] = sv == "HONEYBOOK"

        # --- Cash stress (fully vectorized) ---
        cash_stress = (
            (fico < 580).astype(np.int32) * 2
            + (pastdue > 0).astype(np.int32)
            + (revutil > 80).astype(np.int32)
            + (d60 > 0).astype(np.int32)
        )

        # --- Contradiction score (fully vectorized) ---
        prove_verified_val = np.where(prove_verified_missing, np.nan, prove_verified_raw)
        contradiction = (
            (~prove_missing)
            & (~prove_verified_missing)
            & (prove_verified_val == 1)
            & (~plaid_missing)
            & (plaid_balance < 0)
        ).astype(np.int32)

        # --- Delinquency rate (fully vectorized) ---
        tl_delin_missing = _is_na_vec("tl_delin")
        delin_rate = np.where(
            (tl_total > 0) & (~tl_delin_missing),
            tl_delin / np.maximum(tl_total, 1),
            0.0,
        )

        # --- Taktile pattern matching — DISABLED (2026-02-25) ---
        # See comment in single-predict path for full rationale. The regex
        # matched the attribute name "Fraud Check" (a standard evaluation test
        # that usually PASSES), not actual fraud indicators. Combined with
        # severe temporal confounding (0%->99% coverage Jul-Sep 2025), this
        # collapsed all post-Sep-2025 decisions to review/decline.
        taktile_fail = np.zeros(n, dtype=bool)
        taktile_fraud = np.zeros(n, dtype=bool)
        # _status_pat = re.compile(r"fail|reject|denied", re.IGNORECASE)
        # for i in range(n):
        #     if not taktile_missing[i]:
        #         s = taktile_status_arr[i]
        #         if s is not None and _status_pat.search(str(s)):
        #             taktile_fail[i] = True
        #         o = taktile_output_arr[i]
        #         if o is not None:
        #             ol = str(o).lower()
        #             if "fraud" in ol or "suspicious" in ol or "high risk" in ol or "highrisk" in ol or "high-risk" in ol:
        #                 taktile_fraud[i] = True

        # =================================================================
        # Step 3: Compute scores per category (pure numpy, no Python loops)
        # =================================================================
        # Category scores are computed with sequential cap-aware accumulation

        # --- Category 1: Business Verification (max 30) ---
        # v1.1.0: Tiered missing-QI penalties based on FICO and partner
        # HONEYBOOK + missing QI = +30 (74.4% bad rate)
        # Missing QI + FICO<600 = +30, 600-649 = +25, 650-699 = +20, 700+ = +10
        qi_missing_honeybook = qi_missing & partner_is_honeybook
        qi_missing_low_fico = qi_missing & (~partner_is_honeybook) & (fico < 600)
        # qi_missing_mid_fico removed — bad rate 2.78% was inverted
        qi_missing_upper_mid_fico = qi_missing & (~partner_is_honeybook) & (fico >= 600) & (fico < 700)  # widened from 650-699 to 600-699
        qi_missing_high_fico = qi_missing & (~partner_is_honeybook) & (fico >= 700)

        # NOTE: LFLT (+10) removed — bad rate 4.61% below base rate.
        # NOTE: Missing QI + FICO 600-649 (+25) removed — bad rate 2.78%.
        # Range 600-699 now captured by qi_missing_upper_mid_fico (was 650-699).
        qi_missing_upper_mid_fico = qi_missing & (~partner_is_honeybook) & (fico >= 600) & (fico < 700)
        cat1 = np.where(qi_missing_honeybook, 30,
               np.where(qi_missing_low_fico, 30,
               np.where(qi_missing_upper_mid_fico, 20,
               np.where(qi_missing_high_fico, 10,
               0)))).astype(np.int32)

        r1_3_raw = (~giact_missing) & (bankruptcy_count > 0)
        pts_1_3 = np.minimum(15, np.maximum(30 - cat1, 0))
        r1_3 = r1_3_raw & (pts_1_3 > 0)
        cat1 = np.where(r1_3, cat1 + pts_1_3, cat1)
        cat1 = np.minimum(cat1, 30)

        # --- Category 2: Identity Verification (max 35) ---
        # NOTE: All Cat2 rules removed or disabled (2026-02-25):
        # - Taktile rules: DISABLED (temporal confounding, no signal)
        # - Low phone trust (<500): removed (bad rate 6.03% = base rate, no signal)
        # - Prove not verified: removed (bad rate 2.99% < base rate, inverted)
        cat2 = np.zeros(n, dtype=np.int32)
        r2_1 = (~taktile_missing) & taktile_fail
        cat2 = np.where(r2_1, 20, cat2)

        r2_2_raw = (~taktile_missing) & taktile_fraud
        pts_2_2 = np.minimum(25, np.maximum(35 - cat2, 0))
        r2_2 = r2_2_raw & (pts_2_2 > 0)
        cat2 = np.where(r2_2, cat2 + pts_2_2, cat2)

        # Phone trust and Prove not-verified rules removed — masks kept
        # as zeros for backward compat with reason generation structure
        r2_3 = np.zeros(n, dtype=bool)
        pts_2_3 = np.zeros(n, dtype=np.int32)
        r2_4 = np.zeros(n, dtype=bool)
        pts_2_4 = np.zeros(n, dtype=np.int32)
        cat2 = np.minimum(cat2, 35)

        # --- Category 3: Financial Stress (max 35) ---
        r3_1a = fico < 550
        r3_1b = (~r3_1a) & (fico < 600)
        cat3 = np.where(r3_1a, 15, np.where(r3_1b, 8, 0)).astype(np.int32)

        r3_2_raw = (~plaid_missing) & (plaid_balance < 0)
        pts_3_2 = np.minimum(15, np.maximum(35 - cat3, 0))
        r3_2 = r3_2_raw & (pts_3_2 > 0)
        cat3 = np.where(r3_2, cat3 + pts_3_2, cat3)

        r3_3_raw = (~plaid_missing) & (nsf_count >= 3)
        pts_3_3 = np.minimum(10, np.maximum(35 - cat3, 0))
        r3_3 = r3_3_raw & (pts_3_3 > 0)
        cat3 = np.where(r3_3, cat3 + pts_3_3, cat3)

        r3_4_raw = pastdue > 500
        pts_3_4 = np.minimum(8, np.maximum(35 - cat3, 0))
        r3_4 = r3_4_raw & (pts_3_4 > 0)
        cat3 = np.where(r3_4, cat3 + pts_3_4, cat3)

        # 3.5: Extended negative balance history (>5 days in 90d)
        neg_bal_days_missing = _is_na_vec("negative_balance_days_90d")
        r3_5_raw = (~neg_bal_days_missing) & (neg_bal_days > 5)
        pts_3_5 = np.minimum(8, np.maximum(35 - cat3, 0))
        r3_5 = r3_5_raw & (pts_3_5 > 0)
        cat3 = np.where(r3_5, cat3 + pts_3_5, cat3)

        cat3 = np.minimum(cat3, 35)

        # --- Category 4: Bureau Red Flags (max 25) ---
        r4_1 = d60 > 0
        cat4 = np.where(r4_1, 10, 0).astype(np.int32)

        r4_2_raw = inq6 >= 6
        pts_4_2 = np.minimum(8, np.maximum(25 - cat4, 0))
        r4_2 = r4_2_raw & (pts_4_2 > 0)
        cat4 = np.where(r4_2, cat4 + pts_4_2, cat4)

        r4_3_raw = tl_total <= 1
        pts_4_3 = np.minimum(12, np.maximum(25 - cat4, 0))
        r4_3 = r4_3_raw & (pts_4_3 > 0)
        cat4 = np.where(r4_3, cat4 + pts_4_3, cat4)

        r4_4_raw = delin_rate > 0.5
        pts_4_4 = np.minimum(10, np.maximum(25 - cat4, 0))
        r4_4 = r4_4_raw & (pts_4_4 > 0)
        cat4 = np.where(r4_4, cat4 + pts_4_4, cat4)
        cat4 = np.minimum(cat4, 25)

        # --- Category 5: Coverage / Missingness (max 15) ---
        r5_1 = n_providers == 0
        r5_2 = (~r5_1) & (n_providers <= 1)
        has_multiple = (~r5_1) & (~r5_2)

        cat5 = np.where(r5_1, 15, np.where(r5_2, 8, 0)).astype(np.int32)

        r5_3a_raw = has_multiple & plaid_missing
        pts_5_3a = np.minimum(3, np.maximum(15 - cat5, 0))
        r5_3a = r5_3a_raw & (pts_5_3a > 0)
        cat5 = np.where(r5_3a, cat5 + pts_5_3a, cat5)

        r5_3b_raw = has_multiple & prove_missing
        pts_5_3b = np.minimum(3, np.maximum(15 - cat5, 0))
        r5_3b = r5_3b_raw & (pts_5_3b > 0)
        cat5 = np.where(r5_3b, cat5 + pts_5_3b, cat5)
        cat5 = np.minimum(cat5, 15)

        # --- Category 6: Interaction Flags (max 20) ---
        # v1.2.0: QI + FICO 600-649 interaction removed (bad rate 2.78%)
        # Only Missing QI + FICO<600 (+20) retained
        r6_1a = qi_missing & (fico < 600)
        r6_1b = np.zeros(n, dtype=bool)  # Removed: was qi_missing & (fico >= 600) & (fico < 650)
        cat6 = np.where(r6_1a, 20, 0).astype(np.int32)

        # HONEYBOOK + Missing QI + low FICO interaction (additional)
        r6_1c_raw = qi_missing & partner_is_honeybook & (fico < 650)
        pts_6_1c = np.minimum(8, np.maximum(20 - cat6, 0))
        r6_1c = r6_1c_raw & (pts_6_1c > 0)
        cat6 = np.where(r6_1c, cat6 + pts_6_1c, cat6)

        r6_2_raw = partner_is_paysafe & (fico < 600)
        pts_6_2 = np.minimum(12, np.maximum(20 - cat6, 0))
        r6_2 = r6_2_raw & (pts_6_2 > 0)
        cat6 = np.where(r6_2, cat6 + pts_6_2, cat6)

        # v1.2.0: Removed n_providers condition (temporal confounding).
        # Bureau stress alone (cash_stress >= 3) retained.
        r6_3_raw = (cash_stress >= 3)
        pts_6_3 = np.minimum(10, np.maximum(20 - cat6, 0))
        r6_3 = r6_3_raw & (pts_6_3 > 0)
        cat6 = np.where(r6_3, cat6 + pts_6_3, cat6)

        r6_4_raw = contradiction > 0
        pts_6_4 = np.minimum(8, np.maximum(20 - cat6, 0))
        r6_4 = r6_4_raw & (pts_6_4 > 0)
        cat6 = np.where(r6_4, cat6 + pts_6_4, cat6)
        cat6 = np.minimum(cat6, 20)

        # --- Category 7: Entity Graph (max 40) — OPTIONAL ---
        # Only fires when self._graph_lookup is loaded. Otherwise cat7 = 0.
        cat7 = np.zeros(n, dtype=np.int32)
        r7_1 = np.zeros(n, dtype=bool)  # prior default by connected entity
        r7_2 = np.zeros(n, dtype=bool)  # multiple connected entities (3+)
        r7_3 = np.zeros(n, dtype=bool)  # cross-entity borrower
        r7_2_active = np.zeros(n, dtype=bool)  # after cap check
        r7_3_active = np.zeros(n, dtype=bool)  # after cap check
        pts_7_2 = np.zeros(n, dtype=np.int32)
        pts_7_3 = np.zeros(n, dtype=np.int32)
        graph_connected_count = np.zeros(n, dtype=np.int32)

        if self._graph_lookup is not None and len(self._graph_lookup) > 0:
            # Use the configured lookup key column, falling back to shop_id
            lookup_key_col = self._graph_lookup_key
            key_arr = _get_obj_arr(lookup_key_col) if lookup_key_col in cols else None
            fallback_arr = _get_obj_arr("shop_id") if "shop_id" in cols else None

            for i in range(n):
                # Try primary key, fall back to shop_id
                kid = key_arr[i] if key_arr is not None else None
                if kid is None or (isinstance(kid, float) and math.isnan(kid)):
                    kid = fallback_arr[i] if fallback_arr is not None else None
                if kid is not None and not (isinstance(kid, float) and math.isnan(kid)):
                    entry = self._graph_lookup.get(str(kid))
                    if entry is not None:
                        if entry.get("has_prior_bad", False):
                            r7_1[i] = True
                        cc = entry.get("connected_count", 0)
                        graph_connected_count[i] = cc
                        if cc >= 3:
                            r7_2[i] = True
                        if entry.get("is_cross_entity", False):
                            r7_3[i] = True

            # Rule 7.1: Prior default by connected entity (+25)
            cat7 = np.where(r7_1, 25, 0).astype(np.int32)

            # Rule 7.2: Multiple connected entities (3+) (+10, cap-aware)
            pts_7_2 = np.minimum(10, np.maximum(40 - cat7, 0))
            r7_2_active = r7_2 & (pts_7_2 > 0)
            cat7 = np.where(r7_2_active, cat7 + pts_7_2, cat7)

            # Rule 7.3: Cross-entity borrower (+5, cap-aware)
            pts_7_3 = np.minimum(5, np.maximum(40 - cat7, 0))
            r7_3_active = r7_3 & (pts_7_3 > 0)
            cat7 = np.where(r7_3_active, cat7 + pts_7_3, cat7)

        cat7 = np.minimum(cat7, 40)

        # =================================================================
        # Step 4: Compute totals and decisions (pure numpy)
        # =================================================================
        raw_scores = (cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7).astype(int)
        max_score = MAX_RAW_SCORE_WITH_GRAPH if self.graph_loaded else MAX_RAW_SCORE
        normalized_scores = np.round(raw_scores / max_score, 4)

        reject_thresh = self.thresholds["reject"]
        review_thresh = self.thresholds["review"]
        decisions = np.where(
            raw_scores >= reject_thresh,
            "decline",
            np.where(raw_scores >= review_thresh, "review", "pass"),
        )

        # =================================================================
        # Step 5: Generate reason strings (batch optimized)
        # =================================================================
        # We build reasons per-row using pre-computed masks and vectorized
        # string formatting. This section uses loops over RULES (not rows),
        # with numpy fancy indexing to gather affected row indices.
        reasons_lists = [[] for _ in range(n)]

        # Each rule entry: (mask, format_fn_or_str)
        # We process ~22 rules; for each rule we iterate only over the
        # rows where it fires (sparse).
        _rule_specs = [
            # Cat 1 — v1.2.0: Tiered missing-QI (LFLT and mid-FICO removed)
            (qi_missing_honeybook, None, "MISSING_QI_HONEYBOOK: Missing QI + HONEYBOOK partner (+30)"),
            (qi_missing_low_fico, None, "MISSING_QI_LOW_FICO: Missing QI + FICO<600 (+30)"),
            # qi_missing_mid_fico removed (bad rate 2.78% < base rate)
            # Range widened from 650-699 to 600-699
            (qi_missing_upper_mid_fico, None, "MISSING_QI_MID_FICO: Missing QI + FICO 600-699 (+20)"),
            (qi_missing_high_fico, None, "MISSING_QI_HIGH_FICO: Missing QI + FICO 700+ (+10)"),
            # LFLT rule removed (bad rate 4.61% < base rate)
            (r1_3, "giact_bk", None),
            # Cat 2
            (r2_1, "taktile_fail", None),
            (r2_2, "taktile_fraud", None),
            (r2_3, "phone_trust", None),
            (r2_4, "prove_nv", None),
            # Cat 3
            (r3_1a, "fico_lt550", None),
            (r3_1b, "fico_550_600", None),
            (r3_2, "neg_balance", None),
            (r3_3, "high_nsf", None),
            (r3_4, "high_pastdue", None),
            (r3_5, "ext_neg_bal", None),
            # Cat 4
            (r4_1, "delin60", None),
            (r4_2, "high_inq", None),
            (r4_3, "low_tl", None),
            (r4_4, "high_delin_rate", None),
            # Cat 5
            (r5_1, None, "All providers missing (+15)"),
            (r5_2, "min_coverage", None),
            (r5_3a, None, "Plaid missing (+3)"),
            (r5_3b, None, "Prove missing (+3)"),
            # Cat 6 — v1.2.0: QI interaction (600-649 tier removed)
            (r6_1a, "qi_low_fico", None),
            # r6_1b removed (bad rate 2.78%)
            (r6_1c, "honeybook_qi_interaction", None),
            (r6_2, "paysafe_fico", None),
            (r6_3, "stress_gap", None),
            (r6_4, "provider_disagree", None),
            # Cat 7 (Entity Graph — only fires when graph lookup is loaded)
            (r7_1, None, "Prior default by connected entity (+25)"),
            (r7_2_active if self.graph_loaded else np.zeros(n, dtype=bool), "graph_multi", None),
            (r7_3_active if self.graph_loaded else np.zeros(n, dtype=bool), None, "Cross-entity borrower (+5)"),
        ]

        # Pre-convert float arrays to int arrays for formatting (avoid per-element int())
        fico_int = fico.astype(np.int64)
        d60_int = d60.astype(np.int64)
        inq6_int = inq6.astype(np.int64)
        tl_total_int = tl_total.astype(np.int64)
        nsf_int = nsf_count.astype(np.int64)
        bk_int = bankruptcy_count.astype(np.int64)
        phone_trust_int = phone_trust.astype(np.int64)

        for mask, fmt_key, const_str in _rule_specs:
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue

            if const_str is not None:
                # Constant string -- no per-row formatting needed
                for i in idxs:
                    reasons_lists[i].append(const_str)
            elif fmt_key == "giact_bk":
                for i in idxs:
                    reasons_lists[i].append(f"GIACT bankruptcy count={bk_int[i]} (+{pts_1_3[i]})")
            elif fmt_key == "taktile_fail":
                for i in idxs:
                    reasons_lists[i].append(f"Taktile status FAIL ({taktile_status_arr[i]}) (+20)")
            elif fmt_key == "taktile_fraud":
                for i in idxs:
                    reasons_lists[i].append(f"Taktile fraud flag (+{pts_2_2[i]})")
            elif fmt_key == "phone_trust":
                for i in idxs:
                    reasons_lists[i].append(f"Low phone trust score={phone_trust_int[i]} (+{pts_2_3[i]})")
            elif fmt_key == "prove_nv":
                for i in idxs:
                    reasons_lists[i].append(f"Prove: not verified (+{pts_2_4[i]})")
            elif fmt_key == "fico_lt550":
                for i in idxs:
                    reasons_lists[i].append(f"FICO < 550 ({fico_int[i]}) (+15)")
            elif fmt_key == "fico_550_600":
                for i in idxs:
                    reasons_lists[i].append(f"FICO 550-600 ({fico_int[i]}) (+8)")
            elif fmt_key == "neg_balance":
                for i in idxs:
                    reasons_lists[i].append(f"Negative bank balance=${plaid_balance[i]:,.0f} (+{pts_3_2[i]})")
            elif fmt_key == "high_nsf":
                for i in idxs:
                    reasons_lists[i].append(f"High NSFs={nsf_int[i]} (+{pts_3_3[i]})")
            elif fmt_key == "high_pastdue":
                for i in idxs:
                    reasons_lists[i].append(f"High past due=${pastdue[i]:,.0f} (+{pts_3_4[i]})")
            elif fmt_key == "ext_neg_bal":
                neg_bal_days_int = neg_bal_days.astype(np.int64)
                for i in idxs:
                    reasons_lists[i].append(f"Extended negative balance history ({neg_bal_days_int[i]} days in 90d) (+{pts_3_5[i]})")
            elif fmt_key == "delin60":
                for i in idxs:
                    reasons_lists[i].append(f"Severe delinquencies (60d+)={d60_int[i]} (+10)")
            elif fmt_key == "high_inq":
                for i in idxs:
                    reasons_lists[i].append(f"High inquiries (6mo)={inq6_int[i]} (+{pts_4_2[i]})")
            elif fmt_key == "low_tl":
                for i in idxs:
                    reasons_lists[i].append(f"Zero/minimal tradelines={tl_total_int[i]} (+{pts_4_3[i]})")
            elif fmt_key == "high_delin_rate":
                for i in idxs:
                    reasons_lists[i].append(f"High delinquency rate={delin_rate[i]:.1%} (+{pts_4_4[i]})")
            elif fmt_key == "min_coverage":
                for i in idxs:
                    reasons_lists[i].append(f"Minimal provider coverage ({n_providers[i]}/4) (+8)")
            elif fmt_key == "qi_low_fico":
                for i in idxs:
                    reasons_lists[i].append(f"INTERACTION_QI_LOW_FICO: Missing QI + FICO<600 ({fico_int[i]}) (+20)")
            elif fmt_key == "qi_mid_fico_interaction":
                # Dead code — r6_1b is always False after v1.2.0
                for i in idxs:
                    reasons_lists[i].append(f"INTERACTION_QI_MID_FICO: Missing QI + FICO 600-649 ({fico_int[i]}) (+12)")
            elif fmt_key == "honeybook_qi_interaction":
                for i in idxs:
                    reasons_lists[i].append(f"INTERACTION_HONEYBOOK_QI: HONEYBOOK + Missing QI + low FICO ({fico_int[i]}) (+{pts_6_1c[i]})")
            elif fmt_key == "paysafe_fico":
                for i in idxs:
                    reasons_lists[i].append(f"PAYSAFE + Low FICO ({fico_int[i]}) (+{pts_6_2[i]})")
            elif fmt_key == "stress_gap":
                for i in idxs:
                    reasons_lists[i].append(f"Bureau stress ({cash_stress[i]}) (+{pts_6_3[i]})")
            elif fmt_key == "provider_disagree":
                for i in idxs:
                    reasons_lists[i].append(f"Provider disagreement ({contradiction[i]} contradictions) (+{pts_6_4[i]})")
            elif fmt_key == "graph_multi":
                for i in idxs:
                    reasons_lists[i].append(f"Multiple connected entities ({graph_connected_count[i]}) (+{pts_7_2[i]})")

        return pd.DataFrame(
            {
                "score": raw_scores,
                "normalized_score": normalized_scores,
                "decision": decisions,
                "reasons": reasons_lists,
            },
            index=df.index,
        )

    # ------------------------------------------------------------------
    # Public API: evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        y_col: str = "is_bad",
    ) -> Dict[str, Any]:
        """Evaluate the fraud gate on labeled data.

        Parameters
        ----------
        df : pd.DataFrame
            Data with features and a binary label column.
        y_col : str
            Name of the binary target column.

        Returns
        -------
        dict with metrics: auc, gini, ks, brier, lift_top_decile,
        top_decile_bad_rate, bot_decile_bad_rate, base_rate,
        decision_stats, confusion matrices, etc.
        """
        # Ensure label exists
        if y_col not in df.columns:
            raise ValueError(f"Label column '{y_col}' not found in DataFrame.")

        y_true = df[y_col].values.astype(int)
        results = self.predict_batch(df)
        scores = results["score"].values.astype(float)
        decisions = results["decision"].values
        normalized = results["normalized_score"].values.astype(float)

        # AUC
        auc = roc_auc_score(y_true, scores)
        gini = 2 * auc - 1

        # KS statistic
        fpr, tpr, _ = roc_curve(y_true, scores)
        ks = float(np.max(tpr - fpr))

        # Brier (on normalized scores)
        brier = float(brier_score_loss(y_true, normalized))

        # Lift analysis by decile
        df_eval = pd.DataFrame({"y": y_true, "score": scores})
        df_eval["decile"] = pd.qcut(
            df_eval["score"], 10, labels=False, duplicates="drop"
        )
        top_decile = df_eval["decile"].max()
        bot_decile = df_eval["decile"].min()
        top_bad_rate = df_eval[df_eval["decile"] == top_decile]["y"].mean()
        bot_bad_rate = df_eval[df_eval["decile"] == bot_decile]["y"].mean()
        base_rate = float(y_true.mean())
        lift_top = top_bad_rate / base_rate if base_rate > 0 else 0

        # Decision distribution
        decision_stats = {}
        for dec in ["pass", "review", "decline"]:
            mask = decisions == dec
            n = int(mask.sum())
            n_bad = int(y_true[mask].sum()) if n > 0 else 0
            bad_rate = float(y_true[mask].mean()) if n > 0 else 0.0
            decision_stats[dec] = {
                "count": n,
                "pct": round(n / len(y_true) * 100, 1),
                "n_bad": n_bad,
                "bad_rate": round(bad_rate, 4),
            }

        # Precision/recall at reject threshold
        y_pred_reject = (scores >= self.thresholds["reject"]).astype(int)
        if y_pred_reject.sum() > 0 and y_pred_reject.sum() < len(y_pred_reject):
            cm = confusion_matrix(y_true, y_pred_reject)
            tn, fp, fn, tp = cm.ravel()
            precision_at_reject = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_at_reject = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            precision_at_reject = 0
            recall_at_reject = 0

        return {
            "auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks": round(ks, 4),
            "brier": round(brier, 4),
            "lift_top_decile": round(lift_top, 2),
            "top_decile_bad_rate": round(float(top_bad_rate), 4),
            "bot_decile_bad_rate": round(float(bot_bad_rate), 4),
            "base_rate": round(base_rate, 4),
            "n_obs": len(y_true),
            "n_bad": int(y_true.sum()),
            "decision_stats": decision_stats,
            "precision_at_reject": round(precision_at_reject, 4),
            "recall_at_reject": round(recall_at_reject, 4),
            "thresholds": self.thresholds.copy(),
            "threshold_profile": self.threshold_profile,
        }

    # ------------------------------------------------------------------
    # Public API: set_thresholds
    # ------------------------------------------------------------------

    def set_thresholds(
        self,
        profile: Optional[str] = None,
        review: Optional[int] = None,
        reject: Optional[int] = None,
    ) -> "FraudGate":
        """Update decision thresholds.

        Parameters
        ----------
        profile : str, optional
            Use a named profile ("conservative", "moderate", "aggressive").
        review : int, optional
            Custom review threshold (overrides profile).
        reject : int, optional
            Custom reject threshold (overrides profile).

        Returns
        -------
        self
        """
        if profile is not None:
            if profile not in THRESHOLDS:
                raise ValueError(f"Unknown profile '{profile}'.")
            self.thresholds = THRESHOLDS[profile].copy()
            self.threshold_profile = profile

        if review is not None:
            self.thresholds["review"] = review
        if reject is not None:
            self.thresholds["reject"] = reject

        return self

    # ------------------------------------------------------------------
    # Public API: save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialize FraudGate to disk with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": self.VERSION,
            "thresholds": self.thresholds,
            "threshold_profile": self.threshold_profile,
            "fitted": self._fitted,
            "train_score_percentiles": self._train_score_percentiles,
            "train_score_stats": self._train_score_stats,
            "graph_lookup": self._graph_lookup,
            "graph_lookup_key": self._graph_lookup_key,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FraudGate":
        """Deserialize a FraudGate from disk.

        If the saved state does not contain entity graph data, attempts to
        auto-discover it from well-known companion files:
          1. ``entity_graph.joblib`` next to the loaded file
          2. ``fraud_gate_entity.joblib`` next to the loaded file
          3. ``../data/entity_graph_cross.parquet`` relative to the model dir
        This keeps backward compatibility — FraudGate still works without
        the graph; Category 7 rules are simply skipped.
        """
        path = Path(path)
        state = joblib.load(path)

        # Version check (warn, don't error, for backward compatibility)
        saved_version = state.get("version", "unknown")
        if saved_version != cls.VERSION:
            logger.warning(
                f"FraudGate version mismatch: saved={saved_version}, "
                f"current={cls.VERSION}. Model behavior may differ."
            )

        fg = cls.__new__(cls)
        fg.thresholds = state["thresholds"]
        fg.threshold_profile = state["threshold_profile"]
        fg._fitted = state.get("fitted", False)
        fg._train_score_percentiles = state.get("train_score_percentiles")
        fg._train_score_stats = state.get("train_score_stats")
        fg._graph_lookup = state.get("graph_lookup")
        fg._graph_lookup_key = state.get("graph_lookup_key", "shop_id")

        # Auto-discover entity graph if not embedded in the saved state
        if not fg.graph_loaded:
            fg._auto_load_entity_graph(path.parent)

        return fg

    @staticmethod
    def _auto_load_entity_graph_from_parquet(
        parquet_path: Path,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Build graph lookup from an entity graph parquet file."""
        try:
            entity_df = pd.read_parquet(parquet_path)
            lookup = FraudGate.build_graph_lookup(entity_df, key_col="application_id")
            logger.info(
                "Auto-loaded entity graph from %s (%d entries)",
                parquet_path, len(lookup),
            )
            return lookup
        except Exception as e:
            logger.debug("Failed to load entity graph from %s: %s", parquet_path, e)
            return None

    def _auto_load_entity_graph(self, model_dir: Path) -> None:
        """Try to auto-discover and load entity graph data.

        Search order:
          1. entity_graph.joblib in model_dir (companion file saved by Pipeline)
          2. fraud_gate_entity.joblib in model_dir (entity-specific FraudGate)
          3. ../data/entity_graph_cross.parquet relative to model_dir
        """
        # 1. Companion entity_graph.joblib
        companion = model_dir / "entity_graph.joblib"
        if companion.exists():
            try:
                data = joblib.load(companion)
                self._graph_lookup = data.get("graph_lookup")
                self._graph_lookup_key = data.get(
                    "graph_lookup_key", self._graph_lookup_key
                )
                if self.graph_loaded:
                    logger.info(
                        "Auto-loaded entity graph from %s (%d entries)",
                        companion, len(self._graph_lookup),
                    )
                    return
            except Exception as e:
                logger.debug("Failed to load %s: %s", companion, e)

        # 2. fraud_gate_entity.joblib (has graph embedded)
        entity_fg = model_dir / "fraud_gate_entity.joblib"
        if entity_fg.exists():
            try:
                entity_state = joblib.load(entity_fg)
                gl = entity_state.get("graph_lookup")
                if gl:
                    self._graph_lookup = gl
                    self._graph_lookup_key = entity_state.get(
                        "graph_lookup_key", self._graph_lookup_key
                    )
                    logger.info(
                        "Auto-loaded entity graph from %s (%d entries)",
                        entity_fg, len(self._graph_lookup),
                    )
                    return
            except Exception as e:
                logger.debug("Failed to load %s: %s", entity_fg, e)

        # 3. Entity graph parquet in ../data/
        parquet_path = model_dir.parent / "data" / "entity_graph_cross.parquet"
        if parquet_path.exists():
            lookup = self._auto_load_entity_graph_from_parquet(parquet_path)
            if lookup:
                self._graph_lookup = lookup
                self._graph_lookup_key = "application_id"
                return

        logger.debug(
            "No entity graph found for auto-loading in %s "
            "(Category 7 rules will be inactive)", model_dir,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "unfitted"
        graph_str = f", graph={len(self._graph_lookup)} entries" if self.graph_loaded else ""
        return (
            f"FraudGate(profile='{self.threshold_profile}', "
            f"review={self.thresholds['review']}, "
            f"reject={self.thresholds['reject']}, "
            f"{fitted_str}{graph_str})"
        )
