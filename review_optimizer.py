"""
ReviewOptimizer — Review queue prioritization system that ranks loans in the
"review" bucket by expected default probability to optimize manual review effort.

From the A/B simulation results:
  - ~48.5% of apps go to "review" (the largest bucket)
  - Review queue holds 56% of all defaults
  - Review loans with grade D/E capture 73% of review bads at 17.3% bad rate
  - Goal: rank review-queue loans so reviewers focus on highest-risk ones first

Usage:
    from review_optimizer import ReviewOptimizer

    ro = ReviewOptimizer()
    prioritized = ro.prioritize(review_results)
    summary = ro.get_review_summary(review_results)
    capacity = ro.recommend_capacity(review_results, daily_reviews=20)

The optimizer works on Pipeline.score_batch() output (or shadow score CSV data).
Each row needs at minimum: pd, fraud_score/fraud_normalized, grade, decision.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# Maximum possible raw fraud score (used for normalization when no
# pre-normalized score is available).  Matches FraudGate.MAX_RAW_SCORE_WITH_GRAPH.
FRAUD_SCORE_NORMALIZATION_MAX = 180.0

# ---------------------------------------------------------------------------
# Review tier thresholds
# ---------------------------------------------------------------------------

# Tier definitions — each loan is assigned to the first tier whose ANY
# condition matches, checked in order (urgent first).
TIER_CONFIG = {
    "urgent": {
        "description": "Highest risk — likely to default without intervention",
        "conditions": {
            "grade": ["E"],
            "pd_min": 0.20,
            "fraud_score_min": 80,
        },
        "color": "red",
        "estimated_review_minutes": 10,
    },
    "high": {
        "description": "Elevated risk — strong default signals present",
        "conditions": {
            "grade": ["D"],
            "pd_min": 0.15,
            "fraud_score_min": 60,
        },
        "color": "orange",
        "estimated_review_minutes": 15,
    },
    "medium": {
        "description": "Moderate risk — some concerning indicators",
        "conditions": {
            "grade": ["C"],
            "pd_min": 0.10,
        },
        "color": "yellow",
        "estimated_review_minutes": 20,
    },
    "low": {
        "description": "Lower risk — in review due to grade/threshold rules",
        "conditions": {},
        "color": "green",
        "estimated_review_minutes": 15,
    },
}

# Maximum number of reason codes per loan
MAX_REASONS = 3


class ReviewOptimizer:
    """Ranks review-queue loans by expected default probability.

    Prioritization logic:
        - Primary sort: calibrated PD (from DefaultScorecard)
        - Secondary factors: grade (D/E get priority), fraud score (borderline
          declines), QI status (MISSING = high risk)
        - Review tiers: urgent / high / medium / low
        - Reason codes: top 3 factors driving the review priority

    Parameters
    ----------
    version : str
        Version identifier for tracking.
    composite_weights : dict, optional
        Weights for the composite risk score used for ranking.
        Keys: 'pd', 'fraud', 'grade', 'qi'. Defaults to PD-dominant blend.
    """

    def __init__(
        self,
        version: str = VERSION,
        composite_weights: Optional[Dict[str, float]] = None,
    ):
        self.version = version
        self.composite_weights = composite_weights or {
            "pd": 0.60,
            "fraud": 0.20,
            "grade": 0.15,
            "qi": 0.05,
        }
        # Validate weights sum to 1
        total = sum(self.composite_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Composite weights sum to %.3f (expected 1.0). Normalizing.", total
            )
            for k in self.composite_weights:
                self.composite_weights[k] /= total

    # ------------------------------------------------------------------
    # Column name resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_col(row_or_df, candidates: list, default=None):
        """Find the first available column name from candidates."""
        if isinstance(row_or_df, dict):
            for c in candidates:
                if c in row_or_df and row_or_df[c] is not None:
                    return c
        elif isinstance(row_or_df, pd.DataFrame):
            for c in candidates:
                if c in row_or_df.columns:
                    return c
        return default

    @staticmethod
    def _get_val(row: dict, candidates: list, default=None):
        """Get value from dict trying multiple column names."""
        for c in candidates:
            v = row.get(c)
            if v is not None:
                try:
                    if isinstance(v, float) and math.isnan(v):
                        continue
                except (TypeError, ValueError):
                    pass
                return v
        return default

    # ------------------------------------------------------------------
    # Tier assignment
    # ------------------------------------------------------------------

    def _assign_tier(self, row: dict) -> str:
        """Assign a review tier to a single loan based on its risk signals.

        Tiers are checked in priority order: urgent -> high -> medium -> low.
        A loan matches a tier if **any** of that tier's conditions is met
        (OR logic).  This ensures consistent behaviour across all tiers --
        a single strong risk signal is enough to escalate the loan.
        """
        pd_val = self._get_val(row, ["pd", "our_pd"], default=0.0)
        grade = self._get_val(row, ["grade", "our_grade"], default="C")
        fraud_score = self._get_val(row, ["fraud_score", "our_fraud_score"], default=0)

        try:
            pd_val = float(pd_val)
        except (TypeError, ValueError):
            pd_val = 0.0

        try:
            fraud_score = float(fraud_score)
        except (TypeError, ValueError):
            fraud_score = 0.0

        grade = str(grade).strip().upper() if grade else "C"

        # Check urgent
        if (
            grade in TIER_CONFIG["urgent"]["conditions"].get("grade", [])
            or pd_val >= TIER_CONFIG["urgent"]["conditions"].get("pd_min", 999)
            or fraud_score >= TIER_CONFIG["urgent"]["conditions"].get("fraud_score_min", 999)
        ):
            return "urgent"

        # Check high
        if (
            grade in TIER_CONFIG["high"]["conditions"].get("grade", [])
            or pd_val >= TIER_CONFIG["high"]["conditions"].get("pd_min", 999)
            or fraud_score >= TIER_CONFIG["high"]["conditions"].get("fraud_score_min", 999)
        ):
            return "high"

        # Check medium (OR logic — matches if ANY condition is met)
        cond = TIER_CONFIG["medium"]["conditions"]
        if (
            grade in cond.get("grade", [])
            or pd_val >= cond.get("pd_min", 999)
        ):
            return "medium"

        return "low"

    # ------------------------------------------------------------------
    # Composite risk score
    # ------------------------------------------------------------------

    def _compute_composite_score(self, row: dict) -> float:
        """Compute a composite risk score for sorting within and across tiers.

        Score is in [0, 1] range, higher = riskier.

        Components:
          - pd: directly from calibrated PD (already in [0,1])
          - fraud: normalized fraud score (0-1)
          - grade: mapped to numeric scale (A=0.1, B=0.3, C=0.5, D=0.7, E=0.9)
          - qi: MISSING=1.0, other=0.0
        """
        w = self.composite_weights

        # PD component
        pd_val = self._get_val(row, ["pd", "our_pd"], default=0.0)
        try:
            pd_val = float(pd_val)
        except (TypeError, ValueError):
            pd_val = 0.0
        pd_component = min(pd_val, 1.0)

        # Fraud component (normalize to 0-1 using normalized score if available,
        # else use raw score / 180 as rough normalization)
        fraud_norm = self._get_val(row, ["fraud_normalized", "our_fraud_normalized"], default=None)
        if fraud_norm is not None:
            try:
                fraud_component = float(fraud_norm)
            except (TypeError, ValueError):
                fraud_component = 0.0
        else:
            fraud_raw = self._get_val(row, ["fraud_score", "our_fraud_score"], default=0)
            try:
                fraud_component = min(float(fraud_raw) / FRAUD_SCORE_NORMALIZATION_MAX, 1.0)
            except (TypeError, ValueError):
                fraud_component = 0.0

        # Grade component
        grade = self._get_val(row, ["grade", "our_grade"], default="C")
        grade_map = {"A": 0.1, "B": 0.3, "C": 0.5, "D": 0.7, "E": 0.9, "F": 0.95, "G": 1.0}
        grade_component = grade_map.get(str(grade).strip().upper(), 0.5)

        # QI component
        qi = self._get_val(row, ["qi", "shop_qi"], default=None)
        qi_component = 1.0 if qi is not None and str(qi).upper() == "MISSING" else 0.0

        composite = (
            w["pd"] * pd_component
            + w["fraud"] * fraud_component
            + w["grade"] * grade_component
            + w["qi"] * qi_component
        )
        return round(composite, 6)

    # ------------------------------------------------------------------
    # Reason codes
    # ------------------------------------------------------------------

    def _generate_reasons(self, row: dict) -> List[str]:
        """Generate the top reasons why this loan needs review, sorted by severity.

        Returns at most MAX_REASONS reason strings.
        """
        reasons = []

        pd_val = self._get_val(row, ["pd", "our_pd"], default=0.0)
        try:
            pd_val = float(pd_val)
        except (TypeError, ValueError):
            pd_val = 0.0

        grade = self._get_val(row, ["grade", "our_grade"], default="C")
        grade_str = str(grade).strip().upper() if grade else "C"

        fraud_score = self._get_val(row, ["fraud_score", "our_fraud_score"], default=0)
        try:
            fraud_score = float(fraud_score)
        except (TypeError, ValueError):
            fraud_score = 0.0

        fico = self._get_val(row, ["fico", "j_latest_fico_score", "experian_FICO_SCORE"], default=None)
        qi = self._get_val(row, ["qi", "shop_qi"], default=None)
        partner = self._get_val(row, ["partner"], default=None)

        # Build (priority, reason) pairs — lower priority number = more important
        scored_reasons = []

        # PD-based reasons
        if pd_val >= 0.25:
            scored_reasons.append((1, f"Very high default probability ({pd_val:.1%})"))
        elif pd_val >= 0.15:
            scored_reasons.append((2, f"High default probability ({pd_val:.1%})"))
        elif pd_val >= 0.10:
            scored_reasons.append((4, f"Elevated default probability ({pd_val:.1%})"))

        # Grade-based reasons
        if grade_str == "E":
            scored_reasons.append((1, "Worst credit grade (E)"))
        elif grade_str == "D":
            scored_reasons.append((3, "Poor credit grade (D)"))
        elif grade_str in ("F", "G"):
            scored_reasons.append((2, f"Review-trigger grade ({grade_str})"))

        # Fraud score reasons
        if fraud_score >= 80:
            scored_reasons.append((1, f"High fraud risk score ({fraud_score:.0f})"))
        elif fraud_score >= 60:
            scored_reasons.append((3, f"Elevated fraud risk score ({fraud_score:.0f})"))
        elif fraud_score >= 40:
            scored_reasons.append((5, f"Moderate fraud risk score ({fraud_score:.0f})"))

        # QI reason
        if qi is not None and str(qi).upper() == "MISSING":
            scored_reasons.append((2, "Missing QI (industry classification unavailable)"))

        # FICO reason
        if fico is not None:
            try:
                fico_val = float(fico)
                if fico_val < 550:
                    scored_reasons.append((2, f"Very low FICO score ({int(fico_val)})"))
                elif fico_val < 600:
                    scored_reasons.append((4, f"Low FICO score ({int(fico_val)})"))
                elif fico_val < 650:
                    scored_reasons.append((6, f"Below-average FICO score ({int(fico_val)})"))
            except (TypeError, ValueError):
                pass

        # Partner reason
        if partner is not None and str(partner).upper() == "PAYSAFE":
            scored_reasons.append((5, "PAYSAFE partner (higher historical risk)"))

        # Fraud decision reasons
        fraud_decision = self._get_val(
            row, ["fraud_decision", "our_fraud_decision"], default="pass"
        )
        if fraud_decision == "review":
            scored_reasons.append((3, "Fraud gate flagged for review"))

        # Sort by priority (lower = more important), take top N
        scored_reasons.sort(key=lambda x: x[0])
        reasons = [r[1] for r in scored_reasons[:MAX_REASONS]]

        # Fallback
        if not reasons:
            reasons = ["Flagged by pipeline review rules"]

        return reasons

    # ------------------------------------------------------------------
    # prioritize
    # ------------------------------------------------------------------

    def prioritize(self, pipeline_results: List[dict]) -> List[dict]:
        """Take a list of Pipeline.score() results in 'review' status and
        return them sorted by priority (highest risk first) with additional
        fields for review queue management.

        Parameters
        ----------
        pipeline_results : list of dict
            Each dict is a pipeline scoring result (from Pipeline.score() or
            a row from score_batch() / shadow CSV). Must include at minimum:
            pd (or our_pd), grade (or our_grade).
            Loans that are NOT in review status are silently filtered out.

        Returns
        -------
        list of dict
            Each dict contains:
            - ...all original pipeline result fields...
            - review_priority: int, 1=highest risk (rank within sorted list)
            - review_tier: str, 'urgent'|'high'|'medium'|'low'
            - review_reasons: list of str, top 3 risk factors
            - estimated_review_time: int, estimated minutes to review
            - composite_risk_score: float, the blended score used for sorting
        """
        if not pipeline_results:
            return []

        # Filter to review-only loans
        review_loans = []
        for r in pipeline_results:
            decision = self._get_val(r, ["decision", "our_decision"], default="")
            if str(decision).lower() == "review":
                review_loans.append(r)

        if not review_loans:
            logger.warning("No loans with 'review' decision found in input.")
            return []

        logger.info(f"Prioritizing {len(review_loans)} review-queue loans...")

        # Enrich each loan with tier, score, reasons
        enriched = []
        for loan in review_loans:
            enriched_loan = dict(loan)
            enriched_loan["review_tier"] = self._assign_tier(loan)
            enriched_loan["composite_risk_score"] = self._compute_composite_score(loan)
            enriched_loan["review_reasons"] = self._generate_reasons(loan)
            enriched_loan["estimated_review_time"] = TIER_CONFIG[
                enriched_loan["review_tier"]
            ]["estimated_review_minutes"]
            enriched.append(enriched_loan)

        # Sort: tier priority (urgent > high > medium > low), then composite score desc
        tier_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        enriched.sort(
            key=lambda x: (tier_order.get(x["review_tier"], 99), -x["composite_risk_score"])
        )

        # Assign priority ranks
        for i, loan in enumerate(enriched):
            loan["review_priority"] = i + 1

        logger.info(
            f"Prioritization complete. Tier distribution: "
            f"urgent={sum(1 for l in enriched if l['review_tier']=='urgent')}, "
            f"high={sum(1 for l in enriched if l['review_tier']=='high')}, "
            f"medium={sum(1 for l in enriched if l['review_tier']=='medium')}, "
            f"low={sum(1 for l in enriched if l['review_tier']=='low')}"
        )

        return enriched

    # ------------------------------------------------------------------
    # get_review_summary
    # ------------------------------------------------------------------

    def get_review_summary(self, pipeline_results: List[dict]) -> dict:
        """Aggregate review queue statistics.

        Parameters
        ----------
        pipeline_results : list of dict
            Full list of pipeline results (all decisions). Review loans
            will be filtered out internally.

        Returns
        -------
        dict with keys:
            total_in_queue: int
            tier_breakdown: dict of tier -> {count, pct, bad_count, bad_rate, avg_pd}
            expected_defaults_by_tier: dict
            cumulative_capture_curve: list of dicts (review_pct, default_capture_pct)
            total_estimated_review_hours: float
            grade_breakdown: dict of grade -> {count, bad_count, bad_rate}
        """
        # First prioritize to get enriched results
        prioritized = self.prioritize(pipeline_results)

        if not prioritized:
            return {
                "total_in_queue": 0,
                "tier_breakdown": {},
                "expected_defaults_by_tier": {},
                "cumulative_capture_curve": [],
                "total_estimated_review_hours": 0.0,
                "grade_breakdown": {},
            }

        total = len(prioritized)

        # Count total defaults in the full pipeline results (for capture rate)
        total_bads_all = 0
        total_all = 0
        for r in pipeline_results:
            total_all += 1
            is_bad = self._get_val(r, ["is_bad"], default=0)
            try:
                if int(is_bad) == 1:
                    total_bads_all += 1
            except (TypeError, ValueError):
                pass

        # --- Tier breakdown ---
        tier_breakdown = {}
        for tier in ["urgent", "high", "medium", "low"]:
            tier_loans = [l for l in prioritized if l["review_tier"] == tier]
            if not tier_loans:
                tier_breakdown[tier] = {
                    "count": 0,
                    "pct": 0.0,
                    "bad_count": 0,
                    "bad_rate": 0.0,
                    "avg_pd": 0.0,
                    "avg_fraud_score": 0.0,
                }
                continue

            n = len(tier_loans)
            bad_count = sum(
                1 for l in tier_loans
                if int(self._get_val(l, ["is_bad"], default=0) or 0) == 1
            )
            pds = [
                float(self._get_val(l, ["pd", "our_pd"], default=0) or 0)
                for l in tier_loans
            ]
            fraud_scores = [
                float(self._get_val(l, ["fraud_score", "our_fraud_score"], default=0) or 0)
                for l in tier_loans
            ]

            tier_breakdown[tier] = {
                "count": n,
                "pct": round(n / total * 100, 1),
                "bad_count": bad_count,
                "bad_rate": round(bad_count / n * 100, 1) if n > 0 else 0.0,
                "avg_pd": round(np.mean(pds), 4),
                "avg_fraud_score": round(np.mean(fraud_scores), 1),
            }

        # --- Expected defaults by tier (based on observed is_bad) ---
        expected_defaults_by_tier = {
            tier: info["bad_count"] for tier, info in tier_breakdown.items()
        }

        # --- Cumulative capture curve ---
        # Sort by composite risk score descending (already done by prioritize)
        cumulative_capture = []
        running_bads = 0
        total_review_bads = sum(info["bad_count"] for info in tier_breakdown.values())

        for pct_target in range(10, 101, 10):
            n_to_review = int(total * pct_target / 100)
            subset = prioritized[:n_to_review]
            bads_captured = sum(
                1 for l in subset
                if int(self._get_val(l, ["is_bad"], default=0) or 0) == 1
            )
            cumulative_capture.append({
                "review_pct": pct_target,
                "loans_reviewed": n_to_review,
                "defaults_captured": bads_captured,
                "capture_pct_of_review_bads": round(
                    bads_captured / total_review_bads * 100, 1
                ) if total_review_bads > 0 else 0.0,
                "capture_pct_of_all_bads": round(
                    bads_captured / total_bads_all * 100, 1
                ) if total_bads_all > 0 else 0.0,
            })

        # --- Total estimated review hours ---
        total_minutes = sum(l["estimated_review_time"] for l in prioritized)
        total_hours = round(total_minutes / 60, 1)

        # --- Grade breakdown ---
        grade_breakdown = {}
        for loan in prioritized:
            grade = self._get_val(loan, ["grade", "our_grade"], default="?")
            grade = str(grade).strip().upper()
            if grade not in grade_breakdown:
                grade_breakdown[grade] = {"count": 0, "bad_count": 0}
            grade_breakdown[grade]["count"] += 1
            is_bad = int(self._get_val(loan, ["is_bad"], default=0) or 0)
            if is_bad == 1:
                grade_breakdown[grade]["bad_count"] += 1

        for grade in grade_breakdown:
            n = grade_breakdown[grade]["count"]
            b = grade_breakdown[grade]["bad_count"]
            grade_breakdown[grade]["bad_rate"] = round(b / n * 100, 1) if n > 0 else 0.0

        return {
            "total_in_queue": total,
            "total_bads_in_queue": total_review_bads,
            "total_bads_in_portfolio": total_bads_all,
            "queue_share_of_defaults": round(
                total_review_bads / total_bads_all * 100, 1
            ) if total_bads_all > 0 else 0.0,
            "tier_breakdown": tier_breakdown,
            "expected_defaults_by_tier": expected_defaults_by_tier,
            "cumulative_capture_curve": cumulative_capture,
            "total_estimated_review_hours": total_hours,
            "grade_breakdown": grade_breakdown,
        }

    # ------------------------------------------------------------------
    # recommend_capacity
    # ------------------------------------------------------------------

    def recommend_capacity(
        self, pipeline_results: List[dict], daily_reviews: int = 20
    ) -> dict:
        """Given a capacity constraint, recommend which tiers to review and
        the expected impact.

        Parameters
        ----------
        pipeline_results : list of dict
            Full pipeline results (all decisions).
        daily_reviews : int
            Maximum number of loans a reviewer can review per day.

        Returns
        -------
        dict with:
            daily_capacity: int
            recommended_tiers: list of str (tiers to review)
            recommended_cutoff_priority: int (review loans with priority <= this)
            loans_to_review: int
            days_to_clear_queue: float
            expected_defaults_caught: int
            expected_default_capture_rate: float (fraction of review-queue bads caught)
            expected_all_bads_capture_rate: float (fraction of ALL defaults caught)
            expected_false_positive_rate: float (fraction of reviewed loans that are good)
            expected_precision: float (fraction of reviewed loans that are bad)
            optimal_cutoff: dict with optimal cutoff analysis
            tier_recommendations: list of per-tier recommendations
        """
        prioritized = self.prioritize(pipeline_results)

        if not prioritized:
            return {
                "daily_capacity": daily_reviews,
                "recommended_tiers": [],
                "recommended_cutoff_priority": 0,
                "loans_to_review": 0,
                "days_to_clear_queue": 0.0,
                "expected_defaults_caught": 0,
                "expected_default_capture_rate": 0.0,
                "expected_all_bads_capture_rate": 0.0,
                "expected_false_positive_rate": 0.0,
                "expected_precision": 0.0,
                "optimal_cutoff": {},
                "tier_recommendations": [],
            }

        total_queue = len(prioritized)

        # Count total bads across all pipeline results
        total_bads_all = sum(
            1 for r in pipeline_results
            if int(self._get_val(r, ["is_bad"], default=0) or 0) == 1
        )
        total_review_bads = sum(
            1 for l in prioritized
            if int(self._get_val(l, ["is_bad"], default=0) or 0) == 1
        )

        # --- Per-tier recommendations ---
        tier_recs = []
        cumulative_loans = 0
        cumulative_bads = 0
        cumulative_days = 0.0

        for tier in ["urgent", "high", "medium", "low"]:
            tier_loans = [l for l in prioritized if l["review_tier"] == tier]
            if not tier_loans:
                continue

            n = len(tier_loans)
            bads = sum(
                1 for l in tier_loans
                if int(self._get_val(l, ["is_bad"], default=0) or 0) == 1
            )
            goods = n - bads
            days_needed = math.ceil(n / daily_reviews)

            cumulative_loans += n
            cumulative_bads += bads
            cumulative_days += days_needed

            tier_recs.append({
                "tier": tier,
                "loans": n,
                "defaults": bads,
                "bad_rate": round(bads / n * 100, 1) if n > 0 else 0.0,
                "days_needed": days_needed,
                "cumulative_loans": cumulative_loans,
                "cumulative_defaults": cumulative_bads,
                "cumulative_days": cumulative_days,
                "cumulative_capture_of_review_bads": round(
                    cumulative_bads / total_review_bads * 100, 1
                ) if total_review_bads > 0 else 0.0,
                "cumulative_capture_of_all_bads": round(
                    cumulative_bads / total_bads_all * 100, 1
                ) if total_bads_all > 0 else 0.0,
            })

        # --- Determine recommended cutoff ---
        # Strategy: Always include urgent and high (they have the worst risk).
        # Include additional tiers if their bad rate exceeds the portfolio
        # average AND daily capacity can clear them within a reasonable
        # timeframe.  "Reasonable" is scaled to 90 days (a quarter) since
        # the full queue takes 340+ days at 20/day anyway.
        PORTFOLIO_BAD_RATE = 0.092  # from project context
        MAX_DAYS_TO_CLEAR = 90  # practical quarterly limit

        recommended_tiers = []
        recommended_loans = 0
        recommended_bads = 0

        for rec in tier_recs:
            tier_bad_rate = rec["bad_rate"] / 100
            if tier_bad_rate >= PORTFOLIO_BAD_RATE or rec["tier"] in ("urgent", "high"):
                # Always include urgent and high, plus any tier with above-average risk
                if rec["cumulative_days"] <= MAX_DAYS_TO_CLEAR:
                    recommended_tiers.append(rec["tier"])
                    recommended_loans = rec["cumulative_loans"]
                    recommended_bads = rec["cumulative_defaults"]

        # Fallback: if even urgent exceeds the time limit, still recommend
        # urgent since the reviewer should focus on the riskiest loans.
        if not recommended_tiers and tier_recs:
            recommended_tiers = [tier_recs[0]["tier"]]
            recommended_loans = tier_recs[0]["cumulative_loans"]
            recommended_bads = tier_recs[0]["cumulative_defaults"]

        # Find the priority cutoff (rank of last loan in recommended tiers)
        recommended_cutoff = 0
        for i, loan in enumerate(prioritized):
            if loan["review_tier"] in recommended_tiers:
                recommended_cutoff = i + 1

        # --- Optimal cutoff analysis ---
        # For each possible cutoff N (review top-N loans), compute efficiency
        optimal_analysis = []
        for n in [10, 25, 50, 100, 200, 500, 1000, 2000, 3000, total_queue]:
            if n > total_queue:
                n = total_queue
            subset = prioritized[:n]
            bads_in_subset = sum(
                1 for l in subset
                if int(self._get_val(l, ["is_bad"], default=0) or 0) == 1
            )
            goods_in_subset = n - bads_in_subset

            optimal_analysis.append({
                "top_n": n,
                "pct_of_queue": round(n / total_queue * 100, 1),
                "bads_caught": bads_in_subset,
                "capture_rate_review": round(
                    bads_in_subset / total_review_bads * 100, 1
                ) if total_review_bads > 0 else 0.0,
                "capture_rate_all": round(
                    bads_in_subset / total_bads_all * 100, 1
                ) if total_bads_all > 0 else 0.0,
                "precision": round(
                    bads_in_subset / n * 100, 1
                ) if n > 0 else 0.0,
                "false_positive_rate": round(
                    goods_in_subset / n * 100, 1
                ) if n > 0 else 0.0,
                "days_at_capacity": math.ceil(n / daily_reviews),
            })

        # Remove duplicates from optimal_analysis (if total_queue hit early)
        seen_ns = set()
        deduped = []
        for entry in optimal_analysis:
            if entry["top_n"] not in seen_ns:
                seen_ns.add(entry["top_n"])
                deduped.append(entry)
        optimal_analysis = deduped

        # Compute metrics for the recommended cutoff
        recommended_subset = prioritized[:recommended_loans] if recommended_loans > 0 else []
        rec_goods = recommended_loans - recommended_bads

        return {
            "daily_capacity": daily_reviews,
            "total_in_queue": total_queue,
            "total_review_bads": total_review_bads,
            "total_portfolio_bads": total_bads_all,
            "recommended_tiers": recommended_tiers,
            "recommended_cutoff_priority": recommended_cutoff,
            "loans_to_review": recommended_loans,
            "days_to_clear_queue": math.ceil(recommended_loans / daily_reviews) if daily_reviews > 0 else 0,
            "days_to_clear_full_queue": math.ceil(total_queue / daily_reviews) if daily_reviews > 0 else 0,
            "expected_defaults_caught": recommended_bads,
            "expected_default_capture_rate": round(
                recommended_bads / total_review_bads * 100, 1
            ) if total_review_bads > 0 else 0.0,
            "expected_all_bads_capture_rate": round(
                recommended_bads / total_bads_all * 100, 1
            ) if total_bads_all > 0 else 0.0,
            "expected_false_positive_rate": round(
                rec_goods / recommended_loans * 100, 1
            ) if recommended_loans > 0 else 0.0,
            "expected_precision": round(
                recommended_bads / recommended_loans * 100, 1
            ) if recommended_loans > 0 else 0.0,
            "optimal_cutoff_analysis": optimal_analysis,
            "tier_recommendations": tier_recs,
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ReviewOptimizer(version={self.version}, "
            f"weights={self.composite_weights})"
        )
