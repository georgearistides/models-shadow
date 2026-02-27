"""
Pipeline v6.0 — Simplified scoring pipeline.
=============================================
Chains: FraudRules → DefaultModel (WoE) → Grader → Decision.

Replaces the 5-model v5.7 Pipeline with a clean 3-stage system.
AUC cost: -0.004 (0.6559 vs 0.6597). Lines: ~250 vs ~1,700.

Usage:
    from pipeline_v6 import PipelineV6

    pipe = PipelineV6.load(".")  # loads woe_model.joblib, entity_graph_cross.parquet, config.joblib
    result = pipe.score({"fico": 620, "qi": "MISSING", ...})
    results_df = pipe.score_batch(df)
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from fraud_rules import FraudRules
from default_model import DefaultModel
from grader import Grader
from feature_engine import engineer_bureau_features, engineer_qi_partner_features, engineer_entity_features

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result from scoring a single application."""
    decision: str           # "approve", "review", or "decline"
    pd: float               # calibrated probability of default
    grade: str              # letter grade (A/B/C/D)
    fraud_rules: List[str]  # list of triggered fraud rules
    fraud_decision: str     # "approve", "review", or "decline" from rules


class PipelineV6:
    """Simplified scoring pipeline: rules → WoE PD → grade → decision."""

    # Decision thresholds
    PD_DECLINE = 0.25
    PD_REVIEW = 0.15

    def __init__(self, fraud_rules: FraudRules, default_model: DefaultModel, grader: Grader):
        self.fraud_rules = fraud_rules
        self.default_model = default_model
        self.grader = grader
        self.version = "6.0"

    @classmethod
    def load(cls, model_dir: str) -> "PipelineV6":
        """Load all components from a directory."""
        model_dir = Path(model_dir)

        # Load WoE model
        woe_path = model_dir / "woe_model.joblib"
        default_model = DefaultModel.load(str(woe_path))

        # Load fraud rules with entity graph
        entity_path = model_dir / "entity_graph_cross.parquet"
        fraud_rules_obj = FraudRules.load(str(entity_path) if entity_path.exists() else None)

        # Load grader config (or use defaults)
        config_path = model_dir / "config.joblib"
        if config_path.exists():
            config = joblib.load(config_path)
            grader = Grader.from_config(config.get("grader", {}))
        else:
            grader = Grader()

        return cls(fraud_rules=fraud_rules_obj, default_model=default_model, grader=grader)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer derived features needed by the WoE model."""
        df = df.copy()

        # Alias common column names
        if "experian_FICO_SCORE" in df.columns and "fico" not in df.columns:
            df["fico"] = df["experian_FICO_SCORE"]
        if "shop_qi" in df.columns and "qi" not in df.columns:
            df["qi"] = df["shop_qi"]

        # Fill missing numeric columns with 0
        numeric_cols = ["d30", "d60", "inq6", "revutil", "pastdue",
                        "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
                        "mopmt", "crhist"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Fill missing categoricals
        if "qi" in df.columns:
            df["qi"] = df["qi"].fillna("MISSING")
        if "has_prior_bad" not in df.columns:
            df["has_prior_bad"] = 0
        if "is_repeat" not in df.columns:
            df["is_repeat"] = 0

        # Engineer derived features
        df = engineer_bureau_features(df)
        df = engineer_qi_partner_features(df)
        df = engineer_entity_features(df)

        return df

    def score(self, row: dict) -> ScoringResult:
        """Score a single application."""
        # Step 1: Fraud rules
        fraud_result = self.fraud_rules.check(row)

        # Step 2: PD estimation (always compute, even for fraud declines — for shadow comparison)
        row_df = pd.DataFrame([row])
        row_df = self._prepare_features(row_df)
        pd_value = float(self.default_model.predict_pd_batch(row_df).iloc[0])

        # Step 3: Grade
        grade = self.grader.assign(pd_value)

        # Step 4: Decision
        if fraud_result.decision == "decline":
            decision = "decline"
        elif pd_value > self.PD_DECLINE:
            decision = "decline"
        elif pd_value > self.PD_REVIEW or grade == "D":
            decision = "review"
        elif fraud_result.decision == "review":
            decision = "review"
        else:
            decision = "approve"

        return ScoringResult(
            decision=decision,
            pd=pd_value,
            grade=grade,
            fraud_rules=fraud_result.rules_triggered,
            fraud_decision=fraud_result.decision,
        )

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of applications.

        Returns DataFrame with columns:
            pd, grade, decision, fraud_decision, fraud_rules_triggered
        """
        # Step 1: Fraud rules (row-by-row — only 6 rules, fast enough)
        fraud_results = self.fraud_rules.check_batch(df)

        # Step 2: Feature engineering + PD estimation (vectorized)
        df_feat = self._prepare_features(df)
        pds = self.default_model.predict_pd_batch(df_feat)

        # Step 3: Grades (vectorized)
        grades = self.grader.assign_batch(pds)

        # Step 4: Decisions (vectorized)
        decisions = pd.Series("approve", index=df.index)

        # Fraud declines take priority
        fraud_decline = fraud_results["fraud_decision"] == "decline"
        decisions[fraud_decline] = "decline"

        # PD-based declines
        pd_decline = (pds > self.PD_DECLINE) & ~fraud_decline
        decisions[pd_decline] = "decline"

        # Reviews
        pd_review = (pds > self.PD_REVIEW) & (decisions == "approve")
        decisions[pd_review] = "review"

        grade_review = (grades == "D") & (decisions == "approve")
        decisions[grade_review] = "review"

        fraud_review = (fraud_results["fraud_decision"] == "review") & (decisions == "approve")
        decisions[fraud_review] = "review"

        # Build output
        out = pd.DataFrame({
            "pd": pds.values,
            "grade": grades.values,
            "decision": decisions.values,
            "fraud_decision": fraud_results["fraud_decision"].values,
            "fraud_rules_triggered": fraud_results["fraud_rules_triggered"].values,
        }, index=df.index)

        return out
