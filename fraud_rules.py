"""
FraudRules — Hard rule pre-filter for obvious fraud/risk signals.
=================================================================
Replaces the 24-rule FraudGate with 6 high-signal hard rules.
Parsimony analysis (iter 26) proved only 7 of 29 rules carry signal;
this module keeps the 6 that matter most.

Usage:
    from fraud_rules import FraudRules

    rules = FraudRules.load("entity_graph_cross.parquet")
    result = rules.check({"fico": 540, "qi": "MISSING", "has_prior_bad": 1})
    # -> FraudResult(decision="decline", rules_triggered=["prior_bad_auto_decline", "missing_qi_low_fico"])

    results_df = rules.check_batch(df)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FraudResult:
    """Result from fraud rule check."""
    decision: str  # "approve", "review", or "decline"
    rules_triggered: List[str] = field(default_factory=list)


class FraudRules:
    """Hard rule pre-filter for obvious fraud/risk signals.

    Rules (in priority order):
    1. has_prior_bad → auto-decline (42.86% bad rate, 75 cases)
    2. qi == MISSING AND fico < 600 → auto-decline (44.8% bad rate)
    3. qi == MISSING AND partner == HONEYBOOK → auto-decline (74.4% bad rate)
    4. fico < 550 → review (extreme subprime, 25%+ bad rate)
    5. d60 >= 4 AND fico < 620 → review (bureau stress)
    6. connected_entities >= 3 → review (entity graph cluster)
    """

    def __init__(self, entity_lookup: Optional[dict] = None):
        """
        Args:
            entity_lookup: dict mapping application_id -> {has_prior_bad, connected_entities}
                          Built from entity_graph_cross.parquet.
        """
        self.entity_lookup = entity_lookup or {}

    @classmethod
    def load(cls, entity_parquet_path: str = None):
        """Load FraudRules with entity graph from parquet file."""
        entity_lookup = {}
        if entity_parquet_path:
            path = Path(entity_parquet_path)
            if path.exists():
                edf = pd.read_parquet(path)
                for _, row in edf.iterrows():
                    app_id = str(row.get("application_id", ""))
                    entity_lookup[app_id] = {
                        "has_prior_bad": int(row.get("has_prior_bad", 0) or 0),
                        "connected_entities": int(row.get("prior_loans_ssn", 0) or 0),
                    }
        return cls(entity_lookup=entity_lookup)

    def _get_entity_info(self, app_id):
        """Look up entity graph info for an application."""
        info = self.entity_lookup.get(str(app_id), {})
        return info.get("has_prior_bad", 0), info.get("connected_entities", 0)

    def check(self, row: dict) -> FraudResult:
        """Check a single application against fraud rules.

        Args:
            row: dict with keys: fico, qi, partner, d60, application_id,
                 has_prior_bad (optional — will check entity_lookup if missing)
        """
        fico = float(row.get("fico", 650) or 650)
        qi = str(row.get("qi", "MISSING") or "MISSING").upper()
        partner = str(row.get("partner", "") or "").upper()
        d60 = float(row.get("d60", 0) or 0)

        # Entity info: prefer row-level, fall back to lookup
        has_prior_bad = int(row.get("has_prior_bad", 0) or 0)
        connected = int(row.get("connected_entities", 0) or 0)
        if not has_prior_bad and not connected:
            app_id = row.get("application_id", "")
            if app_id:
                has_prior_bad_lu, connected_lu = self._get_entity_info(app_id)
                has_prior_bad = has_prior_bad or has_prior_bad_lu
                connected = connected or connected_lu

        rules_triggered = []

        # --- Decline rules ---
        if has_prior_bad:
            rules_triggered.append("prior_bad_auto_decline")
        if qi == "MISSING" and fico < 600:
            rules_triggered.append("missing_qi_low_fico")
        if qi == "MISSING" and partner == "HONEYBOOK":
            rules_triggered.append("missing_qi_honeybook")

        # Check for decline
        decline_rules = {"prior_bad_auto_decline", "missing_qi_low_fico", "missing_qi_honeybook"}
        if decline_rules & set(rules_triggered):
            return FraudResult(decision="decline", rules_triggered=rules_triggered)

        # --- Review rules ---
        if fico < 550:
            rules_triggered.append("extreme_subprime")
        if d60 >= 4 and fico < 620:
            rules_triggered.append("bureau_stress")
        if connected >= 3:
            rules_triggered.append("entity_cluster")

        if rules_triggered:
            return FraudResult(decision="review", rules_triggered=rules_triggered)

        return FraudResult(decision="approve", rules_triggered=[])

    def check_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check all applications in a DataFrame.

        Returns DataFrame with columns: fraud_decision, fraud_rules_triggered
        """
        decisions = []
        rules_list = []
        for _, row in df.iterrows():
            result = self.check(row.to_dict())
            decisions.append(result.decision)
            rules_list.append("|".join(result.rules_triggered) if result.rules_triggered else "")

        out = pd.DataFrame({
            "fraud_decision": decisions,
            "fraud_rules_triggered": rules_list,
        }, index=df.index)
        return out
