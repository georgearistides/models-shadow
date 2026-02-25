#!/usr/bin/env python3
"""
build_graph_lookup.py — Build entity graph lookup table for FraudGate.

Creates a JSON lookup file mapping shop_id -> {has_prior_bad, connected_count}
for use by FraudGate Category 7 (Entity Graph) rules.

Two modes:
1. LOCAL (default): Uses master_features.parquet to identify same-shop repeat
   loans. Limited signal because same-shop repeat borrowing requires prior loan
   payoff. Only ~6 loans have prior bad from same shop.

2. DATABRICKS (--databricks): Pulls owner_profiles from Databricks to identify
   cross-entity connections (different shops sharing owner email/phone/tax_id).
   This is the production approach with 15.4% of shops sharing identifiers and
   22.6% bad rate for connected-to-bad entities. NOT IMPLEMENTED YET — requires
   silver_prod.merchant_profile.owner_profiles access.

Temporal safety: Only considers outcomes from loans that originated AND resolved
(reached a terminal state) BEFORE the current loan's signing_date. This prevents
future information leakage.

Usage:
    python3 scripts/models/build_graph_lookup.py              # local mode
    python3 scripts/models/build_graph_lookup.py --output data/graph_lookup.json
    python3 scripts/models/build_graph_lookup.py --databricks  # future: Databricks mode

Output:
    JSON file: {shop_id: {"has_prior_bad": bool, "connected_count": int}, ...}
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "master_features.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "graph_lookup.json"

BAD_STATES = {"CHARGE_OFF", "DEFAULT", "WORKOUT", "PRE_DEFAULT", "TECHNICAL_DEFAULT"}
GOOD_STATES = {"PAID_CLOSED", "PAID", "OVER_PAID"}
TERMINAL_STATES = BAD_STATES | GOOD_STATES
EXCLUDE_STATES = {"REJECTED", "CANCELED", "PENDING_FUNDING", "APPROVED"}


def build_local_lookup(data_path: Path) -> dict:
    """Build graph lookup from local master_features.parquet.

    Uses same-shop repeat loans to identify prior bad outcomes.
    Limited because same-shop repeats mostly require prior loan payoff.

    Returns
    -------
    dict : {shop_id: {"has_prior_bad": bool, "connected_count": int}}
    """
    df = pd.read_parquet(data_path)
    df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
    df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)
    df["is_terminal"] = df["loan_state"].isin(TERMINAL_STATES).astype(int)
    df["signing_date"] = pd.to_datetime(df["signing_date"])

    # Sort by shop and date for temporal processing
    df = df.sort_values(["shop_id", "signing_date"]).reset_index(drop=True)

    lookup = {}

    for shop_id, group in df.groupby("shop_id"):
        if len(group) < 2:
            # Single-loan shop: no connected entities from same-shop perspective
            continue

        group = group.sort_values("signing_date")
        dates = group["signing_date"].values
        bads = group["is_bad"].values
        terminals = group["is_terminal"].values

        for idx, row in group.iterrows():
            current_date = row["signing_date"]

            # Prior loans: originated strictly before this loan
            prior_mask = dates < np.datetime64(current_date)
            # AND in a terminal state (resolved) -- prevents using NORMAL loans
            # whose outcome is still unknown
            prior_terminal = prior_mask & (terminals == 1)
            prior_bad = bads[prior_terminal]

            has_prior_bad = bool(prior_bad.sum() > 0) if len(prior_bad) > 0 else False
            connected_count = int(prior_terminal.sum())

            if has_prior_bad or connected_count > 0:
                lookup[row["shop_id"]] = {
                    "has_prior_bad": has_prior_bad,
                    "connected_count": connected_count,
                }

    return lookup


def build_databricks_lookup() -> dict:
    """Build graph lookup from Databricks owner_profiles.

    NOT YET IMPLEMENTED. Production approach would:
    1. Query silver_prod.merchant_profile.owner_profiles for email, phone, tax_id
    2. Build union-find connected components across shops sharing any identifier
    3. For each connected component, check if any shop has a prior bad loan
    4. Return lookup with cross-entity connections

    The graph analysis (graph-features.md) found:
    - 15.4% of shops share key identifiers (email, phone, tax_id)
    - 22.6% bad rate for loans connected to a bad neighbor (vs 6.5%)
    - AUC 0.66 for neighbor_has_bad feature
    - Strongest link: Owner Tax ID (SSN) — 11.4% both-bad rate

    Join path:
      master_features (shop_id)
      -> shop_profiles (id = shop_id)
        -> merchant_profiles (id = shop_profiles.merchant_profile_id)
        -> owner_profiles (merchant_profile_id, is_applicant=true)

    Then for each identifier type (email, phone, tax_id):
      group shops by shared identifier value -> connected components
    """
    raise NotImplementedError(
        "Databricks mode not yet implemented. Use --local mode or provide "
        "a pre-built lookup file. See docstring for the required join path "
        "and data sources."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build entity graph lookup for FraudGate"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=DATA_PATH,
        help=f"Input parquet path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--databricks",
        action="store_true",
        help="Use Databricks mode (NOT IMPLEMENTED — requires owner_profiles access)",
    )
    args = parser.parse_args()

    if args.databricks:
        lookup = build_databricks_lookup()
    else:
        if not args.data.exists():
            print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
            print("Run /pull-data first to download master_features.parquet", file=sys.stderr)
            sys.exit(1)

        print(f"Building local graph lookup from {args.data}...")
        lookup = build_local_lookup(args.data)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(lookup, f, indent=2)

    # Stats
    n_entries = len(lookup)
    n_prior_bad = sum(1 for v in lookup.values() if v["has_prior_bad"])
    n_connected = sum(1 for v in lookup.values() if v["connected_count"] > 0)
    max_connected = max((v["connected_count"] for v in lookup.values()), default=0)

    print(f"Wrote {n_entries} entries to {args.output}")
    print(f"  Entries with has_prior_bad=True: {n_prior_bad}")
    print(f"  Entries with connected_count>0:  {n_connected}")
    print(f"  Max connected_count:             {max_connected}")
    print()
    print("NOTE: Local mode uses same-shop repeat loans only.")
    print("For cross-entity connections (15.4% of shops, AUC 0.66),")
    print("use --databricks mode once owner_profiles access is available.")


if __name__ == "__main__":
    main()
