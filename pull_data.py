#!/usr/bin/env python3
"""
Pull master dataset from Databricks and save as local parquet files.

Usage:
    python scripts/pull_data.py              # Pull master features dataset
    python scripts/pull_data.py --all        # Pull all datasets (master + default + entity graph + banking)
    python scripts/pull_data.py --entity     # Pull entity graph dataset only (+ master)
    python scripts/pull_data.py --banking    # Pull banking velocity features (+ master), saves merged file
    python scripts/pull_data.py --query SQL  # Pull a custom query → data/custom.parquet

Requires: databricks CLI configured with default profile (dataproducts workspace)
"""

import io
import json
import os
import subprocess
import sys
import hashlib
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT  # flat layout: data files alongside code
QUERIES_DIR = PROJECT_ROOT / "queries"
WAREHOUSE_ID = "3d2cc6b2a32c1b72"
WAIT_TIMEOUT = "50s"


def run_sql(statement: str, use_external_links: bool = False) -> pd.DataFrame:
    """Execute SQL via Databricks Statement Execution API, return DataFrame.

    For large results (>25MB), set use_external_links=True to download via
    pre-signed URLs instead of inline JSON.
    """
    disposition = "EXTERNAL_LINKS" if use_external_links else "INLINE"
    fmt = "ARROW_STREAM" if use_external_links else "JSON_ARRAY"

    payload = json.dumps({
        "warehouse_id": WAREHOUSE_ID,
        "statement": statement,
        "wait_timeout": WAIT_TIMEOUT,
        "disposition": disposition,
        "format": fmt,
    })

    result = subprocess.run(
        ["databricks", "api", "post", "/api/2.0/sql/statements", "--json", payload],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"CLI stderr: {result.stderr[:200]}", file=sys.stderr)

    response = _parse_cli_json(result.stdout)
    status = response.get("status", {}).get("state", "UNKNOWN")

    if status == "PENDING":
        statement_id = response["statement_id"]
        return _poll_statement(statement_id, use_external_links)
    elif status != "SUCCEEDED":
        error = response.get("status", {}).get("error", {})
        msg = error.get("message", str(response.get("status")))
        # Auto-retry with external links if inline limit exceeded
        if "Inline byte limit exceeded" in msg and not use_external_links:
            print("  Result too large for inline, retrying with external links...")
            return run_sql(statement, use_external_links=True)
        raise RuntimeError(f"Query failed ({status}): {msg}")

    if use_external_links:
        return _parse_external_links(response)
    return _parse_inline_response(response)


def _parse_cli_json(stdout: str) -> dict:
    """Parse JSON from databricks CLI output, skipping version info prefix."""
    stdout = stdout.strip()
    json_start = stdout.find("{")
    if json_start == -1:
        raise RuntimeError(f"No JSON in CLI response: {stdout[:500]}")
    return json.loads(stdout[json_start:])


def _poll_statement(statement_id: str, use_external_links: bool = False,
                    max_polls: int = 60, poll_interval: int = 5) -> pd.DataFrame:
    """Poll a pending statement until it completes."""
    print(f"  Query pending, polling (statement_id={statement_id})...")
    for i in range(max_polls):
        time.sleep(poll_interval)
        result = subprocess.run(
            ["databricks", "api", "get", f"/api/2.0/sql/statements/{statement_id}"],
            capture_output=True, text=True
        )
        response = _parse_cli_json(result.stdout)
        status = response.get("status", {}).get("state", "UNKNOWN")
        if status == "SUCCEEDED":
            print(f"  Query completed after {(i+1)*poll_interval}s")
            if use_external_links:
                return _parse_external_links(response)
            return _parse_inline_response(response)
        elif status in ("FAILED", "CANCELED", "CLOSED"):
            error = response.get("status", {}).get("error", {})
            raise RuntimeError(f"Query {status}: {error.get('message', '')}")
        print(f"  Poll {i+1}: {status}...")
    raise RuntimeError(f"Query timed out after {max_polls * poll_interval}s")


def _parse_inline_response(response: dict) -> pd.DataFrame:
    """Parse inline JSON_ARRAY response into DataFrame."""
    manifest = response["manifest"]
    columns = [col["name"] for col in manifest["schema"]["columns"]]
    total_rows = manifest["total_row_count"]
    total_chunks = manifest["total_chunk_count"]

    all_rows = response["result"]["data_array"]

    if total_chunks > 1:
        statement_id = response["statement_id"]
        for chunk_idx in range(1, total_chunks):
            print(f"  Fetching inline chunk {chunk_idx+1}/{total_chunks}...")
            chunk_result = subprocess.run(
                ["databricks", "api", "get",
                 f"/api/2.0/sql/statements/{statement_id}/result/chunks/{chunk_idx}"],
                capture_output=True, text=True
            )
            chunk_data = _parse_cli_json(chunk_result.stdout)
            all_rows.extend(chunk_data["data_array"])

    df = pd.DataFrame(all_rows, columns=columns)
    print(f"  Retrieved {len(df)} rows × {len(df.columns)} columns (expected {total_rows} rows)")
    return df


def _parse_external_links(response: dict) -> pd.DataFrame:
    """Parse EXTERNAL_LINKS response — download Arrow chunks from pre-signed URLs."""
    import pyarrow.ipc as ipc

    manifest = response["manifest"]
    total_rows = manifest["total_row_count"]
    total_chunks = manifest["total_chunk_count"]
    statement_id = response["statement_id"]

    print(f"  Downloading {total_chunks} chunk(s) via external links ({total_rows} rows)...")
    frames = []

    # First chunk link is in the response
    for chunk in response.get("result", {}).get("external_links", []):
        url = chunk["external_link"]
        print(f"  Downloading chunk {chunk['chunk_index']+1}/{total_chunks}...")
        data = urllib.request.urlopen(url).read()
        reader = ipc.open_stream(io.BytesIO(data))
        frames.append(reader.read_pandas())

    # Fetch remaining chunks if any
    downloaded = len(frames)
    while downloaded < total_chunks:
        result = subprocess.run(
            ["databricks", "api", "get",
             f"/api/2.0/sql/statements/{statement_id}/result/chunks/{downloaded}"],
            capture_output=True, text=True
        )
        chunk_resp = _parse_cli_json(result.stdout)
        for chunk in chunk_resp.get("external_links", []):
            url = chunk["external_link"]
            print(f"  Downloading chunk {chunk['chunk_index']+1}/{total_chunks}...")
            data = urllib.request.urlopen(url).read()
            reader = ipc.open_stream(io.BytesIO(data))
            frames.append(reader.read_pandas())
        downloaded = len(frames)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    print(f"  Retrieved {len(df)} rows × {len(df.columns)} columns (expected {total_rows} rows)")
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to appropriate types where possible."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        # Try numeric conversion
        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue
        try:
            pd.to_numeric(sample)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        except (ValueError, TypeError):
            pass
        # Try date conversion for columns with 'date' in name
        if "date" in col.lower() or "signing" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def pull_master_features() -> pd.DataFrame:
    """Pull the master identity/fraud features dataset."""
    sql_path = QUERIES_DIR / "identity_fraud_features.sql"
    sql = sql_path.read_text()
    print(f"Running master features query ({sql_path.name}, {len(sql)} chars)...")
    df = run_sql(sql)
    df = coerce_types(df)
    return df


def pull_entity_graph() -> pd.DataFrame:
    """Pull entity graph cross-entity features (SSN-linked loan history)."""
    sql = """
WITH owner_ssns AS (
  -- Get owner SSN for each application
  SELECT
    ao.application_id,
    ao.shop_id,
    op.ssn_hash
  FROM gold_prod.analytics.loan_offers_databricks ao
  JOIN silver_prod.shop.owner_profiles op ON ao.shop_id = op.shop_id
  WHERE ao.product_name IN ('flex_loan', 'installment_loan')
),
loan_outcomes AS (
  -- Get loan outcomes for SSN-based lookups
  SELECT
    os.ssn_hash,
    l.id as loan_id,
    l.application_id,
    os.shop_id,
    l.signing_date,
    l.state,
    CASE WHEN l.state IN ('CHARGE_OFF','DEFAULT','WORKOUT','PRE_DEFAULT','TECHNICAL_DEFAULT') THEN 1 ELSE 0 END as is_bad
  FROM owner_ssns os
  JOIN gold_prod.analytics.loan_databricks l ON os.application_id = l.application_id
  WHERE l.loan_type IN ('FIXED_FEE_FLEX','FIXED_FEE_INSTALLMENT')
    AND l.signing_date >= '2024-01-01'
    AND l.state NOT IN ('REJECTED','CANCELED','PENDING_FUNDING','APPROVED')
),
prior_history AS (
  -- For each loan, count prior loans by same SSN (signed BEFORE this loan)
  SELECT
    a.loan_id,
    a.application_id,
    a.shop_id,
    a.signing_date,
    a.state,
    a.is_bad,
    COUNT(DISTINCT b.loan_id) as prior_loans_ssn,
    SUM(CASE WHEN b.is_bad = 1 THEN 1 ELSE 0 END) as prior_bad_ssn,
    SUM(CASE WHEN b.state IN ('PAID_CLOSED','PAID','OVER_PAID') THEN 1 ELSE 0 END) as prior_paid_ssn,
    COUNT(DISTINCT b.shop_id) as prior_shops_ssn
  FROM loan_outcomes a
  LEFT JOIN loan_outcomes b ON a.ssn_hash = b.ssn_hash
    AND b.signing_date < a.signing_date
    AND b.loan_id != a.loan_id
  GROUP BY a.loan_id, a.application_id, a.shop_id, a.signing_date, a.state, a.is_bad
)
SELECT
  loan_id,
  application_id,
  shop_id,
  CAST(signing_date AS STRING) as signing_date,
  state,
  is_bad,
  COALESCE(prior_loans_ssn, 0) as prior_loans_ssn,
  COALESCE(prior_bad_ssn, 0) as prior_bad_ssn,
  COALESCE(prior_paid_ssn, 0) as prior_paid_ssn,
  COALESCE(prior_shops_ssn, 0) as prior_shops_ssn,
  CASE WHEN COALESCE(prior_bad_ssn, 0) > 0 THEN 1 ELSE 0 END as has_prior_bad,
  CASE WHEN COALESCE(prior_paid_ssn, 0) > 0 THEN 1 ELSE 0 END as has_prior_paid,
  CASE WHEN COALESCE(prior_loans_ssn, 0) > 0 THEN 1 ELSE 0 END as is_repeat,
  CASE WHEN COALESCE(prior_shops_ssn, 0) > 1 THEN 1 ELSE 0 END as is_cross_entity
FROM prior_history
"""
    print("Running entity graph cross-entity query...")
    df = run_sql(sql)
    df = coerce_types(df)
    return df


def pull_banking_features() -> pd.DataFrame:
    """Pull banking velocity features from Plaid data (balances + transactions)."""
    sql = """
WITH plaid_accounts AS (
    SELECT DISTINCT
        par.application_revision_id as revision_id,
        par.bam_account_id
    FROM silver_prod.banking_insights.plaid_auth_reports par
    WHERE par.bam_account_id IS NOT NULL
),
balance_agg AS (
    SELECT
        pa.revision_id,
        AVG(CASE WHEN b.date >= DATE_SUB(CURRENT_DATE(), 30) THEN CAST(b.current AS DOUBLE) END) as avg_balance_30d,
        AVG(CASE WHEN b.date BETWEEN DATE_SUB(CURRENT_DATE(), 90) AND DATE_SUB(CURRENT_DATE(), 60) THEN CAST(b.current AS DOUBLE) END) as avg_balance_60_90d,
        MIN(CASE WHEN b.date >= DATE_SUB(CURRENT_DATE(), 30) THEN CAST(b.current AS DOUBLE) END) as min_balance_30d,
        MAX(CASE WHEN b.date >= DATE_SUB(CURRENT_DATE(), 30) THEN CAST(b.current AS DOUBLE) END) as max_balance_30d,
        SUM(CASE WHEN CAST(b.current AS DOUBLE) < 0 AND b.date >= DATE_SUB(CURRENT_DATE(), 90) THEN 1 ELSE 0 END) as negative_balance_days_90d
    FROM plaid_accounts pa
    JOIN silver_prod.banking_insights.balances b ON pa.bam_account_id = b.bam_account_id
    GROUP BY pa.revision_id
),
txn_agg AS (
    SELECT
        pa.revision_id,
        COUNT(*) as txn_count_total,
        COUNT(CASE WHEN t.direction = 'CREDIT' AND t.date >= DATE_SUB(CURRENT_DATE(), 90) THEN 1 END) as credit_txn_count_90d,
        SUM(CASE WHEN t.direction = 'CREDIT' AND t.date >= DATE_SUB(CURRENT_DATE(), 90) THEN CAST(t.amount AS DOUBLE) ELSE 0 END) as credit_sum_90d,
        COUNT(CASE WHEN t.direction = 'DEBIT' AND t.date >= DATE_SUB(CURRENT_DATE(), 90) THEN 1 END) as debit_txn_count_90d,
        SUM(CASE WHEN t.direction = 'DEBIT' AND t.date >= DATE_SUB(CURRENT_DATE(), 90) THEN CAST(t.amount AS DOUBLE) ELSE 0 END) as debit_sum_90d,
        COUNT(CASE WHEN t.category_primary = 'NSF_OVERDRAFT' AND t.date >= DATE_SUB(CURRENT_DATE(), 90) THEN 1 END) as nsf_count_90d
    FROM plaid_accounts pa
    JOIN silver_prod.banking_insights.transactions t ON pa.bam_account_id = t.bam_account_id
    GROUP BY pa.revision_id
)
SELECT
    b.*,
    t.txn_count_total,
    t.credit_txn_count_90d,
    t.credit_sum_90d,
    t.debit_txn_count_90d,
    t.debit_sum_90d,
    t.nsf_count_90d
FROM balance_agg b
LEFT JOIN txn_agg t ON b.revision_id = t.revision_id
"""
    print("Running banking velocity features query...")
    df = run_sql(sql)
    df = coerce_types(df)
    return df


def merge_banking_with_master():
    """Join banking velocity features onto master_features and save combined file."""
    master_path = DATA_DIR / "master_features.parquet"
    banking_path = DATA_DIR / "banking_velocity_features.parquet"

    if not master_path.exists():
        print("  WARNING: master_features.parquet not found, skipping merge.")
        return None
    if not banking_path.exists():
        print("  WARNING: banking_velocity_features.parquet not found, skipping merge.")
        return None

    df_master = pd.read_parquet(master_path)
    df_banking = pd.read_parquet(banking_path)

    if "revision_id" not in df_master.columns:
        print("  WARNING: master_features.parquet has no revision_id column, skipping merge.")
        return None

    # Left join: keep all master rows, attach banking where available
    df_merged = df_master.merge(df_banking, on="revision_id", how="left")

    banking_cols = [c for c in df_banking.columns if c != "revision_id"]
    matched = df_merged[banking_cols[0]].notna().sum() if banking_cols else 0
    print(f"  Merged: {len(df_merged)} rows, {matched}/{len(df_merged)} have banking features "
          f"({matched/len(df_merged)*100:.1f}%)")

    out_path = DATA_DIR / "master_with_banking.parquet"
    df_merged.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB, {len(df_merged)} rows x {len(df_merged.columns)} cols)")
    return df_merged


def pull_default_model() -> pd.DataFrame:
    """Pull the simplified default model dataset."""
    sql = (
        "SELECT l.id as loan_id, l.short_id, CAST(l.signing_date AS STRING) as signing_date, "
        "l.state as loan_state, CAST(l.principal/100 AS DOUBLE) as principal, o.partner, o.shop_id, "
        "DATEDIFF(CURRENT_DATE(), l.signing_date) as age, "
        "CAST(crm.score AS DOUBLE) as fico, "
        "CAST(ccs.delinquencies_thirty_day_count AS DOUBLE) as d30, "
        "CAST(ccs.delinquencies_sixty_day_count AS DOUBLE) as d60, "
        "CAST(ccs.inquiries_last_six_months AS DOUBLE) as inq6, "
        "CAST(ccs.revolving_account_credit_available_percentage AS DOUBLE) as revutil, "
        "CAST(ccs.balance_total_past_due_amounts AS DOUBLE) as pastdue, "
        "CAST(ccs.balance_total_installment_accounts AS DOUBLE) as instbal, "
        "CAST(ccs.balance_total_revolving_accounts AS DOUBLE) as revbal, "
        "CAST(ccs.tradelines_total_items AS DOUBLE) as tl_total, "
        "CAST(ccs.tradelines_total_items_paid AS DOUBLE) as tl_paid, "
        "CAST(ccs.tradelines_total_items_currently_delinquent AS DOUBLE) as tl_delin, "
        "CAST(ccs.payment_amount_monthly_total AS DOUBLE) as mopmt, "
        "DATEDIFF(l.signing_date, ccs.tradelines_oldest_date) as crhist, "
        "COALESCE(d.dwop, 0) as dwop, COALESCE(q.qi, 'MISSING') as qi "
        "FROM gold_prod.analytics.loan_databricks l "
        "LEFT JOIN gold_prod.analytics.loan_offers_databricks o ON l.application_id=o.application_id AND o.product_name IN ('flex_loan','installment_loan') "
        "LEFT JOIN (SELECT target_application_id, id as rid FROM (SELECT id, target_application_id, ROW_NUMBER() OVER (PARTITION BY target_application_id ORDER BY fetched_at DESC) as rn FROM silver_prod.bureau.reports WHERE provider_type='EXPERIAN_CIS_CREDIT_REPORT') s WHERE rn=1) cci ON l.application_id=cci.target_application_id "
        "LEFT JOIN silver_prod.bureau.consumer_credit_risk_models crm ON cci.rid=crm.report_id AND crm.model_type='FICO_RISK_MODEL_V8' "
        "LEFT JOIN silver_prod.bureau.consumer_credit_summary_statistics ccs ON cci.rid=ccs.report_id "
        "LEFT JOIN gold_prod.analytics.days_without_payment_databricks d ON l.short_id=d.short_id "
        "LEFT JOIN (SELECT shop_id, qi, ROW_NUMBER() OVER (PARTITION BY shop_id ORDER BY updated_at DESC) as rn FROM bronze_prod.prod_modern_treasury_analytics.shop_qi_history) q ON o.shop_id=q.shop_id AND q.rn=1 "
        "WHERE l.loan_type IN ('FIXED_FEE_FLEX','FIXED_FEE_INSTALLMENT') AND l.signing_date>='2024-01-01' AND crm.score IS NOT NULL"
    )
    print("Running default model query...")
    df = run_sql(sql)
    df = coerce_types(df)
    return df


def save_with_manifest(df: pd.DataFrame, name: str, query_hash: str):
    """Save DataFrame as parquet and update manifest."""
    parquet_path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)
    file_size = parquet_path.stat().st_size

    # Load or create manifest
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"datasets": {}}

    manifest["datasets"][name] = {
        "pulled_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
        "file_size_bytes": file_size,
        "query_hash": query_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    size_mb = file_size / (1024 * 1024)
    print(f"  Saved: {parquet_path} ({size_mb:.1f} MB, {len(df)} rows × {len(df.columns)} cols)")


def print_summary(df: pd.DataFrame, name: str):
    """Print a quick summary of the pulled dataset."""
    print(f"\n--- {name} Summary ---")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    if "signing_date" in df.columns:
        dates = pd.to_datetime(df["signing_date"], errors="coerce")
        print(f"  Date range: {dates.min()} → {dates.max()}")
    if "loan_state" in df.columns:
        bad_states = {"CHARGE_OFF", "DEFAULT", "WORKOUT", "PRE_DEFAULT", "TECHNICAL_DEFAULT"}
        bad_rate = df["loan_state"].isin(bad_states).mean() * 100
        print(f"  Bad rate: {bad_rate:.1f}%")
    elif "is_bad" in df.columns:
        print(f"  Bad rate: {df['is_bad'].mean() * 100:.1f}%")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        sql = " ".join(sys.argv[idx + 1:])
        print(f"Running custom query...")
        df = run_sql(sql)
        df = coerce_types(df)
        query_hash = hashlib.md5(sql.encode()).hexdigest()[:8]
        save_with_manifest(df, "custom", query_hash)
        print_summary(df, "custom")
        return

    # Pull master features
    print("=" * 60)
    print("Pulling master features dataset")
    print("=" * 60)
    sql_text = (QUERIES_DIR / "identity_fraud_features.sql").read_text()
    query_hash = hashlib.md5(sql_text.encode()).hexdigest()[:8]
    df_master = pull_master_features()
    save_with_manifest(df_master, "master_features", query_hash)
    print_summary(df_master, "master_features")

    if "--all" in sys.argv:
        print("\n" + "=" * 60)
        print("Pulling default model dataset")
        print("=" * 60)
        df_default = pull_default_model()
        save_with_manifest(df_default, "default_model", "default_model_v1")
        print_summary(df_default, "default_model")

    if "--all" in sys.argv or "--entity" in sys.argv:
        print("\n" + "=" * 60)
        print("Pulling entity graph dataset")
        print("=" * 60)
        df_entity = pull_entity_graph()
        save_with_manifest(df_entity, "entity_graph_cross", "entity_graph_v1")
        print_summary(df_entity, "entity_graph_cross")

    if "--all" in sys.argv or "--banking" in sys.argv:
        print("\n" + "=" * 60)
        print("Pulling banking velocity features")
        print("=" * 60)
        df_banking = pull_banking_features()
        save_with_manifest(df_banking, "banking_velocity_features", "banking_velocity_v1")
        print_summary(df_banking, "banking_velocity_features")

        # Also create merged master+banking file
        print("\n  Merging banking features with master dataset...")
        merge_banking_with_master()

    print("\n✓ Done. Data cached in data/")


if __name__ == "__main__":
    main()
