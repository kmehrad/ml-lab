"""Applicant-level feature aggregation for Home Credit relational tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "SK_ID_CURR"


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame.columns = [
        f"{prefix}_{column}_{stat}".upper()
        for column, stat in frame.columns.to_flat_index()
    ]
    return frame


def _aggregate_numeric(
    frame: pd.DataFrame,
    columns: list[str],
    prefix: str,
    statistics: tuple[str, ...] = ("mean", "min", "max", "sum"),
) -> pd.DataFrame:
    available = [column for column in columns if column in frame]
    aggregated = frame.groupby(ID_COLUMN)[available].agg(statistics)
    return _flatten_columns(aggregated, prefix)


def _join_count(aggregated: pd.DataFrame, frame: pd.DataFrame, prefix: str) -> None:
    aggregated[f"{prefix}_RECORD_COUNT"] = frame.groupby(ID_COLUMN).size()


def aggregate_bureau(data_dir: Path) -> pd.DataFrame:
    bureau = pd.read_csv(data_dir / "bureau.csv")
    numeric = [
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    ]
    aggregated = _aggregate_numeric(bureau, numeric, "BUREAU")
    _join_count(aggregated, bureau, "BUREAU")

    bureau["BUREAU_IS_ACTIVE"] = bureau["CREDIT_ACTIVE"].eq("Active").astype("int8")
    bureau["BUREAU_IS_CLOSED"] = bureau["CREDIT_ACTIVE"].eq("Closed").astype("int8")
    bureau["BUREAU_HAS_OVERDUE"] = bureau["CREDIT_DAY_OVERDUE"].gt(0).astype("int8")
    bureau["BUREAU_DEBT_CREDIT_RATIO"] = (
        bureau["AMT_CREDIT_SUM_DEBT"]
        / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    )
    indicators = bureau.groupby(ID_COLUMN)[
        [
            "BUREAU_IS_ACTIVE",
            "BUREAU_IS_CLOSED",
            "BUREAU_HAS_OVERDUE",
            "BUREAU_DEBT_CREDIT_RATIO",
        ]
    ].agg(["mean", "max"])
    aggregated = aggregated.join(_flatten_columns(indicators, "BUREAU"))

    balance = pd.read_csv(data_dir / "bureau_balance.csv")
    balance["BB_IS_DELINQUENT"] = balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
    balance["BB_IS_SEVERE"] = balance["STATUS"].isin(["3", "4", "5"]).astype("int8")
    bb_by_loan = balance.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "size"),
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_DELINQUENT_RATE=("BB_IS_DELINQUENT", "mean"),
        BB_SEVERE_RATE=("BB_IS_SEVERE", "mean"),
    )
    bb_by_loan = bb_by_loan.join(
        bureau.set_index("SK_ID_BUREAU")[[ID_COLUMN]],
        how="inner",
    )
    bb_by_client = bb_by_loan.groupby(ID_COLUMN).agg(
        {
            "BB_MONTHS_COUNT": ["mean", "max", "sum"],
            "BB_MONTHS_MIN": ["min"],
            "BB_DELINQUENT_RATE": ["mean", "max"],
            "BB_SEVERE_RATE": ["mean", "max"],
        }
    )
    aggregated = aggregated.join(_flatten_columns(bb_by_client, "BUREAU"))
    return aggregated.reset_index()


def aggregate_previous_applications(data_dir: Path) -> pd.DataFrame:
    previous = pd.read_csv(data_dir / "previous_application.csv")
    day_columns = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for column in day_columns:
        previous[column] = previous[column].replace(365243, np.nan)

    previous["PREV_APPROVED"] = previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("int8")
    previous["PREV_REFUSED"] = previous["NAME_CONTRACT_STATUS"].eq("Refused").astype("int8")
    previous["PREV_CREDIT_APPLICATION_RATIO"] = (
        previous["AMT_CREDIT"] / previous["AMT_APPLICATION"].replace(0, np.nan)
    )
    previous["PREV_DOWN_PAYMENT_RATIO"] = (
        previous["AMT_DOWN_PAYMENT"] / previous["AMT_CREDIT"].replace(0, np.nan)
    )

    numeric = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        *day_columns,
        "PREV_CREDIT_APPLICATION_RATIO",
        "PREV_DOWN_PAYMENT_RATIO",
    ]
    aggregated = _aggregate_numeric(previous, numeric, "PREV")
    _join_count(aggregated, previous, "PREV")
    status = previous.groupby(ID_COLUMN)[["PREV_APPROVED", "PREV_REFUSED"]].agg(
        ["mean", "sum"]
    )
    return aggregated.join(_flatten_columns(status, "PREV")).reset_index()


def aggregate_pos_cash(data_dir: Path) -> pd.DataFrame:
    pos = pd.read_csv(data_dir / "POS_CASH_balance.csv")
    pos["POS_IS_DELINQUENT"] = pos["SK_DPD"].gt(0).astype("int8")
    pos["POS_IS_ACTIVE"] = pos["NAME_CONTRACT_STATUS"].eq("Active").astype("int8")
    numeric = [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
        "POS_IS_DELINQUENT",
        "POS_IS_ACTIVE",
    ]
    aggregated = _aggregate_numeric(pos, numeric, "POS", ("mean", "min", "max"))
    _join_count(aggregated, pos, "POS")
    return aggregated.reset_index()


def aggregate_credit_card(data_dir: Path) -> pd.DataFrame:
    card = pd.read_csv(data_dir / "credit_card_balance.csv")
    card["CC_BALANCE_LIMIT_RATIO"] = (
        card["AMT_BALANCE"] / card["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    )
    card["CC_PAYMENT_MIN_RATIO"] = (
        card["AMT_PAYMENT_CURRENT"]
        / card["AMT_INST_MIN_REGULARITY"].replace(0, np.nan)
    )
    card["CC_IS_DELINQUENT"] = card["SK_DPD"].gt(0).astype("int8")
    numeric = [
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM",
        "SK_DPD",
        "SK_DPD_DEF",
        "CC_BALANCE_LIMIT_RATIO",
        "CC_PAYMENT_MIN_RATIO",
        "CC_IS_DELINQUENT",
    ]
    aggregated = _aggregate_numeric(card, numeric, "CC", ("mean", "min", "max", "sum"))
    _join_count(aggregated, card, "CC")
    return aggregated.reset_index()


def aggregate_installments(data_dir: Path) -> pd.DataFrame:
    installments = pd.read_csv(data_dir / "installments_payments.csv")
    installments["INSTALMENT_DAYS_LATE"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    ).clip(lower=0)
    installments["INSTALMENT_PAYMENT_SHORTFALL"] = (
        installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    ).clip(lower=0)
    installments["INSTALMENT_PAYMENT_RATIO"] = (
        installments["AMT_PAYMENT"]
        / installments["AMT_INSTALMENT"].replace(0, np.nan)
    )
    installments["INSTALMENT_IS_LATE"] = (
        installments["INSTALMENT_DAYS_LATE"].gt(0).astype("int8")
    )
    installments["INSTALMENT_IS_SHORT"] = (
        installments["INSTALMENT_PAYMENT_SHORTFALL"].gt(0).astype("int8")
    )
    numeric = [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "INSTALMENT_DAYS_LATE",
        "INSTALMENT_PAYMENT_SHORTFALL",
        "INSTALMENT_PAYMENT_RATIO",
        "INSTALMENT_IS_LATE",
        "INSTALMENT_IS_SHORT",
    ]
    aggregated = _aggregate_numeric(
        installments,
        numeric,
        "INSTAL",
        ("mean", "min", "max", "sum"),
    )
    _join_count(aggregated, installments, "INSTAL")
    return aggregated.reset_index()


def build_historical_features(data_dir: str | Path) -> pd.DataFrame:
    """Build one row per applicant from all six historical data sources."""
    data_path = Path(data_dir)
    feature_sets = [
        aggregate_bureau(data_path),
        aggregate_previous_applications(data_path),
        aggregate_pos_cash(data_path),
        aggregate_credit_card(data_path),
        aggregate_installments(data_path),
    ]
    combined = feature_sets[0]
    for features in feature_sets[1:]:
        combined = combined.merge(features, on=ID_COLUMN, how="outer")
    return combined


def load_or_build_historical_features(
    data_dir: str | Path,
    cache_path: str | Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Load cached historical features or build and persist them as a pickle."""
    cache = Path(cache_path)
    if cache.exists() and not force:
        return pd.read_pickle(cache)

    features = build_historical_features(data_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(cache)
    return features

