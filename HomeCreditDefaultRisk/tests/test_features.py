"""Tests for historical table feature aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import aggregate_installments


def test_installment_aggregation_calculates_lateness_and_shortfall(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_PREV": [10, 10, 20],
            "NUM_INSTALMENT_VERSION": [1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2, 1],
            "DAYS_INSTALMENT": [-20, -10, -15],
            "DAYS_ENTRY_PAYMENT": [-18, -12, -15],
            "AMT_INSTALMENT": [100.0, 100.0, 200.0],
            "AMT_PAYMENT": [80.0, 100.0, 200.0],
        }
    )
    frame.to_csv(tmp_path / "installments_payments.csv", index=False)

    result = aggregate_installments(tmp_path).set_index("SK_ID_CURR")

    assert result.loc[1, "INSTAL_INSTALMENT_DAYS_LATE_MAX"] == 2
    assert result.loc[1, "INSTAL_INSTALMENT_PAYMENT_SHORTFALL_SUM"] == 20
    assert np.isclose(result.loc[1, "INSTAL_INSTALMENT_IS_LATE_MEAN"], 0.5)
    assert result.loc[2, "INSTAL_RECORD_COUNT"] == 1
