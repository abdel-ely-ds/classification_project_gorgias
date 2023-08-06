from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

import gorgias_ml.constants as cst


class TicketMessageResponse(BaseModel):
    account_id: int = Field(description="account id of customer")
    prediction: str = Field(description="predicted class")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> TicketMessageResponse:
        assert len(df) == 1
        "length of dataframe should be one"

        return TicketMessageResponse(
            account_id=df.iloc[0][cst.ACCOUNT_ID], prediction=df.iloc[0][cst.PREDICTION]
        )
