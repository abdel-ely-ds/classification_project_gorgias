from __future__ import annotations

import ast

import pandas as pd
from pydantic import BaseModel, Field, field_validator
import api.sample as sample
import gorgias_ml.constants as cst


class TicketMessageRequest(BaseModel):
    account_id: int = Field(description="account id of customer")
    email_sentence_embeddings: str = Field(
        default=sample.EMAIL_SENTENCE_EMBEDDINGS,
        description="string representation of dict of embeddings",
    )

    @field_validator("email_sentence_embeddings")
    def validate_email_sentence_embeddings(cls, v):  # noqa:  B902
        try:
            x = ast.literal_eval(v)
            for val in x.values():
                if val is None:
                    raise ValueError("embeddings are None")

        except Exception as e:
            raise e
        return v

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                cst.ACCOUNT_ID: [self.account_id],
                cst.EMBEDDINGS_COL_NAME: [self.email_sentence_embeddings],
            }
        )
