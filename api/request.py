from pydantic import BaseModel, Field, field_validator
import ast


class TicketMessageRequest(BaseModel):
    account_id: int = Field(description="account id of customer")
    email_sentence_embeddings: str = Field(description="string representation of dict of embeddings")

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
