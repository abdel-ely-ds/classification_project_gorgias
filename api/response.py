from pydantic import BaseModel, Field


class TicketMessageResponse(BaseModel):
    account_id: int = Field(description="account id of customer")
    prediction: str = Field(description="predicted class")
