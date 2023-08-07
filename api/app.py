from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import gorgias_ml.constants as cst
from api.request import TicketMessageRequest
from api.response import TicketMessageResponse
from gorgias_ml.contact_reason import ContactReason

app = FastAPI(
    title="contact reason",
    description="prediction contact reason for ticket messages",
    version="v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

contact_reason = ContactReason()


@app.on_event("startup")
async def startup_event():
    global contact_reason

    contact_reason.load_artifacts(
        from_dir=cst.ARTIFACTS_DIRECTORY,
        from_models_dir=cst.MODELS_DIRECTORY,
        from_pipelines_dir=cst.PIPELINES_DIRECTORY,
    )


@app.get("/")
def root():
    return {"message": "working"}


@app.post("/contact_reason", response_model=TicketMessageResponse, status_code=200)
async def predict(request: TicketMessageRequest) -> TicketMessageResponse:
    preds = contact_reason.predict(request.to_dataframe())
    return TicketMessageResponse.from_dataframe(preds)
