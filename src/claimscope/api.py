"""Local inference demo. Load only trusted, locally generated joblib artifacts."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from claimscope.config import CATEGORICAL, FEATURES, NUMERIC

LOG = logging.getLogger(__name__)


class ClaimFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    VehPower: Annotated[float, Field(ge=0, le=100)]
    VehAge: Annotated[float, Field(ge=0, le=150)]
    DrivAge: Annotated[float, Field(ge=0, le=150)]
    Density: Annotated[float, Field(ge=0, le=1_000_000)]
    Area: Annotated[str, Field(min_length=1, max_length=32)]
    VehBrand: Annotated[str, Field(min_length=1, max_length=32)]
    VehGas: Annotated[str, Field(min_length=1, max_length=32)]
    Region: Annotated[str, Field(min_length=1, max_length=32)]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: Annotated[list[ClaimFeatures], Field(min_length=1, max_length=100)]


class Prediction(BaseModel):
    expected_recorded_amount: float
    high_cost_probability: float
    flagged_at_validation_cutoff: bool
    warnings: list[str]


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    regression_model: str
    classification_model: str
    high_cost_amount_threshold: float
    probability_cutoff: float
    amount_unit: str = "source monetary units; currency not independently verified"
    interpretation: str = (
        "Retrospective public-data demo; not verified final-cost or first-report prediction."
    )


def create_app(model_path: Path | None = None):
    path = model_path or Path(os.environ.get("CLAIMSCOPE_MODEL", "models/claimscope.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        try:
            bundle = joblib.load(path)
            if bundle["metadata"]["sklearn_version"] != sklearn.__version__:
                raise ValueError(
                    "Model/runtime scikit-learn versions differ; retrain with this environment"
                )
            if bundle["metadata"]["features"] != FEATURES:
                raise ValueError("Artifact feature contract mismatch")
            app.state.bundle = bundle
            LOG.info("ClaimScope model loaded")
        except Exception:
            LOG.exception("Model unavailable; train locally before inference")
        yield
        app.state.bundle = None

    app = FastAPI(
        title="ClaimScope",
        version="0.1.0",
        description="Independent public-data claim severity research demo.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        # Do not echo potentially sensitive values or non-JSON infinities in error responses.
        details = [{k: e[k] for k in ("type", "loc", "msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    def ready():
        bundle = app.state.bundle
        if bundle is None:
            raise HTTPException(
                503, "Model unavailable. Run the training pipeline and restart the API."
            )
        return bundle

    @app.get("/health")
    def health():
        ready()
        return {"status": "ready", "purpose": "portfolio demonstration"}

    @app.get("/model")
    def model_info():
        return ready()["metadata"]

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: BatchRequest):
        bundle = ready()
        metadata = bundle["metadata"]
        frame = pd.DataFrame([record.model_dump() for record in request.claims])[FEATURES]
        try:
            amounts = bundle["regressor"].predict(frame)
            probabilities = bundle["classifier"].predict_proba(frame)[:, 1]
            if (
                not np.isfinite(amounts).all()
                or (amounts <= 0).any()
                or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError("Non-finite or invalid model output")
        except Exception:
            LOG.exception("Inference failed for batch_size=%d", len(frame))
            raise HTTPException(500, "Prediction failed; inspect server logs.") from None
        predictions = []
        for i, row in frame.iterrows():
            warnings = []
            for column in CATEGORICAL:
                if row[column] not in metadata["categories"][column]:
                    warnings.append(f"Unknown {column}: encoded as all-zero category indicators.")
            for column in NUMERIC:
                low, high = metadata["numeric_training_ranges"][column]
                if not low <= row[column] <= high:
                    warnings.append(f"{column} is outside the training range.")
            predictions.append(
                Prediction(
                    expected_recorded_amount=float(amounts[i]),
                    high_cost_probability=float(probabilities[i]),
                    flagged_at_validation_cutoff=bool(
                        probabilities[i] >= metadata["probability_cutoff"]
                    ),
                    warnings=warnings,
                )
            )
        LOG.info("Scored batch_size=%d", len(frame))
        return PredictionResponse(
            predictions=predictions,
            regression_model=metadata["regression"],
            classification_model=metadata["classification"],
            high_cost_amount_threshold=metadata["high_cost_amount_threshold"],
            probability_cutoff=metadata["probability_cutoff"],
        )

    return app


app = create_app()
