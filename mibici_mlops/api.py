"""Servicio de prediccion sobre las variables preparadas del pipeline."""

from contextlib import asynccontextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from mibici_mlops.config import MODELS_DIR
from mibici_mlops.modeling.predict import predict_frame

NonNegative = Annotated[FiniteFloat, Field(ge=0)]


class PredictionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estacion_id: Annotated[int, Field(gt=0)]
    fecha_hora: datetime
    hora: Annotated[int, Field(ge=0, le=23)]
    hora_sin: Annotated[FiniteFloat, Field(ge=-1, le=1)]
    hora_cos: Annotated[FiniteFloat, Field(ge=-1, le=1)]
    salidas_lag_24: NonNegative
    salidas_lag_168: NonNegative
    salidas_media_7d: NonNegative
    llegadas_lag_24: NonNegative
    llegadas_lag_168: NonNegative
    llegadas_media_7d: NonNegative
    latitud: Annotated[FiniteFloat, Field(ge=-90, le=90)]
    longitud: Annotated[FiniteFloat, Field(ge=-180, le=180)]
    capacidad: NonNegative
    perfil_pc1: FiniteFloat
    perfil_pc2: FiniteFloat
    es_fin_semana: Literal[0, 1]
    es_festivo: Literal[0, 1]
    dia_semana: Annotated[int, Field(ge=0, le=6)]
    zona: Annotated[str, Field(min_length=1)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: Annotated[list[PredictionRow], Field(min_length=1, max_length=1000)]


class Prediction(BaseModel):
    estacion_id: int
    fecha_hora: datetime
    pred_salidas: NonNegative
    pred_llegadas: NonNegative


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("MIBICI_MODEL_PATH", MODELS_DIR / "serving"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not (path / "MLmodel").is_file():
            raise RuntimeError(f"Falta el artefacto completo de MLflow en {path}")
        application.state.model = mlflow.sklearn.load_model(str(path))
        yield

    application = FastAPI(
        title="MiBici Demand API",
        version="1.0.0",
        description="Predice salidas y llegadas con variables ya preparadas por estacion y hora.",
        lifespan=lifespan,
    )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        frame = pd.DataFrame([record.model_dump() for record in payload.records])
        predictions = predict_frame(application.state.model, frame)
        values = predictions[["pred_salidas", "pred_llegadas"]].to_numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("El modelo produjo predicciones no finitas")
        return PredictionResponse(predictions=predictions.to_dict(orient="records"))

    return application


app = create_app()
