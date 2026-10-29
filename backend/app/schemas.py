from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelId = Literal["indobert", "indolem", "finbert"]
Sentiment = Literal["Negatif", "Netral", "Positif"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=8, max_length=8000)
    model_id: ModelId

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank")
        return value


class CompareRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=8, max_length=8000)
    model_ids: list[ModelId] = Field(default=["indobert", "indolem", "finbert"], min_length=1, max_length=3)

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank")
        return value

    @field_validator("model_ids")
    @classmethod
    def unique_models(cls, value: list[ModelId]) -> list[ModelId]:
        if len(set(value)) != len(value):
            raise ValueError("model_ids must not contain duplicates")
        return value


class Probabilities(BaseModel):
    Negatif: float = Field(ge=0, le=1)
    Netral: float = Field(ge=0, le=1)
    Positif: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    model_id: ModelId
    model_name: str
    sentiment: Sentiment
    probabilities: Probabilities
    inference_ms: float = Field(ge=0)
    preprocessing: dict[str, str | int | bool]
    translation: dict[str, str] | None = None


class PredictResponse(BaseModel):
    prediction: Prediction


class ModelStatus(BaseModel):
    id: ModelId
    name: str
    configured: bool
    source: str | None
    downloaded: bool
    loaded: bool
    ready: bool
    state: Literal["unconfigured", "configured", "downloaded", "loaded", "error"]
    detail: str | None = None


class CompareError(BaseModel):
    model_id: ModelId
    model_name: str
    message: str


class Agreement(BaseModel):
    unanimous: bool
    agreeing_models: list[ModelId]
    disagreeing_models: list[ModelId]


class CompareResponse(BaseModel):
    results: list[Prediction]
    errors: list[CompareError]
    agreement: Agreement | None
