import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.schemas import CompareRequest, CompareResponse, ModelId, ModelStatus, PredictRequest, PredictResponse
from app.services.model_registry import MODEL_NAMES, registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", description="Lazy-loaded inference for fine-tuned Indonesian financial sentiment models.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/health")
def health():
    configured = sum(registry.status(model_id)["configured"] for model_id in MODEL_NAMES)
    return {"status": "ok", "ready": configured > 0, "device": str(registry.device), "models_configured": configured}


@app.get("/models")
def models() -> dict[str, list[ModelStatus]]:
    return {"models": [ModelStatus(**registry.status(model_id)) for model_id in MODEL_NAMES]}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.text) > settings.max_text_chars:
        raise HTTPException(status_code=422, detail=f"Text exceeds configured limit of {settings.max_text_chars} characters")
    try:
        result, _ = registry.predict(request.model_id, request.text)
        return {"prediction": result}
    except Exception as exc:
        logging.getLogger(__name__).exception("Prediction failed for %s", request.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    if len(request.text) > settings.max_text_chars:
        raise HTTPException(status_code=422, detail=f"Text exceeds configured limit of {settings.max_text_chars} characters")
    results, errors = [], []
    # Sequential inference intentionally bounds peak memory on CPU hosting tiers.
    for model_id in request.model_ids:
        try:
            result, _ = registry.predict(model_id, request.text)
            results.append(result)
        except Exception as exc:
            logging.getLogger(__name__).exception("Comparison model %s failed", model_id)
            errors.append({"model_id": model_id, "model_name": MODEL_NAMES[model_id], "message": str(exc)})
    sentiment_set = {item["sentiment"] for item in results}
    agreement = None
    if len(results) > 1:
        first = results[0]["sentiment"]
        agreeing = [item["model_id"] for item in results if item["sentiment"] == first] if len(sentiment_set) > 1 else [item["model_id"] for item in results]
        disagreeing = [item["model_id"] for item in results if item["sentiment"] != first] if len(sentiment_set) > 1 else []
        agreement = {"unanimous": len(sentiment_set) == 1, "agreeing_models": agreeing, "disagreeing_models": disagreeing}
    if not results:
        # Partial failure remains a valid response for inspection; client displays each model error.
        return {"results": [], "errors": errors, "agreement": None}
    return {"results": results, "errors": errors, "agreement": agreement}
