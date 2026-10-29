from functools import lru_cache
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent


def _model_default(key: str, local_name: str, hub_id: str) -> str:
    value = os.getenv(key)
    if value:
        return value
    local = PROJECT / "models" / local_name
    return str(local) if local.is_dir() else hub_id


class Settings:
    app_name = "Financial NLP Lab API"
    indobert_model_id = _model_default("INDOBERT_MODEL_ID", "indobert_finetuned", "Tipuls/indobert-financial-sentiment")
    indolem_model_id = _model_default("INDOLEM_MODEL_ID", "indolem_finetuned", "Tipuls/indolem-financial-sentiment")
    finbert_model_id = _model_default("FINBERT_MODEL_ID", "finbert_finetuned", "Tipuls/finbert-financial-sentiment")
    hf_token = os.getenv("HF_TOKEN") or None
    hf_home = os.getenv("HF_HOME", str(ROOT / "hf_cache"))
    max_cached_models = max(1, int(os.getenv("MAX_CACHED_MODELS", "1")))
    max_text_chars = max(100, int(os.getenv("MAX_TEXT_CHARS", "8000")))
    cors_origins = [x.strip().rstrip("/") for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()]
    translation_provider = os.getenv("TRANSLATION_PROVIDER", "google").lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
