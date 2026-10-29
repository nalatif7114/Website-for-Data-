from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import gc
import json
import logging
import threading
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config import get_settings

logger = logging.getLogger(__name__)
LABELS = ("Negatif", "Netral", "Positif")
MODEL_NAMES = {"indobert": "IndoBERT", "indolem": "IndoLEM", "finbert": "FinBERT"}
MODEL_ENV = {"indobert": "indobert_model_id", "indolem": "indolem_model_id", "finbert": "finbert_model_id"}


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    id_to_sentiment: dict[int, str]
    source: str


class ModelRegistry:
    """One-model-at-a-time LRU; protects CPU/RAM constrained hosting."""
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self.states: dict[str, str] = {}
        self.details: dict[str, str] = {}
        self.lock = threading.RLock()

    def source(self, model_id: str) -> str:
        return str(getattr(self.settings, MODEL_ENV[model_id])).strip()

    def _local(self, source: str) -> bool:
        return Path(source).expanduser().is_dir()

    def _map_labels(self, config) -> dict[int, str]:
        raw = getattr(config, "id2label", {}) or {}
        mapped: dict[int, str] = {}
        aliases = {"negatif": "Negatif", "negative": "Negatif", "netral": "Netral", "neutral": "Netral", "positif": "Positif", "positive": "Positif"}
        for raw_index, raw_label in raw.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            label = aliases.get(str(raw_label).strip().lower())
            if label:
                mapped[index] = label
        if set(mapped.values()) != set(LABELS) or len(mapped) != 3:
            raise ValueError(f"Checkpoint label map must explicitly define Negatif, Netral, Positif; found {raw!r}")
        return mapped

    def get(self, model_id: str) -> LoadedModel:
        with self.lock:
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                return self.cache[model_id]
            source = self.source(model_id)
            if not source:
                raise RuntimeError(f"{MODEL_NAMES[model_id]} has no configured checkpoint repository or local path")
            self.states[model_id] = "configured"
            self.details.pop(model_id, None)
            while len(self.cache) >= self.settings.max_cached_models:
                old_id, old = self.cache.popitem(last=False)
                logger.info("Evicting cached %s checkpoint", old_id)
                del old
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            token = None if self._local(source) else self.settings.hf_token
            try:
                logger.info("Loading %s from %s", model_id, source)
                tokenizer = AutoTokenizer.from_pretrained(source, token=token, cache_dir=self.settings.hf_home, local_files_only=self._local(source))
                model = AutoModelForSequenceClassification.from_pretrained(source, token=token, cache_dir=self.settings.hf_home, local_files_only=self._local(source))
                id_to_sentiment = self._map_labels(model.config)
                model.to(self.device)
                model.eval()
                loaded = LoadedModel(tokenizer, model, id_to_sentiment, source)
                self.cache[model_id] = loaded
                self.states[model_id] = "loaded"
                return loaded
            except Exception as exc:
                self.states[model_id] = "error"
                self.details[model_id] = f"Checkpoint could not be loaded or its label map was invalid: {exc}"
                logger.exception("Failed loading %s", model_id)
                raise RuntimeError(self.details[model_id]) from exc

    def status(self, model_id: str) -> dict:
        source = self.source(model_id)
        configured = bool(source)
        is_local = self._local(source) if configured else False
        downloaded = False
        if is_local:
            local_dir = Path(source).expanduser()
            has_weights = any((local_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
            downloaded = (local_dir / "config.json").is_file() and has_weights and (local_dir / "tokenizer_config.json").is_file()
        elif configured:
            try:
                config_file = try_to_load_from_cache(source, "config.json", cache_dir=self.settings.hf_home)
                tokenizer_file = try_to_load_from_cache(source, "tokenizer_config.json", cache_dir=self.settings.hf_home)
                weights_file = any(try_to_load_from_cache(source, name, cache_dir=self.settings.hf_home) for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
                downloaded = bool(config_file and tokenizer_file and weights_file)
            except Exception:
                downloaded = False
        loaded = model_id in self.cache
        state = self.states.get(model_id, "loaded" if loaded else ("downloaded" if downloaded else ("configured" if configured else "unconfigured")))
        if state not in {"error", "loaded"}:
            state = "loaded" if loaded else ("downloaded" if downloaded else ("configured" if configured else "unconfigured"))
        return {"id": model_id, "name": MODEL_NAMES[model_id], "configured": configured, "source": f"local:{Path(source).name}" if is_local else source or None, "downloaded": downloaded or loaded, "loaded": loaded, "ready": loaded, "state": state, "detail": self.details.get(model_id)}

    def loaded_count(self) -> int:
        return len(self.cache)

    def predict(self, model_id: str, text: str) -> tuple[dict, dict | None]:
        with self.lock:
            return self._predict_locked(model_id, text)

    def _predict_locked(self, model_id: str, text: str) -> tuple[dict, dict | None]:
        from time import perf_counter
        from app.services.translation import translate_indonesian

        model_data = self.get(model_id)
        model_input = text
        translation = None
        input_language = "id"
        if model_id == "finbert":
            model_input, provider = translate_indonesian(text, self.settings.translation_provider)
            input_language = "en (translated)"
            translation = {"status": "translated", "translated_text": model_input, "provider": provider}
        encoded = model_data.tokenizer(model_input, truncation=True, max_length=256, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        start = perf_counter()
        with torch.inference_mode():
            logits = model_data.model(**encoded).logits[0]
            raw_probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        duration = (perf_counter() - start) * 1000
        probabilities = {label: 0.0 for label in LABELS}
        for index, label in model_data.id_to_sentiment.items():
            probabilities[label] = float(raw_probs[index])
        sentiment = max(probabilities, key=probabilities.get)
        name = MODEL_NAMES[model_id]
        prediction = {
            "model_id": model_id, "model_name": name, "sentiment": sentiment,
            "probabilities": probabilities, "inference_ms": duration,
            "preprocessing": {"input_language": "id", "model_input_language": input_language, "max_length": 256, "truncated": len(model_data.tokenizer.encode(model_input, add_special_tokens=True)) > 256},
            "translation": translation,
        }
        return prediction, translation


registry = ModelRegistry()
