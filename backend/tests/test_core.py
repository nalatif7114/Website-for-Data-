import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import CompareRequest, PredictRequest, PredictResponse
from app.services.model_registry import LABELS, ModelRegistry
from app.services.translation import _protect_entities, translate_indonesian


class LabelMappingTests(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()

    def test_checkpoint_label_order_maps_explicitly(self):
        config = SimpleNamespace(id2label={0: "Negatif", 1: "Netral", 2: "Positif"})
        self.assertEqual(self.registry._map_labels(config), {0: "Negatif", 1: "Netral", 2: "Positif"})

    def test_english_label_names_are_supported(self):
        config = SimpleNamespace(id2label={0: "negative", 1: "neutral", 2: "positive"})
        self.assertEqual(set(self.registry._map_labels(config).values()), set(LABELS))

    def test_generic_labels_are_rejected_without_verified_mapping(self):
        with self.assertRaisesRegex(ValueError, "explicitly define"):
            self.registry._map_labels(SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))

    def test_missing_model_id_is_not_available_or_replaced(self):
        self.registry.settings.indobert_model_id = ""
        self.assertFalse(self.registry.status("indobert")["configured"])
        with self.assertRaisesRegex(RuntimeError, "no configured checkpoint"):
            self.registry.get("indobert")

    def test_downloaded_checkpoint_is_distinct_from_loaded_and_ready(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as folder:
            root = Path(folder)
            for filename in ("config.json", "model.safetensors", "tokenizer_config.json"):
                (root / filename).touch()
            self.registry.settings.indobert_model_id = folder
            status = self.registry.status("indobert")
        self.assertTrue(status["downloaded"])
        self.assertFalse(status["loaded"])
        self.assertFalse(status["ready"])
        self.assertEqual(status["state"], "downloaded")

    def test_checkpoint_load_failure_is_reported(self):
        with patch("app.services.model_registry.AutoTokenizer.from_pretrained", side_effect=OSError("repository not found")):
            with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
                self.registry.get("indobert")
        self.assertEqual(self.registry.status("indobert")["state"], "error")


class RequestSchemaTests(unittest.TestCase):
    def test_valid_single_and_comparison_requests(self):
        self.assertEqual(PredictRequest(text="Berita saham meningkat.", model_id="indobert").model_id, "indobert")
        self.assertEqual(len(CompareRequest(text="Berita saham meningkat.").model_ids), 3)

    def test_rejects_duplicate_model_ids_and_short_text(self):
        with self.assertRaises(ValueError):
            CompareRequest(text="Berita saham meningkat.", model_ids=["indobert", "indobert"])
        with self.assertRaises(ValueError):
            PredictRequest(text="abc", model_id="indolem")


class TranslationTests(unittest.TestCase):
    def test_financial_entities_are_protected_and_restored(self):
        protected, entities = _protect_entities("IHSG turun, rupiah melemah; BBRI ikut terkoreksi.")
        self.assertNotIn("IHSG", protected)
        self.assertTrue(any(value.upper() == "IHSG" for value in entities.values()))
        translated = "The INDEX fell, the CURRENCY weakened; BBRI also corrected."
        for placeholder, original in entities.items():
            translated = translated.replace(placeholder, original)
        self.assertIn("BBRI", translated)

    def test_translation_failure_does_not_fallback_to_indonesian(self):
        fake = SimpleNamespace(translate=lambda _text: (_ for _ in ()).throw(RuntimeError("offline")))
        with patch("app.services.translation._translator", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "failed after 4 attempts"):
                translate_indonesian("IHSG mengalami penurunan tajam hari ini.", "google")


class PartialComparisonTests(unittest.TestCase):
    def test_model_failure_does_not_discard_other_results(self):
        from app.main import compare, registry
        successful = {
            "model_id": "indobert", "model_name": "IndoBERT", "sentiment": "Positif",
            "probabilities": {"Negatif": 0.1, "Netral": 0.2, "Positif": 0.7},
            "inference_ms": 3.0, "preprocessing": {"input_language": "id"}, "translation": None,
        }

        def prediction(model_id, _text):
            if model_id == "finbert":
                raise RuntimeError("translation unavailable")
            value = {**successful, "model_id": model_id, "model_name": model_id.title()}
            return value, None

        request = CompareRequest(text="Berita saham naik hari ini.", model_ids=["indobert", "finbert"])
        with patch.object(registry, "predict", side_effect=prediction):
            response = compare(request)
        self.assertEqual([item["model_id"] for item in response["results"]], ["indobert"])
        self.assertEqual(response["errors"][0]["model_id"], "finbert")


class ApiContractTests(unittest.TestCase):
    def test_health_and_model_states_do_not_load_checkpoints(self):
        from app.main import health, models
        self.assertEqual(health()["status"], "ok")
        for model in models()["models"]:
            model = model.model_dump()
            self.assertIn("configured", model)
            self.assertIn("downloaded", model)
            self.assertIn("loaded", model)
            self.assertIn("ready", model)
            self.assertFalse(model["loaded"])

    def test_predict_response_schema_and_request_validation(self):
        from app.main import predict, registry
        fake = {
            "model_id": "indobert", "model_name": "IndoBERT", "sentiment": "Positif",
            "probabilities": {"Negatif": 0.1, "Netral": 0.2, "Positif": 0.7}, "inference_ms": 1.0,
            "preprocessing": {"input_language": "id", "model_input_language": "id", "max_length": 256, "truncated": False}, "translation": None,
        }
        with patch.object(registry, "predict", return_value=(fake, None)):
            response = PredictResponse.model_validate(predict(PredictRequest(text="Harga saham tumbuh positif.", model_id="indobert")))
        self.assertEqual(response.prediction.sentiment, "Positif")
        with self.assertRaises(ValueError):
            PredictRequest(text="abc", model_id="indobert")


if __name__ == "__main__":
    unittest.main()
