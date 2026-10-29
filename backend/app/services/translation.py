import re
import time
from functools import lru_cache

PRESERVE_TOKENS = [
    "Bank Indonesia", "Pertamina", "IHSG", "BEI", "OJK", "BPJS", "BUMN", "APBN", "PPN", "PPh",
    "BBRI", "BBCA", "BMRI", "BBNI", "BBTN", "TLKM", "ASII", "UNVR", "GGRM", "rupiah", "Rupiah", "IDR", "PLN", "Telkom", "BI", "BPS",
]


@lru_cache(maxsize=1)
def _translator(provider: str):
    if provider != "google":
        raise RuntimeError(f"Unsupported TRANSLATION_PROVIDER={provider!r}; this research workflow uses deep-translator GoogleTranslator")
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise RuntimeError("FinBERT requires deep-translator. Install backend requirements to enable translation.") from exc
    return GoogleTranslator(source="id", target="en")


def _protect_entities(text: str) -> tuple[str, dict[str, str]]:
    entities: dict[str, str] = {}
    protected = text
    for index, token in enumerate(sorted(set(PRESERVE_TOKENS), key=len, reverse=True)):
        placeholder = f"FINENTITY{index:03d}TOKEN"
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        if pattern.search(protected):
            protected, count = pattern.subn(placeholder, protected)
            if count:
                entities[placeholder] = token
    return protected, entities


def translate_indonesian(text: str, provider: str) -> tuple[str, str]:
    translator = _translator(provider)
    protected, entities = _protect_entities(text[:4500])
    error: Exception | None = None
    for attempt in range(4):
        try:
            translated = translator.translate(protected)
            if not translated or not translated.strip():
                raise RuntimeError("Translation provider returned empty text")
            for placeholder, original in entities.items():
                translated = re.sub(re.escape(placeholder), original, translated, flags=re.IGNORECASE)
            return translated, "deep-translator GoogleTranslator"
        except Exception as exc:  # retries match the research pipeline's four attempts
            error = exc
            if attempt < 3:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Translation failed after 4 attempts: {error}") from error
