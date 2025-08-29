# backend/services/confidence.py
from typing import List, Tuple
from backend.config import settings

# We no longer hardcode single weights; we compute dynamically based on LLM availability.

def get_weights() -> Tuple[float, float, float]:
    """
    Return (w_ocr, w_heur, w_llm) depending on whether LLM is available.
    - If LLM available: weight more on LLM (0.6), keep OCR+heur smaller.
    - If offline (no LLM): weight heuristics higher and OCR medium.
    """
    llm_available = bool(settings.OPENAI_API_KEY and settings.LLM_PROVIDER and settings.LLM_PROVIDER.lower() == "openai")
    if llm_available:
        # LLM is strong — trust it more
        return (0.2, 0.2, 0.6)
    else:
        # Offline mode — heuristics should be stronger
        return (0.4, 0.6, 0.0)

def normalize_ocr_conf(word_confs: List[float]) -> float:
    """Convert list of 0-100 confidences to 0-1 average. If empty: 0.5 default."""
    if not word_confs:
        return 0.5
    avg = sum(word_confs) / (100.0 * len(word_confs))
    return max(0.0, min(1.0, avg))

def heur_score(valid_format: bool) -> float:
    """Heuristic score: format matched -> high, else medium."""
    return 0.9 if valid_format else 0.6

LLM_MAP = {"high": 0.95, "medium": 0.7, "low": 0.45}

def llm_level(level: str | None) -> float:
    if not level:
        return 0.6
    return LLM_MAP.get(level.lower(), 0.6)

def combine(ocr: float, heur: float, llm: float) -> float:
    """Combine signals using dynamic weights; clamp and round."""
    w_ocr, w_heur, w_llm = get_weights()
    raw = w_ocr * ocr + w_heur * heur + w_llm * llm
    # clamp & round
    val = max(0.0, min(1.0, raw))
    return round(val, 3)

def explanation(ocr: float, heur: float, llm: float) -> str:
    w_ocr, w_heur, w_llm = get_weights()
    return f"ocr={ocr:.3f}, heur={heur:.3f}, llm={llm:.3f}, weights=({w_ocr},{w_heur},{w_llm})"
