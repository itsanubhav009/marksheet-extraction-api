# backend/routers/extract.py
import io
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from backend.config import settings
from backend.schemas import (
    ExtractionResponse,
    FieldValue,
    SubjectRow,
    CandidateBlock,
    ResultBlock,
    PageInfo,
)
from backend.services.ocr import run_ocr
from backend.services.preprocess import preprocess_image
from backend.services.heuristics import (
    extract_candidate_block,
    extract_subjects,
    extract_overall,
)
from backend.services.llm_normalizer import normalize_with_llm
from backend.services.confidence import (
    normalize_ocr_conf,
    heur_score,
    llm_level,
    combine,
    explanation,
)
from backend.utils.file_handler import pdf_bytes_to_images

logger = logging.getLogger("router.extract")
router = APIRouter()

# -------------------
# Helpers
# -------------------

def _validate_filetype(file: UploadFile) -> None:
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415, detail=f"Unsupported media type: {file.content_type}"
        )


def _safe_get_normalized(normalized: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "candidate": {},
        "subjects": [],
        "result": {},
        "confidence_hint": {"candidate": {}, "subjects": "Medium", "result": "Medium"},
    }
    if not isinstance(normalized, dict):
        return out
    out["candidate"] = normalized.get("candidate", {}) or {}
    out["subjects"] = normalized.get("subjects", []) or []
    out["result"] = normalized.get("result", {}) or {}
    ch = normalized.get("confidence_hint", {})
    if isinstance(ch, dict):
        out["confidence_hint"] = ch
    return out


def find_bbox_for_text(value: Optional[str], word_rows: List[Dict[str, Any]]) -> Optional[List[int]]:
    if not value or not word_rows:
        return None
    tokens = [t.strip() for t in value.split() if t.strip()]
    if not tokens:
        return None

    words = [w.get("text", "").strip() for w in word_rows]
    words_lower = [w.lower() for w in words]

    first = tokens[0].lower()
    indices = [i for i, w in enumerate(words_lower) if w == first]
    for start in indices:
        j = start
        matched_indices = [start]
        ok = True
        for tkn in tokens[1:]:
            j += 1
            if j >= len(words_lower) or words_lower[j] != tkn.lower():
                ok = False
                break
            matched_indices.append(j)
        if ok:
            xs, ys, x2, y2 = [], [], [], []
            for idx in matched_indices:
                w = word_rows[idx]
                left, top, width, height = int(w.get("left", 0)), int(w.get("top", 0)), int(w.get("width", 0)), int(w.get("height", 0))
                xs.append(left); ys.append(top); x2.append(left + width); y2.append(top + height)
            return [min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)]

    for i, w in enumerate(word_rows):
        if w.get("text", "").strip().lower() == first:
            left, top, width, height = int(w.get("left", 0)), int(w.get("top", 0)), int(w.get("width", 0)), int(w.get("height", 0))
            return [left, top, width, height]

    return None


# -------------------
# Endpoint
# -------------------

@router.post("/extract", response_model=ExtractionResponse)
async def extract_endpoint(file: UploadFile = File(...)) -> ExtractionResponse:
    _validate_filetype(file)
    raw = await file.read()
    if len(raw) > settings.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.MAX_FILE_MB} MB allowed.")

    # --- Step 1: Load as images ---
    try:
        if file.content_type == "application/pdf":
            images = pdf_bytes_to_images(raw, poppler_path=settings.POPPLER_PATH)
        else:
            images = [Image.open(io.BytesIO(raw)).convert("RGB")]
    except Exception as e:
        logger.exception("Failed to load file")
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    if not images:
        raise HTTPException(status_code=422, detail="No pages found in uploaded file")

    # --- Step 2: OCR ---
    page_infos, ocr_texts, ocr_word_confs, ocr_word_rows_all = [], [], [], []
    for i, img in enumerate(images, start=1):
        pre = preprocess_image(img)
        text, word_rows = run_ocr(pre)
        ocr_texts.append(text or "")
        page_infos.append(PageInfo(page_number=i, width=img.width, height=img.height))
        for w in word_rows:
            try:
                ocr_word_confs.append(float(w.get("conf", 0.0)))
            except Exception:
                pass
        ocr_word_rows_all.extend(word_rows)

    joined_text = "\n".join(ocr_texts)
    if not joined_text.strip():
        raise HTTPException(status_code=422, detail="OCR produced no readable text")

    # --- Step 3: Heuristics ---
    heur_bundle = {
        "candidate": extract_candidate_block(joined_text),
        "subjects": extract_subjects(joined_text),
        "result": extract_overall(joined_text),
    }

    # --- Step 4: LLM Normalization (with schema) ---
    llm_used = False
    try:
        normalized_raw = normalize_with_llm(joined_text, heur_bundle, enforce_schema=True)
        llm_used = True
    except Exception as e:
        logger.warning("LLM normalization failed: %s", e)
        normalized_raw = heur_bundle

    normalized = _safe_get_normalized(normalized_raw)

    # --- Step 5: Confidence Scoring ---
    ocr_conf_avg = normalize_ocr_conf(ocr_word_confs)

    # dynamic weight shift
    if llm_used:
        llm_weight = 0.6
        heur_weight = 0.2
        ocr_weight = 0.2
    else:
        llm_weight = 0.2
        heur_weight = 0.5
        ocr_weight = 0.3

    # Candidate block helper
    def mk_field(key: str, raw_val: Optional[str], fmt_ok: bool = True) -> FieldValue:
        ocr_c = ocr_conf_avg * ocr_weight
        heur_c = heur_score(fmt_ok) * heur_weight
        llm_c = llm_level(normalized.get("confidence_hint", {}).get("candidate", {}).get(key)) * llm_weight
        conf = round(ocr_c + heur_c + llm_c, 3)
        note = f"ocr={ocr_c:.3f}, heur={heur_c:.3f}, llm={llm_c:.3f}, weights=(o={ocr_weight},h={heur_weight},l={llm_weight})"
        bbox = find_bbox_for_text(raw_val, ocr_word_rows_all)
        return FieldValue(value=raw_val or None, confidence=conf, raw_text=raw_val, note=note, bbox=bbox)

    candidate_block = CandidateBlock(
        name=mk_field("name", normalized["candidate"].get("name")),
        father_mother_name=mk_field("father_mother_name", normalized["candidate"].get("father_mother_name")),
        roll_no=mk_field("roll_no", normalized["candidate"].get("roll_no")),
        registration_no=mk_field("registration_no", normalized["candidate"].get("registration_no")),
        dob=mk_field("dob", normalized["candidate"].get("dob")),
        exam_year=mk_field("exam_year", normalized["candidate"].get("exam_year")),
        board_university=mk_field("board_university", normalized["candidate"].get("board_university")),
        institution=mk_field("institution", normalized["candidate"].get("institution")),
        issue_date=mk_field("issue_date", normalized["candidate"].get("issue_date")),
        issue_place=mk_field("issue_place", normalized["candidate"].get("issue_place")),
    )

    # Subjects + Results similar (skipped here for brevity but follows same pattern)
    # ...

    # --- Step 6: Response ---
    doc_id = f"{uuid.uuid4()}_{file.filename}"
    warnings = []
    if not llm_used:
        warnings.append("LLM normalization failed/fallback → heuristics only")
    if ocr_conf_avg < 0.45:
        warnings.append("Low OCR average confidence")

    return ExtractionResponse(
        document_id=doc_id,
        pages=page_infos,
        candidate=candidate_block,
        subjects=[],  # TODO: build same as candidate using mk_field + weights
        result=ResultBlock(),
        metadata={
            "ocr_engine": "tesseract/pytesseract",
            "ocr_confidence_avg": ocr_conf_avg,
            "llm_provider": settings.LLM_PROVIDER if llm_used else "disabled",
            "llm_model": settings.OPENAI_MODEL if llm_used else None,
            "total_pages": len(images),
            "warnings": warnings,
        },
    )
