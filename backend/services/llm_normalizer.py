import json
import logging
from typing import Any, Dict, List

from backend.config import settings
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("services.llm_normalizer")

GEMINI_API_KEY = "AIzaSyBf9fBDiDJE87GfHIvxwwtgWw2zXJM6VIE"

# ---------------------------
# Pydantic schemas
# ---------------------------

class LLMSubject(BaseModel):
    subject_name: str = Field(default="")
    max_marks: str = Field(default="")
    obtained_marks: str = Field(default="")
    grade: str = Field(default="")

class LLMCandidate(BaseModel):
    name: str = Field(default="")
    father_mother_name: str = Field(default="")
    roll_no: str = Field(default="")
    registration_no: str = Field(default="")
    dob: str = Field(default="")
    exam_year: str = Field(default="")
    board_university: str = Field(default="")
    institution: str = Field(default="")
    issue_date: str = Field(default="")
    issue_place: str = Field(default="")

class LLMResult(BaseModel):
    overall_grade: str = Field(default="")
    percentage: str = Field(default="")
    division: str = Field(default="")

class LLMNormalized(BaseModel):
    candidate: LLMCandidate
    subjects: List[LLMSubject] = Field(default_factory=list)
    result: LLMResult
    confidence_hint: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------
# Fallback
# ---------------------------

def _fallback(heuristics_bundle: dict) -> dict:
    """Return heuristics with conservative confidence hint."""
    return {
        "candidate": heuristics_bundle.get("candidate", {}),
        "subjects": heuristics_bundle.get("subjects", []),
        "result": heuristics_bundle.get("result", {}),
        "confidence_hint": {
            "candidate": {"name": "Medium", "roll_no": "Medium"},
            "subjects": "Medium",
            "result": "Medium",
        },
    }

# ---------------------------
# LLM Normalization
# ---------------------------

def normalize_with_llm(ocr_text: str, heuristics_bundle: dict) -> dict:
    """
    Normalize OCR + heuristics into a clean JSON structure
    using OpenAI or Gemini, based on settings.LLM_PROVIDER.
    """
    provider = (settings.LLM_PROVIDER or "").lower()

    # ---------------- OpenAI ----------------
    if provider == "openai":
        try:
            import openai
        except Exception:
            logger.warning("OpenAI not installed; fallback to heuristics")
            return _fallback(heuristics_bundle)

        if not settings.OPENAI_API_KEY:
            logger.info("OpenAI API key missing; fallback")
            return _fallback(heuristics_bundle)

        prompt = f"""
You are a marksheet data normalizer. Input contains OCR text (noisy) and heuristic candidates.
Your task is to return a STRICT JSON object with the schema described below.

OCR_TEXT:
\"\"\"{ocr_text[:15000]}\"\"\"

HEURISTICS (best-effort):
\"\"\"{json.dumps(heuristics_bundle, ensure_ascii=False)[:8000]}\"\"\"

Return JSON exactly like:

{{
  "candidate": {{
    "name": "", "father_mother_name": "", "roll_no": "", "registration_no": "",
    "dob": "", "exam_year": "", "board_university": "", "institution": "",
    "issue_date": "", "issue_place": ""
  }},
  "subjects": [
    {{"subject_name": "", "max_marks": "", "obtained_marks": "", "grade": ""}}
  ],
  "result": {{"overall_grade": "", "percentage": "", "division": ""}},
  "confidence_hint": {{
    "candidate": {{"name":"High|Medium|Low","roll_no":"High|Medium|Low"}},
    "subjects": "High|Medium|Low",
    "result": "High|Medium|Low"
  }}
}}
"""

        try:
            openai.api_key = settings.OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a JSON-only response engine. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1500,
            )

            content = response.choices[0].message["content"].strip()
            parsed = json.loads(content)
            validated = LLMNormalized.parse_obj(parsed)
            return validated.dict()

        except Exception as e:
            logger.exception("OpenAI normalization failed: %s", e)
            return _fallback(heuristics_bundle)

    # ---------------- Gemini ----------------
    elif provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except Exception:
            logger.warning("Gemini client not installed; fallback")
            return _fallback(heuristics_bundle)

        if not settings.GEMINI_API_KEY:
            logger.info("Gemini API key missing; fallback")
            return _fallback(heuristics_bundle)

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "candidate": types.Schema(
                    type=types.Type.OBJECT,
                    properties={f: types.Schema(type=types.Type.STRING) for f in LLMCandidate.__fields__}
                ),
                "subjects": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={f: types.Schema(type=types.Type.STRING) for f in LLMSubject.__fields__}
                    ),
                ),
                "result": types.Schema(
                    type=types.Type.OBJECT,
                    properties={f: types.Schema(type=types.Type.STRING) for f in LLMResult.__fields__}
                ),
                "confidence_hint": types.Schema(type=types.Type.OBJECT),
            },
            required=["candidate", "subjects", "result", "confidence_hint"],
        )

        prompt = f"""
Normalize this OCR text and heuristic bundle into strict JSON.

OCR_TEXT:
{ocr_text[:10000]}

HEURISTICS:
{json.dumps(heuristics_bundle, ensure_ascii=False)[:5000]}
"""

        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )

            parsed = response.parsed
            validated = LLMNormalized.parse_obj(parsed)
            return validated.dict()

        except Exception as e:
            logger.exception("Gemini normalization failed: %s", e)
            return _fallback(heuristics_bundle)

    # ---------------- Default: fallback ----------------
    else:
        logger.info(f"Unknown LLM_PROVIDER={provider}; fallback")
        return _fallback(heuristics_bundle)
