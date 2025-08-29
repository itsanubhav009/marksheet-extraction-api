from typing import Any, Dict
from backend.schemas import CandidateBlock, SubjectRow, ResultBlock, ExtractionResponse  # or create smaller models
from pydantic import ValidationError

def validate_llm_output(raw: Any, heur_bundle: dict) -> dict:
    """
    raw: the parsed JSON returned from LLM (or heuristics).
    Returns a dict with keys candidate, subjects, result, confidence_hint (guaranteed).
    """
    out = {
        "candidate": {},
        "subjects": [],
        "result": {},
        "confidence_hint": {}
    }
    try:
        # Basic safe-get
        cand = raw.get("candidate", {})
        out["candidate"] = {k: (cand.get(k) or "").strip() for k in [
            "name","father_mother_name","roll_no","registration_no","dob","exam_year","board_university","institution","issue_date","issue_place"
        ]}
        subs = raw.get("subjects", []) or []
        for s in subs:
            out["subjects"].append({
                "subject_name": s.get("subject_name","").strip(),
                "max_marks": s.get("max_marks","").strip(),
                "obtained_marks": s.get("obtained_marks","").strip(),
                "grade": s.get("grade","").strip()
            })
        res = raw.get("result", {}) or {}
        out["result"] = {
            "overall_grade": res.get("overall_grade","").strip(),
            "percentage": res.get("percentage","").strip(),
            "division": res.get("division","").strip()
        }
        # confidence hint
        out["confidence_hint"] = raw.get("confidence_hint", {"candidate":{}, "subjects":"Medium","result":"Medium"})
        return out
    except Exception:
        # fallback to heuristics
        return {
            **heur_bundle,
            "confidence_hint": {"candidate": "Low", "subjects": "Low", "result": "Low"}
        }
