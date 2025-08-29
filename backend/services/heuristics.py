# backend/services/heuristics.py
"""
Improved heuristics for marksheet extraction.

This module:
- Extracts candidate-level fields (name, father/mother, roll, registration, dob, exam_year, board/university, institution, issue_date/place)
- Extracts subject rows in several common formats
- Extracts overall result / percentage / division hints
- Normalizes dates when possible (returns YYYY-MM-DD or original short form)
- Uses progressive/fallback strategies to increase hit-rate on noisy OCR

Notes:
- Keep heuristics conservative (prefer empty value over wrong value).
- The LLM normalizer should be fed the full OCR text + heuristics for best results.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# -------------------------
# Precompiled regex patterns
# -------------------------
ROLL_PAT = re.compile(
    r"(?:roll[\s_.\-]*no\.?|rno|rollno|roll)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", re.I
)
REGN_PAT = re.compile(
    r"(?:reg(?:istration)?[\s_.\-]*no\.?|regn)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", re.I
)
DOB_PAT = re.compile(
    r"(?:dob|date\s+of\s+birth|d\.o\.b)\s*[:\-]?\s*([0-3]?\d[\/\-\.\s][01]?\d[\/\-\.\s]\d{2,4})",
    re.I,
)
YEAR_PAT = re.compile(
    r"(?:exam\s*year|year\s*of\s*exam|session|academic\s*year)\s*[:\-]?\s*([12]\d{3})",
    re.I,
)
PERCENT_PAT = re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%")
RESULT_PAT = re.compile(r"(?:result|outcome|status)\s*[:\-]?\s*([A-Za-z ]{3,40})", re.I)

# Subjects table formats
SUBJECT_SLASH = re.compile(
    r"^(.{2,80}?)\s+([0-9]{1,3})\s*[/]\s*([0-9]{1,3})(?:\s+([A-Za-z0-9+\-]{1,6}))?$"
)
SUBJECT_SPACE = re.compile(
    r"^(.{2,80}?)\s+([0-9]{1,3})\s+([0-9]{1,3})(?:\s+([A-Za-z0-9+\-]{1,6}))?$"
)
SUBJECT_COL_CSV = re.compile(r"^([^,]+),\s*([0-9]{1,3}),\s*([0-9]{1,3})(?:,\s*([A-Za-z0-9+\-]+))?$")

# Loose subject detection: subject word + two numbers anywhere in line
LOOSE_SUBJECT = re.compile(r"^(.{2,80}?)[\s:,-]+([0-9]{1,3})[^\d]{1,6}([0-9]{1,3})(?:\s+([A-Za-z0-9+\-]{1,6}))?$")

# Division detection
DIVISION_PAT = re.compile(
    r"\b(first division|second division|third division|distinction)\b", re.I
)


# -------------------------
# Utility helpers
# -------------------------
def try_normalize_date(s: Optional[str]) -> Optional[str]:
    """
    Try common date formats and return YYYY-MM-DD if parseable, else return original trimmed string or None.
    """
    if not s:
        return None
    val = s.strip()
    # remove ordinal suffixes like 'st', 'nd', 'th'
    val = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", val, flags=re.I)
    # common formats to try
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %m %Y",
        "%d-%b-%Y",
    ]
    for fmt in formats:
        try:
            d = datetime.strptime(val, fmt)
            return d.strftime("%Y-%m-%d")
        except Exception:
            pass
    # Try to extract 4-digit year if present
    m = re.search(r"(20\d{2}|19\d{2})", val)
    if m:
        return m.group(1)
    # Not parseable -> return original short cleaned value
    cleaned = re.sub(r"\s+", " ", val)
    return cleaned if cleaned else None


def lines_from_text(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def first_n_lines(text: str, n: int = 8) -> List[str]:
    return lines_from_text(text)[:n]


# -------------------------
# Name extraction helpers
# -------------------------
def extract_name_candidates(lines: List[str]) -> Optional[str]:
    """
    Try a sequence of patterns to find the candidate name.
    - Look for explicit labels (Name:, Candidate:, Student:)
    - Look for 'Name of Candidate' style labels
    - Fallback: first header-like line that looks like a name
    """
    # explicit labels (many variants)
    label_patterns = [
        re.compile(r"\bname\s*[:\-]\s*(.+)", re.I),
        re.compile(r"\bcandidate\s*name\s*[:\-]\s*(.+)", re.I),
        re.compile(r"\bstudent\s*name\s*[:\-]\s*(.+)", re.I),
        re.compile(r"\bname\s+of\s+candidate\s*[:\-]\s*(.+)", re.I),
    ]
    for ln in lines[:20]:
        for p in label_patterns:
            m = p.search(ln)
            if m:
                return m.group(1).strip()

    # Another common pattern: "Candidate: <Name>"
    for ln in lines[:20]:
        if re.search(r"\bcandidate\b", ln, re.I) and ":" in ln:
            parts = ln.split(":", 1)
            if len(parts) > 1 and len(parts[1].strip()) > 2:
                return parts[1].strip()

    # Fallback: first non-empty top line that looks like a name (only letters, dots, spaces, max ~5 words)
    for ln in lines[:3]:
        if re.match(r"^[A-Za-z][A-Za-z .]{2,80}$", ln) and 1 <= len(ln.split()) <= 6:
            return ln.strip()

    return None


# -------------------------
# Candidate block extraction
# -------------------------
def extract_candidate_block(ocr_text: str) -> Dict[str, Any]:
    """
    Extract candidate-level details using multiple heuristics.
    Returns dict with keys similar to final schema (None when unknown).
    """
    lines = lines_from_text(ocr_text)
    text = "\n".join(lines)

    out: Dict[str, Any] = {
        "name": None,
        "father_mother_name": None,
        "roll_no": None,
        "registration_no": None,
        "dob": None,
        "exam_year": None,
        "board_university": None,
        "institution": None,
        "issue_date": None,
        "issue_place": None,
        "result_hint": None,
    }

    # 1) regex direct matches from full text
    if (m := ROLL_PAT.search(text)):
        out["roll_no"] = m.group(1).strip()

    if (m := REGN_PAT.search(text)):
        out["registration_no"] = m.group(1).strip()

    if (m := DOB_PAT.search(text)):
        out["dob"] = try_normalize_date(m.group(1).strip())

    if (m := YEAR_PAT.search(text)):
        out["exam_year"] = m.group(1).strip()

    if (m := RESULT_PAT.search(text)):
        out["result_hint"] = m.group(1).strip()

    if (m := PERCENT_PAT.search(text)):
        # capture a likely percent near top as a hint
        out.setdefault("percentage_hint", m.group(1).strip())

    # 2) targeted name extraction
    name = extract_name_candidates(lines)
    if name:
        out["name"] = name

    # 3) father/mother name
    for ln in lines[:20]:
        if re.search(r"(father|mother|parent|guardian)\b", ln, re.I) and ":" in ln:
            val = ln.split(":", 1)[1].strip()
            out["father_mother_name"] = val
            break

    # 4) Board / Institution detection (look in header area)
    header = lines[:8]
    for ln in header:
        if re.search(r"\b(board|university|council|board of)\b", ln, re.I):
            out["board_university"] = ln.strip()
        if re.search(r"\b(school|college|institute|institution|academy|department)\b", ln, re.I):
            out["institution"] = ln.strip()

    # 5) Issue date/place heuristics
    m = re.search(r"(?:issued\s*(?:on|date)?[:\-\s]*)?([0-3]?\d[\/\-\.\s][01]?\d[\/\-\.\s]\d{2,4})", text, re.I)
    if m:
        out["issue_date"] = try_normalize_date(m.group(1).strip())

    m2 = re.search(r"(?:issued\s*(?:at|from|in)[:\-\s]*)([A-Za-z0-9 ,.-]{2,80})", text, re.I)
    if m2:
        out["issue_place"] = m2.group(1).strip()

    return out


# -------------------------
# Subjects extraction
# -------------------------
def _parse_subject_line(ln: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse a single line into subject_name, obtained_marks, max_marks, grade.
    Returns None if not parsable.
    """
    ln = ln.strip()
    # try strict patterns first
    for patt in (SUBJECT_SLASH, SUBJECT_SPACE, SUBJECT_COL_CSV):
        m = patt.match(ln)
        if m:
            groups = m.groups()
            subj = groups[0].strip()
            obt = groups[1].strip()
            mx = groups[2].strip()
            grd = groups[3].strip() if len(groups) > 3 and groups[3] else None
            return {"subject_name": subj, "obtained_marks": obt, "max_marks": mx, "grade": grd}

    # loose pattern: try to find two numbers in line (obtained and max) and treat rest as subject
    m = re.search(r"([0-9]{1,3})\D{1,6}([0-9]{1,3})", ln)
    if m:
        # subject text is everything before first number
        idx = m.start(1)
        subj = ln[:idx].strip(" .:-")
        obt = m.group(1)
        mx = m.group(2)
        # optional grade at end
        grd_match = re.search(r"([A-Za-z0-9+\-]{1,6})\s*$", ln)
        grd = grd_match.group(1) if grd_match else None
        if subj:
            return {"subject_name": subj, "obtained_marks": obt, "max_marks": mx, "grade": grd}

    # cannot parse
    return None


def extract_subjects(ocr_text: str) -> List[Dict[str, Any]]:
    """
    Extract subject rows from OCR text. Returns list of dictionaries.
    """
    rows: List[Dict[str, Any]] = []
    lines = lines_from_text(ocr_text)

    # try to locate a block that looks like a subject table (one or more consecutive lines matching patterns)
    candidate_block = []
    for ln in lines:
        parsed = _parse_subject_line(ln)
        if parsed:
            candidate_block.append(parsed)
        else:
            # if we already started collecting and encounter a non-matching line, break
            if candidate_block:
                # continue scanning to allow interleaved non-matching lines (robustness)
                # but limit lookahead: collect up to 25 lines total
                if len(candidate_block) >= 1 and len(candidate_block) < 25:
                    continue
                else:
                    break

    # if we found candidate block, return it
    if candidate_block:
        return candidate_block

    # Fallback: scan all lines and collect any loose subject-like matches
    for ln in lines:
        parsed = _parse_subject_line(ln)
        if parsed:
            rows.append(parsed)

    return rows


# -------------------------
# Overall result extraction
# -------------------------
def extract_overall(ocr_text: str) -> Dict[str, Any]:
    """
    Extract overall grade, percentage, and division hints.
    """
    out = {"overall_grade": None, "percentage": None, "division": None}

    if (m := PERCENT_PAT.search(ocr_text)):
        out["percentage"] = m.group(1)

    if (m := RESULT_PAT.search(ocr_text)):
        out["overall_grade"] = m.group(1).strip()

    if (m := DIVISION_PAT.search(ocr_text)):
        out["division"] = m.group(1).strip()

    return out