# backend/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

BBox = List[int]  # [x, y, w, h]

class FieldValue(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: Optional[BBox] = None
    raw_text: Optional[str] = None
    note: Optional[str] = None

class SubjectRow(BaseModel):
    subject_name: FieldValue
    max_marks: Optional[FieldValue] = None
    obtained_marks: Optional[FieldValue] = None
    grade: Optional[FieldValue] = None

class CandidateBlock(BaseModel):
    name: FieldValue
    father_mother_name: Optional[FieldValue] = None
    roll_no: Optional[FieldValue] = None
    registration_no: Optional[FieldValue] = None
    dob: Optional[FieldValue] = None
    exam_year: Optional[FieldValue] = None
    board_university: Optional[FieldValue] = None
    institution: Optional[FieldValue] = None
    issue_date: Optional[FieldValue] = None
    issue_place: Optional[FieldValue] = None

class PageInfo(BaseModel):
    page_number: int
    width: Optional[int] = None
    height: Optional[int] = None

class ResultBlock(BaseModel):
    overall_grade: Optional[FieldValue] = None
    percentage: Optional[FieldValue] = None
    division: Optional[FieldValue] = None

class ExtractionResponse(BaseModel):
    document_id: str
    pages: List[PageInfo]
    candidate: CandidateBlock
    subjects: List[SubjectRow]
    result: ResultBlock
    metadata: dict
