"""
Universal Document Intelligence Pipeline — V2.

This module handles multi-format document ingestion with explicit validation
states, quality checks, and human-readable error messages. It never assumes
a file is a resume based on filename alone — it detects the actual content
type, extracts text using the appropriate parser, evaluates extraction
quality, and classifies the document type before any candidate analysis.

Validation States:
  - VALID_RESUME: Document is a valid, readable resume with sufficient info
  - RESUME_REQUIRES_OCR: Image-based/scanned resume detected, needs OCR
  - LOW_EXTRACTION_QUALITY: Text extracted but quality is poor/incomplete
  - NOT_A_RESUME: Document is valid but not a resume (invoice, ID, etc.)
  - CORRUPTED_FILE: File is corrupted or password-protected
  - UNSUPPORTED_DOCUMENT: Format not supported
  - INSUFFICIENT_INFORMATION: Resume too sparse for meaningful analysis
  - EXTRACTION_FAILED: Unexpected error during extraction

Design principles:
  1. Never silently continue when validation fails
  2. Explain problems in simple human language
  3. Tell the user what to do next
  4. Never fabricate missing information
  5. Never generate job recommendations for invalid documents
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger("services.document_intelligence")


class DocumentValidationState(str, Enum):
    """Explicit validation states for document processing."""
    VALID_RESUME = "valid_resume"
    RESUME_REQUIRES_OCR = "resume_requires_ocr"
    LOW_EXTRACTION_QUALITY = "low_extraction_quality"
    NOT_A_RESUME = "not_a_resume"
    CORRUPTED_FILE = "corrupted_file"
    UNSUPPORTED_DOCUMENT = "unsupported_document"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    EXTRACTION_FAILED = "extraction_failed"


class DocumentType(str, Enum):
    """Detected document types."""
    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    JOB_DESCRIPTION = "job_description"
    CERTIFICATE = "certificate"
    MARKSHEET = "marksheet"
    INVOICE = "invoice"
    ID_DOCUMENT = "id_document"
    PORTFOLIO = "portfolio"
    UNKNOWN = "unknown"


@dataclass
class DocumentValidationResult:
    """Complete validation result with human-readable messages."""
    state: DocumentValidationState
    document_type: DocumentType
    raw_text: str
    file_type_detected: str
    file_type_expected: str | None
    extraction_quality_score: float  # 0.0 to 1.0
    
    # Human-readable messages
    user_message: str
    next_action: str
    
    # Technical details for debugging
    technical_details: dict[str, Any]
    
    # Extracted metadata (only populated for valid resumes)
    detected_name: str | None = None
    detected_email: str | None = None
    detected_phone: str | None = None
    detected_location: str | None = None
    detected_total_experience_years: float | None = None
    detected_education_level: int | None = None  # 0=none,1=bachelor,2=master,3=phd
    detected_skills_count: int = 0
    detected_experience_entries: int = 0


# Detection patterns for document classification
RESUME_INDICATORS = [
    r"\b(work experience|professional experience|employment history)\b",
    r"\b(education|academic background|qualifications)\b",
    r"\b(skills|technical skills|core competencies)\b",
    r"\b(projects|personal projects|academic projects)\b",
    r"\b(certifications|certificates|licenses)\b",
    r"\b(references|achievements|awards)\b",
]

COVER_LETTER_INDICATORS = [
    r"\b(dear hiring manager|dear recruiter|to whom it may concern)\b",
    r"\b(i am writing to apply|i would like to apply)\b",
    r"\b(sincerely|best regards|yours faithfully)\b",
]

JOB_DESCRIPTION_INDICATORS = [
    r"\b(we are hiring|we are looking for|join our team)\b",
    r"\b(responsibilities include|key responsibilities|duties)\b",
    r"\b(requirements|qualifications required|must have)\b",
    r"\b(what we offer|benefits|perks)\b",
]

CERTIFICATE_INDICATORS = [
    r"\b(certificate of completion|this certifies that|hereby certified)\b",
    r"\b(issued by|awarded to|completion date)\b",
]

MARKSHEET_INDICATORS = [
    r"\b(marks obtained|grade point|gpa|percentage)\b",
    r"\b(subject|course|credit|semester)\b",
    r"\b(result sheet|transcript|academic record)\b",
]

INVOICE_INDICATORS = [
    r"\b(invoice number|bill to|amount due|payment terms)\b",
    r"\b(tax id|gst|vat|total amount)\b",
]

ID_DOCUMENT_INDICATORS = [
    r"\b(passport number|national id|driver.?s? license)\b",
    r"\b(date of birth|place of birth|issue date|expiry date)\b",
]

# Email and phone patterns
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# PDF extractors sometimes insert spaces around @ or dots in emails.
EMAIL_SPACED_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}"
)
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
MAILTO_PATTERN = re.compile(r"mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", re.I)

# Education level patterns
EDUCATION_LEVELS = {
    3: [r"\bphd\b", r"\bdoctorate\b", r"\bd\.?phil\b"],
    2: [r"\bmaster[''']?s?\b", r"\bmsc\b", r"\bm\.sc\b", r"\bmba\b", 
        r"\bm\.eng\b", r"\bpostgraduate\b"],
    1: [r"\bbachelor[''']?s?\b", r"\bbsc\b", r"\bb\.sc\b", 
        r"\bb\.eng\b", r"\bbe\b", r"\bundergraduate\b", r"\bdiploma\b"],
}

# Experience year patterns
EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?|yr)\s+(?:of\s+)?experience", re.I),
    re.compile(r"(?:minimum|min\.?\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)", re.I),
    re.compile(r"(\d+)\+?\s*(?:years?|yrs?|yr)\s+(?:of\s+)?(?:professional|work|industry)", re.I),
]


def detect_file_type_by_content(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Detect actual file type from magic bytes, not just extension.
    
    Returns: (detected_type, expected_type_from_extension)
    """
    # Magic byte signatures
    if file_bytes[:4] == b"%PDF":
        detected = "application/pdf"
    elif file_bytes[:2] == b"PK":
        # Could be DOCX, XLSX, PPTX, ODT - need more checking
        if b"word/" in file_bytes[:1000]:
            detected = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif b"xl/" in file_bytes[:1000]:
            detected = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif b"ppt/" in file_bytes[:1000]:
            detected = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        else:
            detected = "application/zip"  # Generic ZIP
    elif file_bytes[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        detected = "application/msword"  # Old .doc format
    elif file_bytes[:8] == b"mimetype":
        detected = "application/vnd.oasis.opendocument.text"  # ODT
    elif file_bytes[:6] in (b"\x89PNG\r\n", b"GIF87a", b"GIF89a"):
        detected = "image/png" if file_bytes[:6] == b"\x89PNG\r\n" else "image/gif"
    elif file_bytes[:2] == b"\xff\xd8":
        detected = "image/jpeg"
    elif file_bytes[:8] == b"RIFF" and file_bytes[8:12] == b"WEBP":
        detected = "image/webp"
    else:
        # Try UTF-8/ASCII text
        try:
            text_sample = file_bytes[:1000].decode("utf-8")
            if text_sample.strip():
                detected = "text/plain"
            else:
                detected = "application/octet-stream"
        except UnicodeDecodeError:
            detected = "application/octet-stream"
    
    # Expected type from extension
    ext = Path(filename).suffix.lower()
    extension_map = {
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".rtf": "application/rtf",
        ".odt": "application/vnd.oasis.opendocument.text",
        ".html": "text/html",
        ".htm": "text/html",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    expected = extension_map.get(ext, "application/octet-stream")
    
    return detected, expected


def extract_text_from_bytes(file_bytes: bytes, file_type: str, filename: str) -> tuple[str, dict]:
    """Extract text from file bytes using appropriate parser.
    
    Returns: (extracted_text, extraction_metadata)
    """
    metadata = {
        "parser_used": None,
        "pages_extracted": 0,
        "images_found": 0,
        "extraction_warnings": [],
    }
    
    try:
        if file_type == "application/pdf":
            return _extract_from_pdf(file_bytes, filename, metadata)
        elif file_type in ("application/msword", 
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            return _extract_from_docx(file_bytes, metadata)
        elif file_type == "text/plain":
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            metadata["parser_used"] = "utf-8_decoder"
            metadata["pages_extracted"] = 1
            return text, metadata
        elif file_type in ("text/html", "text/html"):
            return _extract_from_html(file_bytes, metadata)
        elif file_type.startswith("image/"):
            metadata["parser_used"] = "image_requires_ocr"
            metadata["images_found"] = 1
            return "", metadata  # Empty text signals OCR needed
        elif file_type == "application/rtf":
            return _extract_from_rtf(file_bytes, metadata)
        elif file_type == "application/vnd.oasis.opendocument.text":
            return _extract_from_odt(file_bytes, metadata)
        else:
            # Try as plain text fallback
            try:
                text = file_bytes.decode("utf-8", errors="ignore").strip()
                metadata["parser_used"] = "fallback_utf-8"
                return text, metadata
            except Exception:
                metadata["extraction_warnings"].append("Could not decode as text")
                return "", metadata
                
    except Exception as e:
        metadata["extraction_warnings"].append(f"Extraction error: {str(e)}")
        logger.exception("document.extraction_error", filename=filename)
        return "", metadata


def _extract_from_pdf(file_bytes: bytes, filename: str, metadata: dict) -> tuple[str, dict]:
    """Extract text from PDF with quality checks."""
    try:
        from PyPDF2 import PdfReader
        
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
            except Exception as e:
                metadata["extraction_warnings"].append(f"Page {i+1} extraction failed: {e}")
        
        metadata["parser_used"] = "PyPDF2"
        metadata["pages_extracted"] = len(pages)
        
        # Check if PDF has extractable text or is image-based
        total_chars = sum(len(p) for p in pages)
        if total_chars < 100 and len(pdf_reader.pages) > 0:
            # Likely an image-based/scanned PDF
            metadata["images_found"] = len(pdf_reader.pages)
            metadata["extraction_warnings"].append(
                "PDF appears to be image-based/scanned. OCR required."
            )
            return "", metadata
        
        full_text = "\n".join(pages).strip()
        return full_text, metadata
        
    except ImportError:
        metadata["extraction_warnings"].append("PyPDF2 not installed")
        return "", metadata
    except Exception as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            metadata["extraction_warnings"].append("PDF is password-protected")
        else:
            metadata["extraction_warnings"].append(f"PDF extraction failed: {e}")
        return "", metadata


def _extract_from_docx(file_bytes: bytes, metadata: dict) -> tuple[str, dict]:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text.strip())
        
        metadata["parser_used"] = "python-docx"
        metadata["pages_extracted"] = len(paragraphs)
        
        return "\n".join(paragraphs), metadata
        
    except ImportError:
        metadata["extraction_warnings"].append("python-docx not installed")
        # Fallback: treat as ZIP and try to extract raw XML
        return _extract_docx_fallback(file_bytes, metadata)
    except Exception as e:
        metadata["extraction_warnings"].append(f"DOCX extraction failed: {e}")
        return "", metadata


def _extract_docx_fallback(file_bytes: bytes, metadata: dict) -> tuple[str, dict]:
    """Fallback DOCX extraction using zipfile directly."""
    import zipfile
    import xml.etree.ElementTree as ET
    
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            document_xml = z.read("word/document.xml")
            root = ET.fromstring(document_xml)
            
            # Extract all text nodes
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for para in root.iter(".//w:p"):
                text_parts = []
                for run in para.iter(".//w:t"):
                    if run.text:
                        text_parts.append(run.text)
                if text_parts:
                    paragraphs.append("".join(text_parts))
            
            metadata["parser_used"] = "docx_zip_fallback"
            return "\n".join(paragraphs), metadata
            
    except Exception as e:
        metadata["extraction_warnings"].append(f"DOCX fallback failed: {e}")
        return "", metadata


def _extract_from_html(file_bytes: bytes, metadata: dict) -> tuple[str, dict]:
    """Extract text from HTML, removing tags and scripts."""
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        
        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        
        # Decode common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        metadata["parser_used"] = "html_regex"
        metadata["pages_extracted"] = 1
        
        return text, metadata
        
    except Exception as e:
        metadata["extraction_warnings"].append(f"HTML extraction failed: {e}")
        return "", metadata


def _extract_from_rtf(file_bytes: bytes, metadata: dict) -> tuple[str, dict]:
    """Extract text from RTF file."""
    try:
        # Simple RTF text extraction (strips RTF control words)
        text = file_bytes.decode("utf-8", errors="ignore")
        
        # Remove RTF control sequences
        text = re.sub(r"\\[a-z]+\d*\s?", " ", text)
        text = re.sub(r"\\[{}]", "", text)
        text = re.sub(r"[{}]", "", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        metadata["parser_used"] = "rtf_regex"
        metadata["pages_extracted"] = 1
        
        return text, metadata
        
    except Exception as e:
        metadata["extraction_warnings"].append(f"RTF extraction failed: {e}")
        return "", metadata


def _extract_from_odt(file_bytes: bytes, metadata: dict) -> tuple[str, dict]:
    """Extract text from ODT file."""
    import zipfile
    import xml.etree.ElementTree as ET
    
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            content_xml = z.read("content.xml")
            root = ET.fromstring(content_xml)
            
            ns = {"office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
                  "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
            
            paragraphs = []
            for para in root.iter(".//text:p"):
                if para.text:
                    paragraphs.append(para.text)
            
            metadata["parser_used"] = "odt_xml"
            metadata["pages_extracted"] = len(paragraphs)
            
            return "\n".join(paragraphs), metadata
            
    except Exception as e:
        metadata["extraction_warnings"].append(f"ODT extraction failed: {e}")
        return "", metadata


def classify_document_type(text: str) -> tuple[DocumentType, float]:
    """Classify document type based on content indicators.
    
    Returns: (document_type, confidence_score)
    """
    if len(text.strip()) < 50:
        return DocumentType.UNKNOWN, 0.0
    
    text_lower = text.lower()
    
    # Score each document type
    scores = {
        DocumentType.RESUME: sum(1 for p in RESUME_INDICATORS if re.search(p, text_lower)),
        DocumentType.COVER_LETTER: sum(1 for p in COVER_LETTER_INDICATORS if re.search(p, text_lower)),
        DocumentType.JOB_DESCRIPTION: sum(1 for p in JOB_DESCRIPTION_INDICATORS if re.search(p, text_lower)),
        DocumentType.CERTIFICATE: sum(1 for p in CERTIFICATE_INDICATORS if re.search(p, text_lower)),
        DocumentType.MARKSHEET: sum(1 for p in MARKSHEET_INDICATORS if re.search(p, text_lower)),
        DocumentType.INVOICE: sum(1 for p in INVOICE_INDICATORS if re.search(p, text_lower)),
        DocumentType.ID_DOCUMENT: sum(1 for p in ID_DOCUMENT_INDICATORS if re.search(p, text_lower)),
    }
    
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]
    
    # Calculate confidence based on indicator count and dominance
    total_indicators = sum(scores.values())
    if total_indicators == 0:
        return DocumentType.UNKNOWN, 0.0
    
    confidence = min(1.0, max_score / 2.0)  # 2+ indicators = high confidence
    
    # Check if second-place is close (ambiguous)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] <= 1:
        confidence *= 0.7  # Reduce confidence for ambiguous cases
    
    return max_type, confidence


def _normalize_extraction_text(text: str) -> str:
    """Normalize common PDF extraction artifacts before field detection."""
    # Strip zero-width / BOM characters that break email regexes.
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse soft hyphen leftovers.
    cleaned = cleaned.replace("\u00ad", "")
    return cleaned


def extract_personal_info(text: str) -> dict:
    """Extract personal information from resume text."""
    text = _normalize_extraction_text(text)
    lines = text.split("\n")

    # Email — exact match, mailto:, or PDF-spaced variants
    emails = EMAIL_PATTERN.findall(text)
    email = emails[0] if emails else None
    if not email:
        mailto = MAILTO_PATTERN.search(text)
        if mailto:
            email = mailto.group(1)
    if not email:
        spaced = EMAIL_SPACED_PATTERN.search(text)
        if spaced:
            email = re.sub(r"\s+", "", spaced.group(0))

    # Phone
    phones = PHONE_PATTERN.findall(text)
    phone = phones[0] if phones else None

    # Name (heuristic: often on first non-empty line, capitalized)
    name = None
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 60:
            # Check if mostly capitalized words (likely a name)
            words = line.split()
            if all(w[0].isupper() for w in words if w) and len(words) <= 4:
                # Exclude common non-name headers
                if not any(h in line.lower() for h in ["resume", "cv", "curriculum", "contact"]):
                    name = line
                    break

    # Location (look for city names)
    location = None
    location_patterns = [
        r"\b(mumbai|delhi|bangalore|bengaluru|chennai|hyderabad|pune|ahmedabad|kolkata)\b",
        r"\b(london|new york|san francisco|berlin|singapore|toronto|dublin)\b",
        r"\b(remote|worldwide|global)\b",
        r"\b([A-Za-z][A-Za-z\s]+),\s*(Maharashtra|Karnataka|Tamil Nadu|Delhi|India)\b",
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            location = match.group(0).strip().title() if match.lastindex and match.lastindex >= 2 else match.group(1).title()
            break

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
    }


def extract_education_level(text: str) -> int:
    """Extract highest education level from text."""
    text_lower = text.lower()
    for level in (3, 2, 1):
        for pattern in EDUCATION_LEVELS[level]:
            if re.search(pattern, text_lower):
                return level
    return 0


def extract_experience_years(text: str) -> float | None:
    """Extract total years of experience from text."""
    matches = []
    for pattern in EXPERIENCE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) == 2:  # Range like "3-5 years"
                try:
                    avg = (float(groups[0]) + float(groups[1])) / 2
                    matches.append(avg)
                except ValueError:
                    pass
            elif groups:  # Single value
                try:
                    matches.append(float(groups[0]))
                except ValueError:
                    pass
    
    return max(matches) if matches else None


def calculate_extraction_quality(text: str, metadata: dict) -> float:
    """Calculate extraction quality score (0.0 to 1.0)."""
    if not text.strip():
        return 0.0
    
    score = 1.0
    
    # Penalize for very short text
    char_count = len(text.strip())
    if char_count < 200:
        score -= 0.4
    elif char_count < 500:
        score -= 0.2
    elif char_count < 1000:
        score -= 0.1
    
    # Penalize for extraction warnings
    warning_count = len(metadata.get("extraction_warnings", []))
    score -= min(0.3, warning_count * 0.1)
    
    # Penalize for low word density (many spaces/special chars)
    word_count = len(text.split())
    if char_count > 0:
        word_density = word_count / (char_count / 5)  # Expected ~5 chars per word
        if word_density < 0.5:
            score -= 0.2
    
    return max(0.0, min(1.0, score))


def validate_resume_sufficiency(text: str, personal_info: dict) -> tuple[bool, list[str]]:
    """Check if resume has sufficient information for analysis.

    Formal work experience is optional — student/fresher resumes with
    projects, education, and skills are valid. Skills are detected both
    via section headers and via taxonomy skill extraction so tech-stack
    lists without a "Skills" heading still count.
    """
    missing = []
    text = _normalize_extraction_text(text)
    text_lower = text.lower()

    has_education = any(re.search(p, text_lower) for p in [
        r"\b(education|degree|university|college|school|b\.?tech|b\.?e\.|m\.?tech|bachelor|master)\b"
    ])
    has_skills_section = any(re.search(p, text_lower) for p in [
        r"\b(skill|proficient|familiar|expertise|tool|technology|tech stack|"
        r"technologies|programming languages|frameworks|languages)\b"
    ])
    has_projects_or_certs = any(re.search(p, text_lower) for p in [
        r"\b(project|projects|certification|certifications|hackathon|internship)\b"
    ])

    from app.services.taxonomy import extract_skills
    extracted_skills = extract_skills(text)
    has_skills = has_skills_section or len(extracted_skills) > 0

    if not has_education:
        missing.append("education")
    if not has_skills:
        missing.append("skills")
    if not personal_info.get("email"):
        missing.append("contact email")
    if len(text.strip()) < 300:
        missing.append("sufficient detail")

    # Fresher path: education + (skills or projects) + enough text is enough.
    # Formal work experience is never required.
    if has_education and has_skills and len(text.strip()) >= 300:
        # Email may fail on some PDF extractions; do not reject otherwise-complete fresher resumes.
        soft_missing = [m for m in missing if m != "contact email"]
        if not soft_missing:
            return True, []

    if has_education and has_projects_or_certs and len(text.strip()) >= 300 and has_skills:
        return True, []

    is_sufficient = len(missing) <= 1
    return is_sufficient, missing


def validate_document(
    file_bytes: bytes,
    filename: str,
    extracted_text: str,
    file_type_detected: str,
    file_type_expected: str | None,
    extraction_metadata: dict,
) -> DocumentValidationResult:
    """Main validation function that produces complete validation result."""
    
    # Check for empty/corrupted file
    if not file_bytes or len(file_bytes) == 0:
        return DocumentValidationResult(
            state=DocumentValidationState.CORRUPTED_FILE,
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_type_detected=file_type_detected,
            file_type_expected=file_type_expected,
            extraction_quality_score=0.0,
            user_message="The uploaded file appears to be empty.",
            next_action="Please upload a valid file with content.",
            technical_details={"error": "empty_file"},
        )
    
    # Check for type mismatch
    if file_type_expected and file_type_detected != file_type_expected:
        logger.warning(
            "document.type_mismatch",
            filename=filename,
            detected=file_type_detected,
            expected=file_type_expected,
        )
        # Continue anyway - user might have renamed extension
    
    # Check for image files requiring OCR
    if file_type_detected.startswith("image/") or extraction_metadata.get("images_found", 0) > 0:
        if extracted_text.strip():
            # Some text was extracted, continue
            pass
        else:
            return DocumentValidationResult(
                state=DocumentValidationState.RESUME_REQUIRES_OCR,
                document_type=DocumentType.UNKNOWN,
                raw_text="",
                file_type_detected=file_type_detected,
                file_type_expected=file_type_expected,
                extraction_quality_score=0.0,
                user_message="This appears to be an image or scanned document. Text extraction requires OCR (Optical Character Recognition).",
                next_action="Please upload a text-based PDF or Word document, or use a document with selectable text.",
                technical_details={"requires_ocr": True},
            )
    
    # Check extraction quality
    quality_score = calculate_extraction_quality(extracted_text, extraction_metadata)
    
    if quality_score < 0.3:
        return DocumentValidationResult(
            state=DocumentValidationState.LOW_EXTRACTION_QUALITY,
            document_type=DocumentType.UNKNOWN,
            raw_text=extracted_text,
            file_type_detected=file_type_detected,
            file_type_expected=file_type_expected,
            extraction_quality_score=quality_score,
            user_message="The document text could not be extracted clearly. The file may be corrupted, password-protected, or contain mostly images.",
            next_action="Please upload a different version of the document, preferably a text-based PDF or Word file.",
            technical_details={
                "quality_score": quality_score,
                "warnings": extraction_metadata.get("extraction_warnings", []),
            },
        )
    
    # Classify document type
    doc_type, type_confidence = classify_document_type(extracted_text)
    
    # Check if it's NOT a resume
    if doc_type != DocumentType.RESUME:
        type_messages = {
            DocumentType.COVER_LETTER: (
                "This appears to be a cover letter rather than a resume.",
                "Please upload your resume/CV document instead."
            ),
            DocumentType.JOB_DESCRIPTION: (
                "This appears to be a job description rather than a resume.",
                "Please upload your resume/CV document. Job descriptions can be analyzed separately."
            ),
            DocumentType.CERTIFICATE: (
                "This appears to be a certificate rather than a resume.",
                "Please upload your resume/CV document. Certificates can be added as supporting documents."
            ),
            DocumentType.MARKSHEET: (
                "This appears to be a marksheet or transcript rather than a resume.",
                "Please upload your resume/CV document. Marksheets can be added as supporting documents."
            ),
            DocumentType.INVOICE: (
                "This appears to be an invoice or billing document.",
                "Please upload your resume/CV document instead."
            ),
            DocumentType.ID_DOCUMENT: (
                "This appears to be an ID document (passport, license, etc.).",
                "Please upload your resume/CV document instead. Do not share ID documents."
            ),
            DocumentType.PORTFOLIO: (
                "This appears to be a portfolio document.",
                "Please upload your resume/CV document. Portfolio links can be included in your resume."
            ),
            DocumentType.UNKNOWN: (
                "The document type could not be determined.",
                "Please ensure you are uploading a resume/CV document in PDF, DOCX, or TXT format."
            ),
        }
        user_msg, next_act = type_messages.get(doc_type, ("Document type not recognized.", "Please upload a valid resume."))
        
        return DocumentValidationResult(
            state=DocumentValidationState.NOT_A_RESUME,
            document_type=doc_type,
            raw_text=extracted_text,
            file_type_detected=file_type_detected,
            file_type_expected=file_type_expected,
            extraction_quality_score=quality_score,
            user_message=user_msg,
            next_action=next_act,
            technical_details={"document_type_confidence": type_confidence},
        )
    
    # Extract personal info
    normalized_text = _normalize_extraction_text(extracted_text)
    personal_info = extract_personal_info(normalized_text)
    
    # Check sufficiency
    is_sufficient, missing_elements = validate_resume_sufficiency(normalized_text, personal_info)
    
    if not is_sufficient:
        missing_str = ", ".join(missing_elements)
        return DocumentValidationResult(
            state=DocumentValidationState.INSUFFICIENT_INFORMATION,
            document_type=DocumentType.RESUME,
            raw_text=extracted_text,
            file_type_detected=file_type_detected,
            file_type_expected=file_type_expected,
            extraction_quality_score=quality_score,
            user_message=f"The resume appears incomplete. Missing: {missing_str}.",
            next_action="Please upload a more detailed resume with education, skills, and contact information. Formal work experience is optional for student or fresher resumes.",
            technical_details={"missing_elements": missing_elements},
        )
    
    # Success! Valid resume
    exp_years = extract_experience_years(extracted_text)
    edu_level = extract_education_level(extracted_text)
    
    # Import skill extraction for counting
    from app.services.taxonomy import extract_skills
    skills = extract_skills(extracted_text)
    
    # Count experience entries (rough heuristic)
    exp_entries = len(re.findall(
        r"\b(engineer|developer|analyst|manager|scientist|intern|consultant|designer|specialist)\b",
        extracted_text, re.I
    ))
    
    return DocumentValidationResult(
        state=DocumentValidationState.VALID_RESUME,
        document_type=DocumentType.RESUME,
        raw_text=extracted_text,
        file_type_detected=file_type_detected,
        file_type_expected=file_type_expected,
        extraction_quality_score=quality_score,
        user_message="Resume successfully processed! We've extracted your information and will use it to find relevant job opportunities.",
        next_action="Continue to view personalized job recommendations.",
        technical_details={
            "quality_score": quality_score,
            "type_confidence": type_confidence,
        },
        detected_name=personal_info.get("name"),
        detected_email=personal_info.get("email"),
        detected_phone=personal_info.get("phone"),
        detected_location=personal_info.get("location"),
        detected_total_experience_years=exp_years,
        detected_education_level=edu_level,
        detected_skills_count=len(skills),
        detected_experience_entries=exp_entries,
    )


async def process_uploaded_document(
    file_bytes: bytes,
    filename: str,
) -> DocumentValidationResult:
    """Main entry point for document processing pipeline.
    
    This orchestrates the full validation flow:
    1. Detect actual file type from content
    2. Extract text using appropriate parser
    3. Validate extraction quality
    4. Classify document type
    5. Check resume sufficiency
    6. Extract metadata
    
    Returns a complete validation result with human-readable messages.
    """
    # Step 1: Detect file type
    file_type_detected, file_type_expected = detect_file_type_by_content(file_bytes, filename)
    
    logger.info(
        "document.detected",
        filename=filename,
        detected=file_type_detected,
        expected=file_type_expected,
    )
    
    # Check for unsupported formats early
    supported_types = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/html",
        "application/rtf",
        "application/vnd.oasis.opendocument.text",
    ]
    image_types = ["image/jpeg", "image/png", "image/gif", "image/webp", "image/tiff"]
    
    if not (file_type_detected in supported_types or file_type_detected.startswith("image/")):
        return DocumentValidationResult(
            state=DocumentValidationState.UNSUPPORTED_DOCUMENT,
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_type_detected=file_type_detected,
            file_type_expected=file_type_expected,
            extraction_quality_score=0.0,
            user_message=f"This file format ({file_type_detected}) is not supported.",
            next_action="Please upload a PDF, DOCX, TXT, RTF, ODT, or HTML document. Image files require OCR processing.",
            technical_details={"supported_formats": supported_types + image_types},
        )
    
    # Step 2: Extract text
    extracted_text, extraction_metadata = extract_text_from_bytes(
        file_bytes, file_type_detected, filename
    )
    
    logger.info(
        "document.extracted",
        filename=filename,
        parser=extraction_metadata.get("parser_used"),
        pages=extraction_metadata.get("pages_extracted", 0),
        text_length=len(extracted_text),
    )
    
    # Step 3-6: Validate and classify
    result = validate_document(
        file_bytes=file_bytes,
        filename=filename,
        extracted_text=extracted_text,
        file_type_detected=file_type_detected,
        file_type_expected=file_type_expected,
        extraction_metadata=extraction_metadata,
    )
    
    logger.info(
        "document.validated",
        filename=filename,
        state=result.state.value,
        document_type=result.document_type.value,
        quality_score=result.extraction_quality_score,
    )
    
    return result
