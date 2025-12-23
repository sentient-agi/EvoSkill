#!/usr/bin/env python3
"""
Treasury Document Parser - Extract and validate tabular data from historical Treasury documents

Parses historical Treasury Bulletin documents (especially from FRASER archives) to extract
tabular data with confidence scoring and source context verification.

Usage:
    python treasury_document_parser.py --pdf document.pdf --table-name "Vessels Entered"
    python treasury_document_parser.py --pdf document.pdf --headers "Year,American,Foreign" --page 5
    python treasury_document_parser.py --pdf document.pdf --extract-all --output tables.json
    python treasury_document_parser.py --validate results.json --source document.pdf

Examples:
    # Extract specific table with known headers
    python treasury_document_parser.py --pdf treasury_bulletin_1940.pdf \
        --headers "Fiscal year,Vessels entered,Tonnage" \
        --page 152

    # Extract all tables and get confidence scores
    python treasury_document_parser.py --pdf treasury_bulletin_1940.pdf \
        --extract-all --min-confidence 0.7

    # Validate extracted values against source
    python treasury_document_parser.py --validate extraction.json \
        --source treasury_bulletin_1940.pdf --show-context

    # Search for specific data pattern
    python treasury_document_parser.py --pdf treasury_bulletin_1940.pdf \
        --search "American vessels" --year-range 1930-1940

Output:
    JSON with extracted tables, confidence scores, and source context for verification.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Attempt imports for PDF processing
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class ExtractedValue:
    """A single extracted value with metadata."""
    value: str
    numeric_value: Optional[float]
    row_index: int
    col_index: int
    column_header: str
    confidence: float
    source_context: str
    page_number: int
    bounding_box: Optional[tuple] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedTable:
    """A complete extracted table with metadata."""
    table_id: str
    page_number: int
    table_title: Optional[str]
    headers: list[str]
    rows: list[list[str]]
    confidence_scores: list[list[float]]
    average_confidence: float
    source_context: str
    extraction_method: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_numeric(value: str) -> tuple[Optional[float], float]:
    """
    Parse a numeric value from text, handling common formats in historical documents.

    Returns:
        Tuple of (numeric_value, confidence_score)
    """
    if not value or value.strip() in ('', '-', '...', 'n.a.', 'N.A.'):
        return None, 0.9  # High confidence it's intentionally blank

    original = value.strip()
    cleaned = original

    # Remove common OCR artifacts
    cleaned = cleaned.replace('O', '0').replace('o', '0')
    cleaned = cleaned.replace('l', '1').replace('I', '1')
    cleaned = cleaned.replace(',', '')
    cleaned = cleaned.replace(' ', '')

    # Handle parentheses as negative
    is_negative = False
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = cleaned[1:-1]
        is_negative = True

    # Handle footnote markers
    cleaned = re.sub(r'[*†‡§\d]$', '', cleaned)

    confidence = 1.0

    # Penalize confidence for cleaned values
    if original != cleaned:
        confidence *= 0.9

    # Try to parse as float
    try:
        # Handle percentages
        if '%' in original:
            cleaned = cleaned.replace('%', '')
            num = float(cleaned)
            if is_negative:
                num = -num
            confidence *= 0.95  # Slight penalty for percentage conversion
            return num, confidence

        num = float(cleaned)
        if is_negative:
            num = -num
        return num, confidence

    except ValueError:
        # Try extracting just digits
        digits_only = re.sub(r'[^\d.-]', '', cleaned)
        if digits_only:
            try:
                num = float(digits_only)
                if is_negative:
                    num = -num
                return num, 0.5  # Low confidence for extracted digits
            except ValueError:
                pass

        return None, 0.1  # Very low confidence


def compute_header_match_score(found_headers: list[str], expected_headers: list[str]) -> float:
    """
    Compute similarity score between found and expected headers.

    Uses fuzzy matching to handle OCR errors in historical documents.
    """
    if not expected_headers:
        return 1.0  # No expectations means any headers are fine

    if not found_headers:
        return 0.0

    def normalize(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', s.lower())

    matches = 0
    for expected in expected_headers:
        exp_norm = normalize(expected)
        for found in found_headers:
            found_norm = normalize(found)
            # Check for substring match (handles partial OCR)
            if exp_norm in found_norm or found_norm in exp_norm:
                matches += 1
                break
            # Check for edit distance (handles OCR errors)
            if len(exp_norm) > 3 and len(found_norm) > 3:
                if sum(a == b for a, b in zip(exp_norm, found_norm)) / max(len(exp_norm), len(found_norm)) > 0.7:
                    matches += 0.8
                    break

    return matches / len(expected_headers)


def get_surrounding_context(page_text: str, target: str, context_chars: int = 200) -> str:
    """Extract text surrounding a target value for verification."""
    idx = page_text.find(target)
    if idx == -1:
        # Try normalized search
        normalized_text = re.sub(r'\s+', ' ', page_text)
        normalized_target = re.sub(r'\s+', ' ', target)
        idx = normalized_text.find(normalized_target)
        if idx == -1:
            return f"[Context not found for: {target[:50]}...]"

    start = max(0, idx - context_chars)
    end = min(len(page_text), idx + len(target) + context_chars)

    context = page_text[start:end]
    # Clean up for display
    context = re.sub(r'\s+', ' ', context).strip()

    if start > 0:
        context = "..." + context
    if end < len(page_text):
        context = context + "..."

    return context


def extract_tables_pdfplumber(pdf_path: str, page_numbers: Optional[list[int]] = None,
                               expected_headers: Optional[list[str]] = None) -> list[ExtractedTable]:
    """
    Extract tables using pdfplumber (best for native PDF tables).
    """
    if not HAS_PDFPLUMBER:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        pages_to_process = page_numbers if page_numbers else range(len(pdf.pages))

        for page_num in pages_to_process:
            if page_num >= len(pdf.pages):
                continue

            page = pdf.pages[page_num]
            page_text = page.extract_text() or ""
            page_tables = page.extract_tables()

            for table_idx, table_data in enumerate(page_tables):
                if not table_data or len(table_data) < 2:
                    continue

                # First row is typically headers
                headers = [str(h) if h else f"Column_{i}" for i, h in enumerate(table_data[0])]
                rows = [[str(c) if c else "" for c in row] for row in table_data[1:]]

                # Compute confidence scores for each cell
                confidence_matrix = []
                for row in rows:
                    row_confidences = []
                    for cell in row:
                        _, conf = normalize_numeric(cell)
                        row_confidences.append(conf)
                    confidence_matrix.append(row_confidences)

                # Calculate average confidence
                all_confs = [c for row in confidence_matrix for c in row]
                avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

                # Adjust confidence based on header match
                if expected_headers:
                    header_score = compute_header_match_score(headers, expected_headers)
                    avg_conf *= header_score

                # Get context around first data value
                first_value = rows[0][0] if rows and rows[0] else ""
                source_context = get_surrounding_context(page_text, first_value)

                # Detect potential issues
                warnings = []
                if len(set(len(r) for r in rows)) > 1:
                    warnings.append("Inconsistent row lengths detected")
                if avg_conf < 0.5:
                    warnings.append("Low overall confidence - manual verification recommended")
                if any(h.startswith("Column_") for h in headers):
                    warnings.append("Some column headers could not be detected")

                table = ExtractedTable(
                    table_id=f"page{page_num + 1}_table{table_idx + 1}",
                    page_number=page_num + 1,
                    table_title=None,  # Could be enhanced to detect titles
                    headers=headers,
                    rows=rows,
                    confidence_scores=confidence_matrix,
                    average_confidence=round(avg_conf, 3),
                    source_context=source_context,
                    extraction_method="pdfplumber",
                    warnings=warnings
                )
                tables.append(table)

    return tables


def extract_tables_ocr(pdf_path: str, page_numbers: Optional[list[int]] = None,
                       expected_headers: Optional[list[str]] = None) -> list[ExtractedTable]:
    """
    Extract tables using OCR for scanned documents.
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")
    if not HAS_OCR:
        raise ImportError("pytesseract and Pillow are required. Install with: pip install pytesseract pillow")

    tables = []
    doc = fitz.open(pdf_path)

    pages_to_process = page_numbers if page_numbers else range(len(doc))

    for page_num in pages_to_process:
        if page_num >= len(doc):
            continue

        page = doc[page_num]

        # Render page to image
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OCR the image
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Get full text for context
        page_text = pytesseract.image_to_string(img)

        # Group text by lines
        lines = {}
        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                line_num = ocr_data['line_num'][i]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append({
                    'text': text,
                    'left': ocr_data['left'][i],
                    'conf': ocr_data['conf'][i]
                })

        # Sort lines and detect table structure
        sorted_lines = sorted(lines.items())

        # Simple heuristic: consecutive lines with similar column positions form a table
        current_table_rows = []
        current_confidences = []

        for line_num, words in sorted_lines:
            # Sort words by x position
            words = sorted(words, key=lambda w: w['left'])
            row = [w['text'] for w in words]
            confs = [w['conf'] / 100.0 for w in words]  # Normalize to 0-1

            if len(row) >= 2:  # Likely a table row
                current_table_rows.append(row)
                current_confidences.append(confs)

        if len(current_table_rows) >= 3:  # At least header + 2 data rows
            headers = current_table_rows[0]
            rows = current_table_rows[1:]
            confidence_matrix = current_confidences[1:]

            # Normalize row lengths
            max_cols = max(len(r) for r in rows + [headers])
            headers = headers + [f"Column_{i}" for i in range(len(headers), max_cols)]
            rows = [r + [""] * (max_cols - len(r)) for r in rows]
            confidence_matrix = [c + [0.5] * (max_cols - len(c)) for c in confidence_matrix]

            all_confs = [c for row in confidence_matrix for c in row]
            avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

            # Apply header match penalty
            if expected_headers:
                header_score = compute_header_match_score(headers, expected_headers)
                avg_conf *= header_score

            # OCR generally has lower confidence
            avg_conf *= 0.8

            warnings = [
                "Extracted via OCR - manual verification strongly recommended",
                "Column alignment may be imprecise"
            ]

            table = ExtractedTable(
                table_id=f"page{page_num + 1}_ocr_table1",
                page_number=page_num + 1,
                table_title=None,
                headers=headers,
                rows=rows,
                confidence_scores=confidence_matrix,
                average_confidence=round(avg_conf, 3),
                source_context=page_text[:500] + "..." if len(page_text) > 500 else page_text,
                extraction_method="ocr_tesseract",
                warnings=warnings
            )
            tables.append(table)

    doc.close()
    return tables


def extract_tables(pdf_path: str,
                   page_numbers: Optional[list[int]] = None,
                   expected_headers: Optional[list[str]] = None,
                   prefer_ocr: bool = False) -> list[ExtractedTable]:
    """
    Extract tables from a PDF document using the best available method.

    Args:
        pdf_path: Path to PDF file
        page_numbers: Optional list of 0-indexed page numbers to process
        expected_headers: Optional list of expected column headers for validation
        prefer_ocr: Force OCR extraction even if native text is available

    Returns:
        List of extracted tables with confidence scores
    """
    tables = []

    if not prefer_ocr and HAS_PDFPLUMBER:
        tables = extract_tables_pdfplumber(pdf_path, page_numbers, expected_headers)

    # If no tables found or OCR preferred, try OCR
    if (not tables or prefer_ocr) and HAS_OCR and HAS_PYMUPDF:
        ocr_tables = extract_tables_ocr(pdf_path, page_numbers, expected_headers)
        if not tables:
            tables = ocr_tables
        else:
            # Merge, preferring higher confidence
            for ocr_table in ocr_tables:
                if not any(t.page_number == ocr_table.page_number for t in tables):
                    tables.append(ocr_table)

    return tables


def search_for_pattern(pdf_path: str, pattern: str,
                       page_numbers: Optional[list[int]] = None) -> list[dict]:
    """
    Search for a text pattern in the PDF and return matching contexts.

    Useful for locating specific data series like "American vessels entered".
    """
    matches = []

    if HAS_PDFPLUMBER:
        with pdfplumber.open(pdf_path) as pdf:
            pages = page_numbers if page_numbers else range(len(pdf.pages))

            for page_num in pages:
                if page_num >= len(pdf.pages):
                    continue

                page = pdf.pages[page_num]
                text = page.extract_text() or ""

                # Case-insensitive search
                pattern_lower = pattern.lower()
                text_lower = text.lower()

                idx = 0
                while True:
                    found = text_lower.find(pattern_lower, idx)
                    if found == -1:
                        break

                    context = get_surrounding_context(text, text[found:found + len(pattern)], 300)
                    matches.append({
                        "page": page_num + 1,
                        "position": found,
                        "match": text[found:found + len(pattern)],
                        "context": context
                    })
                    idx = found + 1

    return matches


def validate_extraction(extraction_file: str, source_pdf: str,
                        show_context: bool = False) -> dict:
    """
    Validate extracted values against the source PDF.

    Returns validation report with confidence adjustments.
    """
    with open(extraction_file, 'r') as f:
        extraction = json.load(f)

    validation_results = {
        "status": "valid",
        "total_values_checked": 0,
        "values_verified": 0,
        "values_uncertain": 0,
        "values_not_found": 0,
        "details": []
    }

    if HAS_PDFPLUMBER:
        with pdfplumber.open(source_pdf) as pdf:
            for table in extraction.get("tables", [extraction]):
                page_num = table.get("page_number", 1) - 1
                if page_num >= len(pdf.pages):
                    continue

                page = pdf.pages[page_num]
                page_text = page.extract_text() or ""

                for row_idx, row in enumerate(table.get("rows", [])):
                    for col_idx, value in enumerate(row):
                        if not value or value.strip() == "":
                            continue

                        validation_results["total_values_checked"] += 1

                        # Check if value exists in page text
                        if value in page_text:
                            validation_results["values_verified"] += 1
                            status = "verified"
                        elif re.sub(r'\s+', '', value) in re.sub(r'\s+', '', page_text):
                            validation_results["values_verified"] += 1
                            status = "verified_normalized"
                        else:
                            # Try to find similar value
                            numeric, _ = normalize_numeric(value)
                            if numeric:
                                # Search for the numeric representation
                                num_str = str(int(numeric)) if numeric == int(numeric) else str(numeric)
                                if num_str in page_text:
                                    validation_results["values_uncertain"] += 1
                                    status = "uncertain"
                                else:
                                    validation_results["values_not_found"] += 1
                                    status = "not_found"
                            else:
                                validation_results["values_not_found"] += 1
                                status = "not_found"

                        if show_context or status != "verified":
                            context = get_surrounding_context(page_text, value) if status != "not_found" else ""
                            validation_results["details"].append({
                                "row": row_idx,
                                "col": col_idx,
                                "value": value,
                                "status": status,
                                "context": context if show_context else None
                            })

    # Determine overall status
    if validation_results["total_values_checked"] == 0:
        validation_results["status"] = "no_data"
    elif validation_results["values_not_found"] > validation_results["values_verified"]:
        validation_results["status"] = "invalid"
    elif validation_results["values_uncertain"] > 0:
        validation_results["status"] = "needs_review"

    return validation_results


def get_value_with_context(pdf_path: str, page: int, row_header: str,
                            col_header: str) -> Optional[ExtractedValue]:
    """
    Extract a specific value by row and column headers with full context.

    Useful for targeted extraction when you know exactly what you're looking for.
    """
    tables = extract_tables(pdf_path, page_numbers=[page - 1])

    for table in tables:
        # Find column index
        col_idx = None
        for i, h in enumerate(table.headers):
            if col_header.lower() in h.lower() or h.lower() in col_header.lower():
                col_idx = i
                break

        if col_idx is None:
            continue

        # Find row by first column value
        for row_idx, row in enumerate(table.rows):
            if row and (row_header.lower() in row[0].lower() or row[0].lower() in row_header.lower()):
                value = row[col_idx] if col_idx < len(row) else None
                if value:
                    numeric, conf = normalize_numeric(value)
                    return ExtractedValue(
                        value=value,
                        numeric_value=numeric,
                        row_index=row_idx,
                        col_index=col_idx,
                        column_header=table.headers[col_idx],
                        confidence=conf * table.average_confidence,
                        source_context=table.source_context,
                        page_number=table.page_number
                    )

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract and validate tabular data from historical Treasury documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all tables from a document
  python treasury_document_parser.py --pdf treasury_bulletin.pdf --extract-all

  # Extract specific table by headers
  python treasury_document_parser.py --pdf treasury_bulletin.pdf \\
      --headers "Fiscal year,Vessels entered,Tonnage" --page 152

  # Search for specific data pattern
  python treasury_document_parser.py --pdf treasury_bulletin.pdf \\
      --search "American vessels"

  # Validate extracted data against source
  python treasury_document_parser.py --validate results.json \\
      --source treasury_bulletin.pdf --show-context

  # Extract specific value by row/column
  python treasury_document_parser.py --pdf treasury_bulletin.pdf \\
      --row "1939" --col "American vessels" --page 152

Notes:
  - For historical documents, OCR may be required (--prefer-ocr)
  - Confidence scores help identify values needing manual verification
  - Use --show-context to see source text around extracted values
  - Headers are matched fuzzy to handle OCR errors
        """
    )

    # Input options
    parser.add_argument("--pdf", type=str,
                        help="Path to PDF document")
    parser.add_argument("--page", type=int, action="append", dest="pages",
                        help="Specific page number(s) to process (1-indexed)")

    # Extraction options
    parser.add_argument("--extract-all", action="store_true",
                        help="Extract all tables from the document")
    parser.add_argument("--headers", type=str,
                        help="Comma-separated expected column headers")
    parser.add_argument("--table-name", type=str,
                        help="Name/title of table to find")
    parser.add_argument("--prefer-ocr", action="store_true",
                        help="Force OCR extraction for scanned documents")

    # Targeted extraction
    parser.add_argument("--row", type=str,
                        help="Row header/label to find")
    parser.add_argument("--col", type=str,
                        help="Column header to find")

    # Search
    parser.add_argument("--search", type=str,
                        help="Search for text pattern in document")

    # Validation
    parser.add_argument("--validate", type=str,
                        help="Path to extraction JSON file to validate")
    parser.add_argument("--source", type=str,
                        help="Source PDF for validation")
    parser.add_argument("--show-context", action="store_true",
                        help="Show source context for each value")

    # Output options
    parser.add_argument("--output", "-o", type=str,
                        help="Output file path (default: stdout)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence threshold (0-1)")
    parser.add_argument("--compact", action="store_true",
                        help="Compact JSON output")

    args = parser.parse_args()

    try:
        result = None

        if args.validate:
            # Validation mode
            if not args.source:
                parser.error("--source is required with --validate")
            result = validate_extraction(args.validate, args.source, args.show_context)

        elif args.search and args.pdf:
            # Search mode
            page_nums = [p - 1 for p in args.pages] if args.pages else None
            result = {
                "pattern": args.search,
                "matches": search_for_pattern(args.pdf, args.search, page_nums)
            }

        elif args.row and args.col and args.pdf:
            # Targeted extraction
            if not args.pages:
                parser.error("--page is required with --row and --col")
            value = get_value_with_context(args.pdf, args.pages[0], args.row, args.col)
            result = value.to_dict() if value else {"error": "Value not found"}

        elif args.pdf:
            # Table extraction mode
            page_nums = [p - 1 for p in args.pages] if args.pages else None
            expected_headers = args.headers.split(",") if args.headers else None

            tables = extract_tables(
                args.pdf,
                page_numbers=page_nums,
                expected_headers=expected_headers,
                prefer_ocr=args.prefer_ocr
            )

            # Filter by confidence
            if args.min_confidence > 0:
                tables = [t for t in tables if t.average_confidence >= args.min_confidence]

            # Filter by table name if provided
            if args.table_name:
                tables = [t for t in tables if args.table_name.lower() in
                         (t.table_title or "").lower() or
                         args.table_name.lower() in t.source_context.lower()]

            result = {
                "source": args.pdf,
                "tables_found": len(tables),
                "tables": [t.to_dict() for t in tables]
            }

        else:
            parser.error("Either --pdf, --validate, or --search is required")
            return

        # Output
        output_str = json.dumps(result, indent=None if args.compact else 2)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_str)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(output_str)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
