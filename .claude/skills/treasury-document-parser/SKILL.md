---
name: treasury-document-parser
description: >
  Parse and extract tabular data from historical Treasury Bulletin documents (especially FRASER archives)
  with confidence scoring and source verification. Use this skill when: (1) Extracting tables from
  historical government PDF documents, (2) Parsing vessel statistics, trade data, or fiscal figures
  from Treasury Bulletins, (3) Verifying extracted numerical values against source documents,
  (4) Working with scanned/OCR documents that require confidence scoring, (5) Cross-referencing
  extracted data between similar tables to detect extraction errors. The skill handles both native
  PDF tables and scanned documents requiring OCR, with built-in validation for historical document
  quirks like ambiguous column headers (e.g., "vessels entered" vs "vessels cleared").
---

# Treasury Document Parser

Extract and validate tabular data from historical Treasury documents with confidence scoring.

## Quick Start

Extract all tables with confidence scores:

```bash
python scripts/treasury_document_parser.py --pdf treasury_bulletin_1940.pdf --extract-all
```

Output:
```json
{
  "source": "treasury_bulletin_1940.pdf",
  "tables_found": 3,
  "tables": [
    {
      "table_id": "page152_table1",
      "page_number": 152,
      "headers": ["Fiscal year", "American vessels", "Foreign vessels"],
      "rows": [["1939", "12,345", "8,234"], ...],
      "average_confidence": 0.87,
      "source_context": "...Table 24.—Vessels entered and cleared...",
      "warnings": []
    }
  ]
}
```

## Common Use Cases

### Extract Specific Table by Headers

When you know the column structure:

```bash
python scripts/treasury_document_parser.py --pdf bulletin.pdf \
    --headers "Fiscal year,American vessels,Tonnage" \
    --page 152
```

### Search for Data Location

Find which page contains your data:

```bash
python scripts/treasury_document_parser.py --pdf bulletin.pdf \
    --search "vessels entered"
```

### Targeted Value Extraction

Get a specific cell with full context:

```bash
python scripts/treasury_document_parser.py --pdf bulletin.pdf \
    --row "1939" --col "American vessels" --page 152
```

Output includes source context for verification:
```json
{
  "value": "12,345",
  "numeric_value": 12345.0,
  "confidence": 0.92,
  "source_context": "...1939     12,345     8,234     5,123..."
}
```

### Validate Extracted Data

Cross-check extraction against source:

```bash
python scripts/treasury_document_parser.py --validate extraction.json \
    --source bulletin.pdf --show-context
```

### OCR for Scanned Documents

Force OCR for image-based PDFs:

```bash
python scripts/treasury_document_parser.py --pdf scanned_bulletin.pdf \
    --prefer-ocr --page 152
```

## Disambiguation Workflow

For tables with similar column names (e.g., "entered" vs "cleared"):

1. **Search first**: Find all occurrences of your target data
   ```bash
   python scripts/treasury_document_parser.py --pdf bulletin.pdf --search "American vessels"
   ```

2. **Check context**: Review `source_context` in results to identify correct table

3. **Extract with headers**: Specify exact headers to avoid confusion
   ```bash
   python scripts/treasury_document_parser.py --pdf bulletin.pdf \
       --headers "Fiscal year,Vessels entered,Net tonnage" --page 152
   ```

4. **Validate**: Verify extraction matches expected values
   ```bash
   python scripts/treasury_document_parser.py --validate results.json \
       --source bulletin.pdf --show-context
   ```

## Confidence Scoring

| Score | Interpretation |
|-------|----------------|
| > 0.9 | High confidence - use directly |
| 0.7-0.9 | Good confidence - spot-check |
| 0.5-0.7 | Moderate - manual verification recommended |
| < 0.5 | Low - cross-reference required |

Filter by confidence:
```bash
python scripts/treasury_document_parser.py --pdf bulletin.pdf \
    --extract-all --min-confidence 0.7
```

## Dependencies

Required:
- `pdfplumber` - Primary table extraction

Optional (for scanned documents):
- `pymupdf` - PDF rendering
- `pytesseract` - OCR
- `pillow` - Image processing

Install all:
```bash
pip install pdfplumber pymupdf pytesseract pillow
```

## Reference

See [references/extraction_guide.md](references/extraction_guide.md) for:
- Common OCR errors in historical documents
- Year and unit disambiguation
- Multi-column table handling
- Footnote interpretation
