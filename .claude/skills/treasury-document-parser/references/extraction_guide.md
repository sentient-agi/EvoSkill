# Treasury Document Extraction Guide

## Overview

Historical Treasury Bulletin documents from FRASER archives present unique extraction challenges. This guide covers common issues and best practices.

## Document Types

### Native PDF (Text-Selectable)
- Modern digitized documents with embedded text
- Use default extraction (pdfplumber)
- Typically high confidence (>0.8)

### Scanned/Image-Based PDF
- Historical documents photographed/scanned
- Require OCR extraction (`--prefer-ocr`)
- Lower confidence, requires verification

## Common Tables in Treasury Bulletins

### Vessel Statistics Tables
Typical headers:
- "Fiscal year"
- "American vessels" / "Foreign vessels"
- "Entered" / "Cleared"
- "Number" / "Net tonnage"

**Disambiguation Note**: Tables may contain both "vessels entered" AND "vessels cleared" - ensure you extract the correct series:
- "Entered": Vessels arriving at US ports
- "Cleared": Vessels departing from US ports

### Trade Statistics
- "Imports" / "Exports"
- "Country of origin" / "Country of destination"
- "Value" (typically in dollars or thousands of dollars)

## Confidence Score Interpretation

| Score Range | Interpretation | Recommended Action |
|-------------|----------------|-------------------|
| 0.9 - 1.0   | High confidence | Use directly |
| 0.7 - 0.9   | Good confidence | Spot-check recommended |
| 0.5 - 0.7   | Moderate confidence | Manual verification recommended |
| 0.3 - 0.5   | Low confidence | Cross-reference with source |
| < 0.3       | Very low confidence | Do not use without verification |

## Common OCR Errors

### Character Substitutions
| Intended | OCR Result | Example |
|----------|------------|---------|
| 0 (zero) | O (letter) | 1,O00 -> 1,000 |
| 1 (one)  | l (ell) or I | l,234 -> 1,234 |
| 8        | 3 or B | 2,3OO -> 2,800 |
| 5        | S | S,432 -> 5,432 |

### Number Grouping Issues
- Thousands separators may be OCR'd as periods: `1.234` vs `1,234`
- Spaces inserted in numbers: `12 345` vs `12345`
- Decimal points confused with commas

## Extraction Workflow

### 1. Initial Search
```bash
# Find where your data appears
python treasury_document_parser.py --pdf bulletin.pdf --search "American vessels"
```

### 2. Targeted Extraction
```bash
# Extract specific page with expected headers
python treasury_document_parser.py --pdf bulletin.pdf \
    --page 152 \
    --headers "Fiscal year,American vessels,Tonnage"
```

### 3. Validation
```bash
# Verify extraction against source
python treasury_document_parser.py --validate extraction.json \
    --source bulletin.pdf --show-context
```

### 4. Cross-Reference
When possible, cross-reference extracted values with:
- Different tables in the same document
- Other years' bulletins
- Published summaries or indexes

## Handling Multi-Column Tables

Historical documents often have complex multi-column layouts:

```
                    Vessels Entered          |       Vessels Cleared
Fiscal Year    American    Foreign     |    American    Foreign
                No.   Tons   No.  Tons  |     No.  Tons   No.  Tons
```

**Extraction Tips**:
1. Specify exact column headers: `--headers "Fiscal year,American No.,American Tons,Foreign No.,Foreign Tons"`
2. Extract full table and filter programmatically
3. Use `--show-context` to verify column alignment

## Year Disambiguation

Fiscal years in Treasury documents:
- FY runs October 1 - September 30
- "1940" in a Treasury Bulletin might mean:
  - Fiscal Year 1940 (Oct 1939 - Sep 1940)
  - Calendar Year 1940
  - The year the bulletin was published

**Always check document context for fiscal year conventions.**

## Data Units

Common units in historical Treasury documents:

| Description | Typical Unit |
|-------------|-------------|
| Vessel tonnage | Net tons |
| Trade value | Dollars (not thousands) |
| Tariff revenue | Dollars |
| Percentages | Whole numbers (not decimals) |

**Warning**: Some tables show values in thousands - check table headers and footnotes.

## Footnotes and Annotations

Historical tables often include:
- `*` Revised figure
- `p` Preliminary
- `e` Estimated
- `...` Data not available
- `-` Zero or negligible

These are captured in the raw extraction but may affect numeric parsing confidence.
