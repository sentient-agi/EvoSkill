---
name: treasury-data-local-first-protocol
description: >
  MANDATORY local-first data retrieval protocol for ANY query involving Treasury, fiscal, budget,
  or financial time-series data. This skill MUST be triggered BEFORE using external sources (FRED,
  forecasts.org, BLS, web search) for: interest on public debt, budget outlays, budget receipts,
  Treasury yields, TIC/international capital data, savings bonds, federal securities ownership,
  or any fiscal/Treasury data. Supersedes all table-specific skills. The agent MUST search
  treasury_bulletins_transformed/ files FIRST and extract values from local authoritative sources
  before considering external data.
---

# Treasury Data Local-First Protocol

**INVIOLABLE RULE: Local Treasury Bulletin files are the authoritative source. Search them FIRST.**

## Mandatory Pre-Check Workflow

For ANY query involving Treasury, fiscal, budget, or financial data, execute these steps IN ORDER:

### Step 1: List Available Local Files

```bash
# Use Glob to find Treasury Bulletins for the relevant time period
Glob pattern: treasury_bulletins_transformed/treasury_bulletin_YYYY_MM.txt
```

Identify files spanning the query's date range. Treasury Bulletins contain historical data across fiscal years.

### Step 2: Search for Relevant Keywords

Use Grep to locate the specific data within local files:

| Query Type | Search Keywords |
|------------|-----------------|
| Interest on debt | `Interest on public debt`, `Treasury Department` |
| Budget outlays | `Budget Outlays by Agencies`, `Net expenditures` |
| Budget receipts | `Budget Receipts`, `Principal Sources` |
| Treasury yields | `Yields`, `long-term bonds`, `Market quotations` |
| International capital | `Capital movements`, `International financial` |
| Savings bonds | `savings bonds`, `savings notes` |
| Federal securities | `Ownership of Federal`, `Public Debt` |

### Step 3: Locate Table and Extract Values

Treasury Bulletins use consistent table naming:
- "Table 3. - Budget Outlays by Agencies" contains interest on public debt
- "Table 2. - Budget Receipts by Principal Sources" contains revenue data
- "Average yields of long-term bonds" section contains yield data

**Read the identified file and locate:**
1. The exact table header line
2. Column headers (often in markdown table format with `|` delimiters)
3. The specific column matching your query
4. Data rows with fiscal year values

**Example extraction from treasury_bulletin_1968_12.txt lines 436-447:**
- Table: "Budget Outlays by Agencies"
- Column: "Treasury Department > Net expenditures Interest on public debt 4/"
- FY 1961 value: 8957 (millions of dollars)
- FY 1968 value: 14585 (millions of dollars)

### Step 4: Only If Local Data Unavailable

Proceed to external sources ONLY after documenting:
```
Local Search Attempted: Yes
Files Searched: [list files]
Keywords Used: [list keywords]
Result: Data not found because [specific reason]
```

## External Source Prohibition

**NEVER use these as a FIRST choice:**
- FRED (Federal Reserve Economic Data)
- forecasts.org
- BLS data aggregators
- Web search results
- Any third-party data portal

**These sources may have:**
- Different series definitions (e.g., FRED interest data ≠ Treasury Bulletin "Interest on public debt")
- Different fiscal year boundaries
- Revised vs. original publication values
- 30%+ discrepancies from authoritative sources

## Cross-Validation Protocol

If external data must be used:

1. Compare against any available local data
2. Document discrepancies > 1%
3. If discrepancy exists, prefer local Treasury Bulletin values
4. Explain the discrepancy in reasoning

## Required Output Format

Every extracted value MUST include:

```
Query: [what data was requested]
Source Priority: Local Treasury Bulletin
File: treasury_bulletin_YYYY_MM.txt
Table: [exact table name from file]
Column: [exact column header]
Line number: [line in file where data found]
Value: [extracted value with units - e.g., "8,957 million dollars"]
Alternative source (if any): [name, value if used]
Discrepancy: [percentage difference if applicable]
```

## Common Data Locations

| Data Type | Table Name | Column Pattern |
|-----------|------------|----------------|
| Interest on public debt | Budget Outlays by Agencies | Treasury Department > Net expenditures Interest on public debt |
| Agency expenditures | Budget Outlays by Agencies | [Agency name] > Net expenditures |
| Tax revenue | Budget Receipts by Principal Sources | [Tax type] > [Fiscal year columns] |
| Computed interest rate | Computed Interest Charge | Computed Interest Rate |
| Maturity distribution | Maturity Distribution | Average Length |

## Fiscal Year Convention

Treasury Bulletins use fiscal year notation:
- "1961." or "1961" = Fiscal Year 1961 (ended June 30, 1961 pre-1977; Sept 30 post-1976)
- Monthly data labeled by calendar date (e.g., "1968-January")
- Values in millions of dollars unless otherwise noted (check header for "In billions")
