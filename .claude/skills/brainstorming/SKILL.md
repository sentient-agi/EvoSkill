---
name: brainstorming
description: IMMEDIATELY USE THIS SKILL when answering data analysis questions from treasury_bulletins_parsed - internal design thinking that identifies the question type, selects applicable skills, and converges on the best analytical approach before implementation
---

# Internal Design Thinking for Treasury Data Analysis

## Overview

Structured self-dialogue before answering treasury/fiscal data questions. Analyze the question, identify which skills apply, reason through the approach, then execute.

**Core principle:** Think first, execute second. Map the question to the right skill chain before acting.

## The Process

### Phase 1: Question Classification

Analyze the question to determine:

- **Data type**: Interest on debt, budget outlays, receipts, yields, international capital, savings bonds, securities ownership
- **Time scope**: Single point, time series, year-over-year comparison
- **Calculation type**: Raw extraction, inflation-adjusted, ratio, regression, risk metric (ES/VaR), forecast
- **Output format**: Percentage, absolute value, [slope, intercept], ratio
- **Calculation definition**: If the question uses a specific term (e.g., "mid-point normalized difference", "absolute difference", "ARIMA forecast"), explicitly state the mathematical formula/definition BEFORE proceeding

### Phase 2: Skill Selection

Review available skills and determine which apply:

| Question Pattern | Required Skills |
|-----------------|-----------------|
| Any Treasury/fiscal data lookup | `data-extraction-verification` (ALWAYS) |
| Inflation adjustment, regression, trend | `data-extraction-verification` |
| ES, VaR, forecasting, currency conversion | `data-extraction-verification` |
| Final numeric answer | `data-extraction-verification` |

**Skill chain reasoning:**
- "This question requires extracting [X] from Treasury data, applying data-extraction-verification protocol"
- "For calculations, verify each intermediate value before proceeding"

### Phase 3: Approach Design

For the selected skills, map out the execution path:

1. **Data retrieval**: Which treasury_bulletins_parsed files? What grep patterns?
2. **Transformations**: Inflation adjustment? Return calculation? Currency conversion?
3. **Analysis**: Regression? Risk metric? Ratio calculation?
4. **Formula verification**:
   - Write out the exact formula to be used
   - Verify it matches the standard definition of the term in the question
   - Common formulas:
     - Mid-point normalized difference: `(V2 - V1) / ((V1 + V2) / 2)` → result is a dimensionless ratio
     - Absolute difference: `|A - B|` → result has same units as inputs
     - Percentage change: `(new - old) / old × 100` → result is percentage
     - ARIMA forecasts: specify order (p,d,q) and verify it matches question requirements
5. **Validation**: Apply data-extraction-verification at each extraction step
6. **Output**: What format does the question expect?

State assumptions explicitly:
- "Assuming fiscal year convention from Treasury Bulletins..."
- "Using CPI base period of [X] because..."
- "Interpreting 'rate' as percentage without symbol..."

### Phase 4: Design Summary

Before execution, articulate the plan:

```
## Analysis Plan

**Question type:** [classification]

**Skills to apply:**
1. data-extraction-verification → verify all extracted values

**Data sources:** treasury_bulletins_parsed/[files]

**Key steps:**
1. [step]
2. [step]
...

**Assumptions:** [list any]

**Proceeding with analysis.**
```

### Phase 5: Execute

- Apply `data-extraction-verification` protocol for all data extraction
- Verify each intermediate value before proceeding to calculations
- Complete pre-output verification before stating final answer

### Phase 6: Pre-Output Verification (MANDATORY)

Before outputting, check:
1. Units/dimensions correct for question type?
2. Value in reasonable magnitude range?
3. Formula actually computed (not just numerator)?
4. Answer addresses exactly what was asked?

If ANY check fails → re-verify inputs and calculation.

## Remember

- This is internal reasoning - no questions, no waiting
- ALWAYS apply data-extraction-verification when extracting values from tables
- State assumptions so errors are traceable
- The goal is selecting the right approach, not documenting
- Once the plan is clear, execute immediately
