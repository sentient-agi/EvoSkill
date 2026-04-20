---
name: brainstorming
description: IMMEDIATELY USE THIS SKILL before answering any task - internal design thinking that identifies the question type, selects applicable skills, and converges on the best approach before implementation
---

# Internal Design Thinking

## Overview

Structured self-dialogue before answering any question or task. Analyze what's being asked, identify which skills apply, reason through the approach, then execute.

**Core principle:** Think first, execute second. Map the question to the right skill chain before acting.

## The Process

### Phase 1: Question Classification

Analyze the question to determine:

- **Domain**: What subject area does this fall into?
- **Data type**: What kind of data or input is involved?
- **Time scope**: Single point, time series, comparison, or not applicable?
- **Operation type**: Raw lookup, calculation, transformation, generation, analysis?
- **Output format**: What form should the answer take (number, text, code, list, etc.)?
- **Key terms**: If the question uses a specific term or method, explicitly state its definition/formula BEFORE proceeding

### Phase 2: Skill Selection

Review available skills and determine which apply:

1. List all skills currently available in `.claude/skills/`
2. For each skill, check if it is relevant to the question's domain and operation type
3. If multiple skills apply, determine the order they should be chained

**Skill chain reasoning:**
- "This question requires [X], which maps to skill [Y]"
- "After [Y], I need to apply [Z] for the final transformation"
- "No existing skill covers [W] — proceed with general reasoning"

### Phase 3: Approach Design

For the selected skills, map out the execution path:

1. **Data retrieval**: Where does the relevant data live? What search patterns to use?
2. **Transformations**: What processing or conversions are needed?
3. **Analysis**: What computation, reasoning, or synthesis is required?
4. **Formula verification** (if applicable):
   - Write out the exact formula to be used
   - Verify it matches the standard definition of the term in the question
   - Confirm units and dimensions are consistent throughout
5. **Validation**: How will you verify intermediate results are correct?
6. **Output**: What format does the question expect?

State assumptions explicitly:
- "Assuming [X] convention because..."
- "Interpreting [term] as [definition] because..."
- "Using [method] since the question specifies..."

### Phase 4: Design Summary

Before execution, articulate the plan:

```
## Analysis Plan

**Question type:** [classification]

**Skills to apply:**
1. [skill name] → [what it does for this question]
2. [skill name] → [what it does for this question]

**Data sources:** [where to find the relevant data]

**Key steps:**
1. [step]
2. [step]
...

**Precision note** (if applicable): [number of decimal places, rounding rules, etc.]

**Assumptions:** [list any]

**Proceeding with execution.**
```

### Phase 5: Execute

- Follow the plan from Phase 4
- Verify each intermediate result before proceeding to the next step
- Complete pre-output verification before stating final answer

### Phase 6: Pre-Output Verification (MANDATORY)

Before outputting, check:
1. Units/dimensions correct for the question type?
2. Value in reasonable range for the domain?
3. Computation fully completed (not partial)?
4. Answer addresses exactly what was asked?

If ANY check fails → re-verify inputs and reasoning.

## Remember

- This is internal reasoning — no questions, no waiting
- ALWAYS verify data extracted from files or tables before using it in calculations
- State assumptions so errors are traceable
- The goal is selecting the right approach, not documenting
- Once the plan is clear, execute immediately