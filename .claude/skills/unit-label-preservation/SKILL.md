---
name: unit-label-preservation
description: Enforces appending the correct unit or label to numeric answers when the problem context implies one. Use this skill whenever producing a final numeric answer—e.g. after arithmetic, counting, measurement, or currency calculations—to check whether the original question mentions a countable noun or unit word (apples, dollars, meters, items, etc.) and, if so, append it to the result. A bare number is never acceptable when the question supplies a unit/label. Triggered by any question that combines a numeric answer with concrete nouns or measurement units.
---

# Unit Label Preservation

Attach the correct unit or label to every numeric answer when the question context implies one.

## Rule

1. Scan the original question for countable nouns or unit words (e.g. apples, dollars, meters, items, km/h, kg).
2. If found, append the unit/label to the numeric result with a space separator.
3. If no unit is present in the question, output the bare number.

## Format

| Situation | Output format |
|-----------|--------------|
| Unit/label found in question | `<number> <unit>` — e.g. `8 apples`, `3 meters` |
| Currency symbol present | prefix symbol — e.g. `$12.50` |
| No unit in question | bare number — e.g. `4` |

## Examples

| Question | Correct answer |
|----------|---------------|
| "If I have 3 apples and add 5 more, how many do I have?" | `8 apples` |
| "How many dollars did she spend if she bought 3 items at $4 each?" | `12 dollars` |
| "A car travels 60 km/h for 2 hours. How far does it go?" | `120 km` |
| "2 + 2 = ?" | `4` |

## Common Mistakes to Avoid

- Outputting `8` when the question clearly asks about apples → always `8 apples`.
- Dropping the unit from the label when rephrasing: "The answer is 8" → wrong; "The answer is 8 apples" → correct.
- Adding a unit when none appears in the question — do not invent units.
