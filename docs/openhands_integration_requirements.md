# OpenHands Requirement For EvoSkill

For EvoSkill, the main remaining OpenHands requirement is native final structured output.

## Why EvoSkill Needs This

EvoSkill does not just need event logs. It needs one final payload that can be parsed and validated against a schema.

Examples:

- `AgentResponse`
- `SkillProposerResponse`
- `ToolGeneratorResponse`
- `PromptProposerResponse`
- `PromptGeneratorResponse`

## What Event JSON Solves

OpenHands event JSON is useful for:

- tracing the run
- logging tool calls
- debugging failures
- persisting execution history

But event JSON is not the same as a guaranteed final schema-valid result.

## What EvoSkill Actually Needs

One of these needs to be true:

- the SDK or headless mode can accept a JSON schema and return a validated final payload
- or OpenHands has an officially supported strict-final-JSON contract that is reliable enough for automated parsing and validation

The final output should be suitable for:

- `json.loads(...)`
- `response_model.model_validate(...)`

## Questions For OpenHands

1. Can a run be constrained to an exact final JSON schema?
2. If yes, where is that final structured payload returned in SDK and headless mode?
3. If the model violates the schema, what is the failure behavior?
4. If native schema enforcement does not exist, what is the recommended strict-JSON pattern for automated systems like EvoSkill?

## Bottom Line

If OpenHands can guarantee final schema-valid output, EvoSkill integration becomes straightforward.

If not, EvoSkill would need a prompt-based JSON extraction layer, which is more brittle for proposer and generator flows.
