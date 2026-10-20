"""Agent profiles: one sub-package per benchmark (today only ssbv2_agent, the SpreadsheetBench v2 grader).

Deliberately imports nothing. This file used to import every profile eagerly, so importing one module of one
profile (the pool machine's grade self-test imports src.agent_profiles.ssbv2_agent.dataset and .scorer) loaded
every other profile and, through them, the SDK harnesses and claude_agent_sdk. Callers import the sub-module they
need by its full path.
"""
