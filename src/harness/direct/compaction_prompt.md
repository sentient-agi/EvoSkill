You are approaching your context window limit. Consolidate everything informational from this session into a single structured summary message. Discard filler, repeated searches, dead ends, and verbose tool outputs — keep only the useful findings.

This summary, along with the first user message in this session, will be the ONLY artifacts provided to a fresh instance of yourself (same system prompt). That instance will pick up from where you leave off and continue the analysis.

Your summary MUST include:
1. **Questions analyzed**: which questions you examined, their GT, pass/fail status
2. **Failure diagnosis**: what the solver did wrong (specific turns, search patterns, mistakes)
3. **Key evidence found**: important facts, document IDs, trace excerpts worth keeping
4. **Skill rules planned**: the exact rules you intend to write and why
5. **Files created/modified**: paths and what you wrote
6. **Remaining work**: what the next instance needs to do

Only summarize what you currently know. Do NOT call any tools to gather more information. Do NOT try to read any new files--if you think a specific file is relevant, prompt the fresh instance to read it in your summary. 
Take you time to think through before answering. Respond with a single text summary that begins with "# Summary of Session".