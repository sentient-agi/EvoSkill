"""One tau3 rollout with Claude Code as the tau2 agent.

tau2's own text Orchestrator runs the conversation: build_text_orchestrator with the banking_knowledge environment
(retrieval variant alltools, the leaderboard's, see THE RETRIEVAL below), the customer (tau2's user
simulator, or any user registered in tau2's registry) and the agent registered here as "claude_code_agent". That agent
forwards the conversation to ONE long-lived Claude Code process (`claude -p --input-format stream-json --output-format
stream-json --verbose` plus the SSB solver's own flags), and the domain tools reach Claude Code as an in-process MCP
server named "tau2", spoken over the CLI's stdin/stdout CONTROL PROTOCOL. No Agent SDK is used (it would add
`--setting-sources ""` and its own flags).

  usage: tau2_cc_harness.py --job JOB.json      (exit 0: a graded end; 75: an infrastructure end; 2: a bad job)

THE SOLVER: Claude Code 2.1.281, claude-sonnet-5-5, default effort (no --effort), Claude Code's stock
system prompt (nothing appended), the SSB solver's argv and CC_ENV (scripts/cc_solver/cc_solver_lib.py; copied here
because this file runs under the tau3 venv without the EvoSkill repo; tests/test_tau2_cc_harness.py checks the copies
are equal) WITHOUT --max-turns (no step or turn cap), plus ENABLE_TOOL_SEARCH=false so the domain tools
load up front as tau2's agent gets them. Launched either here (`claude_cmd`, an explicit allow-list environment) or,
with job "container", in the SSB solver's container: `evoctl enter --cid cc-... --cwd /evo/work --keep-fd3 --
/usr/bin/env HOME=... <CC_ENV> claude ...` (scripts/daytona/evolver_container_ctl.py runs ARGV with `env -i` and its
own base environment, so claude's variables travel in the argv, never in this process's environment). The Claude
credential is never read here: job "token_fd" names an fd passed through to claude (fd 3 in a container, evoctl's
--keep-fd3) and closed here once claude has started; job "token_kind" says what is on it, a subscription token
("oauth": CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR, the default) or an Anthropic API key ("api_key":
CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR).

THE FIRST MESSAGE: exactly FIRST_MESSAGE_TEMPLATE (tests/test_tau2_cc_harness.py pins its bytes) with {policy} = tau2's banking_knowledge policy for alltools (the domain_policy tau2 hands its
agent: prompts/all_tools.md, i.e. the policy header, the three knowledge base tools KB_search_bm25, KB_search_dense
with tau2's dense instruction "The `KB_search_dense` tool uses **OpenAI API** with embedding model
`text-embedding-3-large` for dense retrieval." and `shell` with components/shell_instructions.md) and {customer} = the
customer's first message; every later customer message goes in verbatim. An optional job field first_message_prefix
(default "": none) is put in front of that first message, in the same stdin user message; recorded (sha256, chars) in harness.json and run.json only when set.

THE RETRIEVAL: tau2's own "alltools" variant, unchanged in name, tools (17: KB_search_bm25,
KB_search_dense, shell and the 14 domain tools) and policy text. Right after tau2 is imported with the data root and the
customer's key is in litellm, run_job calls tau3_retrieval.install(root, key=key) (scripts/tau3_cc/tau3_retrieval.py,
next to this file), which changes three things in THIS process only: KB_search_dense's embeddings go through OpenRouter
(openai/text-embedding-3-large, the same model, provider Azure) with the customer's OpenRouter key, held in memory
(no key: a KB_search_dense call raises inside tau2's tool and the agent gets tau2's tool error, the conversation goes
on); the 698 documents' embeddings come READ-ONLY from <root>/.embeddings_cache (a miss raises: never re-embedded); and
every KB `shell` sandbox (Anthropic's sandbox-runtime srt 0.0.23, run by tau2 in THIS process: on the pool machine,
outside every container) also denies reads of the data root, /opt, /root, /home and /Users. install()'s record (no
secret: the key is a boolean) goes into harness.json and run.json's harness as "retrieval". tau2 finds srt and its
helpers on this process's PATH (Linux: srt, rg, bwrap, socat; macOS: srt, rg); a missing one raises when the
environment is built (an infrastructure end, exit 75), as does a missing or stale embeddings cache. harness.json's
"shell_bins" records that PATH and where each binary was found (shell_bins()), on every end. The key is never
put in an environment, argv, file or log; a tool result is passed through redact() before it goes to claude.

THE BRIDGE (generate_next_message, one thread, synchronous):
  - The first customer message starts claude and sends the control request `initialize` naming sdkMcpServers
    ["tau2"]; it waits for that request's control_response (INIT_TIMEOUT_S, its own bound: no agent time), answering
    the CLI's MCP requests meanwhile. Then the customer's message goes in as a stream-json user message.
  - The harness reads claude's stdout: MCP `initialize`, `notifications/*`, `tools/list` and `ping` are answered at
    once from the agent's tau2 tools (Tool.openai_schema: name, description, parameters as inputSchema). A `tools/call`
    becomes AssistantMessage(tool_calls=[ONE ToolCall]) for tau2: id = the call's _meta "claudecode/toolUseId" (the
    toolu_ id of the tool_use block in Claude Code's transcript), else the oldest unmatched tool_use block of the turn
    with the same name and arguments, else a counter id tau2_call_NNNN (a block that arrives AFTER its call is then
    matched to that call and recorded in cc.late_ids, never handed to a later call); no id is ever used twice. tau2
    executes the call in its environment and hands the ToolMessage back to this agent, whose content (and error flag)
    answers the waiting MCP request, so tau2 records call and result adjacent and paired by id, as its own LLMAgent's.
  - A turn's `result` event ends the agent's message: its non-empty text becomes AssistantMessage(content=text), which
    tau2 routes to the customer. Text Claude writes before a tool call, thinking, and Claude Code's own tools (Skill,
    Bash, Read, ...) never enter tau2's trajectory; they stay in Claude Code's transcript.

THE ONLY LIMIT: 20 minutes of AGENT time (job agent_time_s, 1200). The agent clock sums Claude Code's own
time per turn: from sending the customer's message to the turn's `result` event, its tool calls (tau2's and its own)
included; the customer's time never counts, nor does starting claude. tau2's max_steps is tau2's own default, 200
(MAX_STEPS = tau2.config.DEFAULT_MAX_STEPS: a conversation the customer never ends -- agent and gpt-5-nano customer trading closing pleasantries, with 1-2 s agent turns that leave the
agent clock unspent -- ends at step 200 as tau2's max_steps, scored 0, instead of at the leg's 60-minute hard stop as an
infrastructure re-run), tau2's own wall timeout is off (None), claude gets no --max-turns; tau2's max_errors (10 tool
errors in a row) stays. At the limit the conversation ends as tau2's timeout (scored 0) -- unless Claude Code was retrying
the API at that moment (a system/api_retry event, or a rate_limit_event with status "rejected", not followed by an
assistant event): then it is an infrastructure_error.

THE CUSTOMER: tau2's "user_simulator" on openrouter/openai/gpt-5-nano with EMPTY user llm args (no
temperature, no reasoning effort: tau2's DEFAULT_LLM_ARGS_USER is replaced by {}; tau2 itself still adds the rollout's
seed through set_seed). Its OpenRouter key arrives ONLY on an inherited fd (job "key_fd"), is read once (KEY_READ_S
bound) into litellm.openrouter_key (memory) and the fd is closed: never argv, env, disk, container or grader (an
OPENROUTER_API_KEY / OR_API_KEY / OPENAI_API_KEY in this process's environment is dropped before tau2 loads, recorded
by name). Every customer call is bounded and RETRIED as that same call: each attempt runs on a copy of the customer's
state in a daemon thread with its own timeout (customer_timeout_s, 120), up to customer_attempts (4) attempts, the
wait before attempt k+1 being customer_backoff_s * 3^(k-1) (5, 15, 45 s); an attempt that raises, hangs or returns a
message tau2's validate() refuses is retried (a hung attempt is abandoned: it works on its own copy of the state).
Only when every attempt fails does the rollout end as infrastructure_error. The customer's time is recorded
(harness.json "customer") and never enters the agent clock.

HOW A CONVERSATION ENDS (tau2's termination reason; the harness's exit code):
  - the customer writes ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE###: user_stop (tau2's rule); 0, graded;
  - Claude's reply contains ###STOP###: agent_stop (tau2's convention for agents that may end a conversation, the
    STOP_TOKEN of LLMSoloAgent and the gym agent; is_stop below); 0, graded;
  - tau2's too_many_errors (10): 0, graded (0);
  - the agent clock runs out while Claude Code is not retrying the API: timeout; 0, graded (0);
  - the solver's own ends: Claude ends a turn with no text, calls EndConversation, or its turn ends is_error with
    terminal_reason prompt_too_long / malformed_tool_use(_exhausted) (or an error_max_turns result): agent_error; 0,
    graded (0; tau2 scores every end but user_stop/agent_stop 0);
  - infrastructure_error, 75, never graded: any other is_error result (API error, 429 / quota, error_during_execution,
    ...), the agent clock running out while Claude Code retries the API, claude exiting or its pipe breaking mid-turn,
    no answer to initialize, the tau2 MCP server not connected or its tools never listed before Claude answered, every
    attempt of a customer call failing, SIGTERM / SIGINT to the harness, or anything raised outside the agent. An
    infrastructure end writes NO tau2/simulation.json: its record is tau2/simulation.infrastructure_error.json +
    run.json + final_db.json (when tau2 could finalize), so the user's grader raises (grader_failed) instead of scoring.
  Ending from inside the agent is tau2_bundle.HarnessEnd -> run_conversation (orch._finalize, tau2 at commit 363133a).

THE BUNDLE PATH <workdir>/tau2 is cleared at the start (an old or planted bundle) and again before the harness writes
(planted during the conversation), on every path; an exception escaping run_job and a signal clear it too.

SIGNALS: SIGTERM and SIGINT raise Terminated (a BaseException, so no `except Exception` in tau2 swallows it; further
signals are ignored while cleaning up): claude's process group is killed, <workdir>/tau2 is cleared (or holds only the
infrastructure record), harness.json is written, exit 75.

AFTERWARDS: claude's stdin is closed ("Closing the stream tells the CLI to finish the current turn and exit"; the
2.1.281 stdin schema); after close_grace_s (mid_turn_grace_s when it was mid-turn) the process group is killed. Then the
bundle: tau2_bundle.write_bundle into <workdir>/tau2/ (simulation.json, run.json, final_db.json). out_dir gets
claude.stream.jsonl (every stdout line), claude.stdin.jsonl (every line sent), claude.stderr and harness.json (the
summary: end, agent clock, customer calls, API retries, ids). Claude Code's own transcript is <home>/.claude/projects/
<cwd with every non-alphanumeric character as '-'>/<session_id>.jsonl (CLI 2.1.281's naming); its path is recorded in
run.json harness.cc.transcript. HOME is an explicit job setting ("home", required; a local launch refuses this
process's own HOME). Seed per (task, sample): the first 31 bits of sha256("<domain>/<task id>#s<k>").

PROTOCOL SHAPES, each read from a real source (no model, no network, the binary never executed):
  - claude_agent_sdk 0.2.91 (site-packages/claude_agent_sdk, bundled CLI 2.1.165): _internal/query.py
    initialize request {"subtype":"initialize","hooks":null} and request ids "req_<n>_<hex>" (203-228, 501-528);
    control_request routing, control_cancel_request (247-322); mcp_message handling and the success / error
    control_response shapes (454-499); the MCP answers: initialize {"protocolVersion":"2024-11-05",
    "capabilities":{"tools":{}},"serverInfo":{...}}, tools/list {"tools":[{name, description, inputSchema}]},
    tools/call {"content":[{"type":"text","text":...}], "isError": true when an error}, notifications/initialized
    {"jsonrpc":"2.0","result":{}}, unknown method -32601 (585-721); _internal/transport/subprocess_cli.py --mcp-config
    {"mcpServers":{name:{"type":"sdk","name":name}}} (307-329) and --input-format stream-json (408);
    _internal/client.py the user message {"type":"user","session_id":"","message":{"role":"user","content":...},
    "parent_tool_use_id":null} (209-215).
  - Claude Code 2.1.281 linux-x64 (.cache/claude_code/2.1.281/linux-x64/claude, read with grep -a and dd only): the
    stdin schema's initialize {subtype, hooks?, sdkMcpServers?: string[], ...}; "mcp_message ... SDK-hosted MCP server
    (one named in initialize.sdkMcpServers ...) ... the success response carries the server's JSON-RPC reply under
    mcp_response"; print.ts's initialize handler registers {type:"sdk",name} per sdkMcpServers name and skips a name
    already registered; the MCP tool call passes _meta {"claudecode/toolUseId": <tool_use id>} to callTool;
    "initialize is optional ... Closing the stream tells the CLI to finish the current turn and exit"; the user message
    schema; result subtypes success (is_error, api_error_status, result, terminal_reason, session_id) and
    error_during_execution / error_max_turns / ... (errors[]); the terminal_reason enum @194799743 (blocking_limit,
    rapid_refill_breaker, prompt_too_long, image_error, model_error, api_error, malformed_tool_use_exhausted,
    aborted_streaming, aborted_tools, ..., max_turns, completed); system/api_retry @194942199 {attempt, max_retries,
    retry_delay_ms, error_status (null for connection errors), error, no_response?} "Emitted when an API request fails
    with a retryable error and will be retried after a delay"; rate_limit_event @194896413 with rate_limit_info.status
    allowed / allowed_warning / rejected (@194867590); the system/init schema (cwd, tools, mcp_servers[{name,status,
    source}], ...); END_CONVERSATION_TOOL_NAME "EndConversation"; the project directory name: cwd.replace(
    /[^a-zA-Z0-9]/g,"-"), cut at 200 characters plus a hash; the flag rules "--input-format=stream-json requires
    --print", "... requires output-format=stream-json" and "When using --print, --output-format=stream-json requires
    --verbose" (stream_argv meets all three); the MCP client's protocol versions include "2024-11-05".
  NOT verified end to end: no real claude has spoken this protocol with this harness; tests/fake_claude_cc.py speaks
  the shapes above.

IMPORT: the standard library, tau2_bundle and tau3_retrieval (both standard library at import, both next to this file:
on the machine all bridge files sit in one directory) at import time; tau2 is imported inside run_job, after
PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True and TAU2_DATA_DIR (= the task file's parents[4], the
grader's rule) are set. Runs under the tau3-bench venv (tau2 1.0.1, commit 363133a, with the knowledge extra).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import tau2_bundle as B  # noqa: E402  (standard library at import)
import tau3_retrieval as TR  # noqa: E402  (standard library at import)

NAME = "tau2_cc_harness"
AGENT = "claude_code_agent"
SERVER = "tau2"
TOOL_PREFIX = f"mcp__{SERVER}__"
CC_VERSION = "2.1.281"                       # the pinned Claude Code (recorded; init's claude_code_version is kept)
STOP_TOKEN = "###STOP###"                    # tau2's agent stop token (LLMSoloAgent.STOP_TOKEN, GymAgent)
END_CONVERSATION = "EndConversation"         # Claude Code's END_CONVERSATION_TOOL_NAME (2.1.281)
EXIT_GRADED, EXIT_USAGE, EXIT_INFRA = 0, 2, 75
MCP_PROTOCOL = "2024-11-05"                  # the Agent SDK's in-process server answer (query.py:591)
INIT_TIMEOUT_S = 60.0                        # the SDK's initialize timeout (claude's start: never agent time)
KEY_READ_S = 30.0                            # read_key's bound on the key fd
KEY_ENV = ("OPENROUTER_API_KEY", "OR_API_KEY", "OPENAI_API_KEY")    # litellm would read these; dropped at start
ENV_PASS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "USER", "LOGNAME", "SHELL")  # names, never keys
INFRA_SIMULATION = "simulation.infrastructure_error.json"
MAX_STEPS = 200                              # tau2.config.DEFAULT_MAX_STEPS (checked in run_job)
# is_error results that are the solver's own end (agent_error, scored 0); every other is_error result is infrastructure
SOLVER_TERMINAL_REASONS = ("prompt_too_long", "malformed_tool_use", "malformed_tool_use_exhausted")
TOKEN_FD_IN_CONTAINER = 3                    # evoctl enter --keep-fd3 keeps fd 3 only
# job "token_kind" -> the variable naming the credential's fd for claude (Claude Code reads the fd once at start)
TOKEN_FD_VARS = {"oauth": "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR", "api_key": "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"}
SHELL_BINS = ("srt", "rg", "bwrap", "socat")  # alltools' shell sandbox: tau2 needs all four on Linux, srt + rg on macOS
CID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

# The SSB solver's own flags (scripts/cc_solver/cc_solver_lib.py: CC_DISALLOWED, THINKING_DISPLAY, claude_argv, CC_ENV),
# copied because this file runs without the EvoSkill repo; the tests compare them with cc_solver_lib's.
CC_DISALLOWED = ("WebFetch", "WebSearch", "Task", "Agent", "RemoteTrigger", "SendMessage", "ListAgents", "Workflow",
                 "CronCreate", "CronDelete", "CronList", "ScheduleWakeup")
THINKING_DISPLAY = "summarized"
CC_ENV = {
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_ORG_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_API_SKILL": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_CODE_SKILL": "1",
    "CLAUDE_CODE_DISABLE_POLICY_SKILLS": "1",
    "DISABLE_DOCTOR_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "BASH_DEFAULT_TIMEOUT_MS": "600000",
}
TAU2_ENV = {"ENABLE_TOOL_SEARCH": "false"}  # alltools' 17 tools load up front, as tau2's agent gets them
MCP_CONFIG = json.dumps({"mcpServers": {SERVER: {"type": "sdk", "name": SERVER}}}, separators=(",", ":"))

# ── The solver's first user message, its final newline included (tests/test_tau2_cc_harness.py pins the bytes). The
# greeting it quotes is tau2's DEFAULT_FIRST_AGENT_MESSAGE, the first message of tau2's trajectory.
FIRST_MESSAGE_TEMPLATE = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "\n"
    "<policy>\n"
    "{policy}\n"
    "</policy>\n"
    "\n"
    "You started with \"Hi! How can I help you today?\", and the user replied:\n"
    "\n"
    "{customer}\n"
    "\n"
    "Please continue the conversation. Your output is sent to the user, in full, verbatim; any tool calling turns or "
    "reasoning tokens remain invisible to the user.\n")
GREETING = "Hi! How can I help you today?"   # tau2's DEFAULT_FIRST_AGENT_MESSAGE (orchestrator.py:47-49)
# litellm model-map entries for the GPT-6 customer models, copied from litellm main @0fce5bcc
# (model_prices_and_context_window.json, committed 2026-10-03T06:05:55Z). tau2-bench pins litellm 1.81.11 (uv.lock;
# pyproject <1.82.7) and this bridge reads its BUNDLED map (LITELLM_LOCAL_MODEL_COST_MAP: the machine cannot fetch the
# remote one), which predates GPT-6: reasoning_effort was silently dropped for gpt-6-luna/sol (tau2 sets drop_params)
# and the customer's cost read 0. Registered in run_job before tau2 builds anything.
LITELLM_GPT6_ENTRIES = json.loads(r'''{"gpt-6-luna":{"cache_creation_input_token_cost":1.25e-07,"cache_creation_input_token_cost_above_272k_tokens":2.5e-07,"cache_creation_input_token_cost_above_272k_tokens_batches":1.25e-07,"cache_creation_input_token_cost_above_272k_tokens_flex":1.25e-07,"cache_creation_input_token_cost_above_272k_tokens_priority":5e-07,"cache_creation_input_token_cost_batches":6.25e-08,"cache_creation_input_token_cost_flex":6.25e-08,"cache_creation_input_token_cost_priority":2.5e-07,"cache_read_input_token_cost":1e-08,"cache_read_input_token_cost_above_272k_tokens":2e-08,"cache_read_input_token_cost_above_272k_tokens_batches":1e-08,"cache_read_input_token_cost_above_272k_tokens_flex":1e-08,"cache_read_input_token_cost_above_272k_tokens_priority":4e-08,"cache_read_input_token_cost_batches":5e-09,"cache_read_input_token_cost_flex":5e-09,"cache_read_input_token_cost_priority":2e-08,"input_cost_per_token":1e-07,"input_cost_per_token_above_272k_tokens":2e-07,"input_cost_per_token_above_272k_tokens_batches":1e-07,"input_cost_per_token_above_272k_tokens_flex":1e-07,"input_cost_per_token_above_272k_tokens_priority":4e-07,"input_cost_per_token_batches":5e-08,"input_cost_per_token_flex":5e-08,"input_cost_per_token_priority":2e-07,"litellm_provider":"openai","max_input_tokens":922000,"max_output_tokens":128000,"max_tokens":128000,"mode":"chat","output_cost_per_token":5e-07,"output_cost_per_token_above_272k_tokens":7.5e-07,"output_cost_per_token_above_272k_tokens_batches":3.75e-07,"output_cost_per_token_above_272k_tokens_flex":3.75e-07,"output_cost_per_token_above_272k_tokens_priority":1.5e-06,"output_cost_per_token_batches":2.5e-07,"output_cost_per_token_flex":2.5e-07,"output_cost_per_token_priority":1e-06,"regional_processing_uplift_multiplier_eu":1.1,"regional_processing_uplift_multiplier_us":1.1,"search_context_cost_per_query":{"search_context_size_high":0.01,"search_context_size_low":0.01,"search_context_size_medium":0.01},"source":"https://developers.openai.com/api/docs/pricing","supported_endpoints":["/v1/chat/completions","/v1/batch","/v1/responses"],"supported_modalities":["text","image"],"supported_output_modalities":["text"],"supports_computer_use":true,"supports_function_calling":true,"supports_max_reasoning_effort":true,"supports_minimal_reasoning_effort":false,"supports_native_streaming":true,"supports_none_reasoning_effort":true,"supports_parallel_function_calling":true,"supports_pdf_input":true,"supports_prompt_cache_breakpoint":true,"supports_prompt_caching":true,"supports_reasoning":true,"supports_response_schema":true,"supports_system_messages":true,"supports_tool_choice":true,"supports_vision":true,"supports_web_search":true,"supports_xhigh_reasoning_effort":true},"gpt-6-sol":{"cache_creation_input_token_cost":2.5e-06,"cache_creation_input_token_cost_above_272k_tokens":5e-06,"cache_creation_input_token_cost_above_272k_tokens_batches":2.5e-06,"cache_creation_input_token_cost_above_272k_tokens_flex":2.5e-06,"cache_creation_input_token_cost_above_272k_tokens_priority":1e-05,"cache_creation_input_token_cost_batches":1.25e-06,"cache_creation_input_token_cost_flex":1.25e-06,"cache_creation_input_token_cost_priority":5e-06,"cache_read_input_token_cost":2e-07,"cache_read_input_token_cost_above_272k_tokens":4e-07,"cache_read_input_token_cost_above_272k_tokens_batches":2e-07,"cache_read_input_token_cost_above_272k_tokens_flex":2e-07,"cache_read_input_token_cost_above_272k_tokens_priority":8e-07,"cache_read_input_token_cost_batches":1e-07,"cache_read_input_token_cost_flex":1e-07,"cache_read_input_token_cost_priority":4e-07,"input_cost_per_token":2e-06,"input_cost_per_token_above_272k_tokens":4e-06,"input_cost_per_token_above_272k_tokens_batches":2e-06,"input_cost_per_token_above_272k_tokens_flex":2e-06,"input_cost_per_token_above_272k_tokens_priority":8e-06,"input_cost_per_token_batches":1e-06,"input_cost_per_token_flex":1e-06,"input_cost_per_token_priority":4e-06,"litellm_provider":"openai","max_input_tokens":922000,"max_output_tokens":128000,"max_tokens":128000,"mode":"chat","output_cost_per_token":1e-05,"output_cost_per_token_above_272k_tokens":1.5e-05,"output_cost_per_token_above_272k_tokens_batches":7.5e-06,"output_cost_per_token_above_272k_tokens_flex":7.5e-06,"output_cost_per_token_above_272k_tokens_priority":3e-05,"output_cost_per_token_batches":5e-06,"output_cost_per_token_flex":5e-06,"output_cost_per_token_priority":2e-05,"regional_processing_uplift_multiplier_eu":1.1,"regional_processing_uplift_multiplier_us":1.1,"search_context_cost_per_query":{"search_context_size_high":0.01,"search_context_size_low":0.01,"search_context_size_medium":0.01},"source":"https://developers.openai.com/api/docs/pricing","supported_endpoints":["/v1/chat/completions","/v1/batch","/v1/responses"],"supported_modalities":["text","image"],"supported_output_modalities":["text"],"supports_computer_use":true,"supports_function_calling":true,"supports_max_reasoning_effort":true,"supports_minimal_reasoning_effort":false,"supports_native_streaming":true,"supports_none_reasoning_effort":true,"supports_parallel_function_calling":true,"supports_pdf_input":true,"supports_prompt_cache_breakpoint":true,"supports_prompt_caching":true,"supports_reasoning":true,"supports_response_schema":true,"supports_system_messages":true,"supports_tool_choice":true,"supports_vision":true,"supports_web_search":true,"supports_xhigh_reasoning_effort":true},"openrouter/openai/gpt-6-luna":{"cache_creation_input_token_cost":1.25e-07,"cache_creation_input_token_cost_above_272k_tokens":2.5e-07,"cache_read_input_token_cost":1e-08,"cache_read_input_token_cost_above_272k_tokens":2e-08,"input_cost_per_token":1e-07,"input_cost_per_token_above_272k_tokens":2e-07,"litellm_provider":"openrouter","max_input_tokens":1050000,"max_output_tokens":128000,"max_tokens":128000,"mode":"chat","output_cost_per_token":5e-07,"output_cost_per_token_above_272k_tokens":7.5e-07,"source":"https://openrouter.ai/api/v1/models","supports_audio_input":false,"supports_function_calling":true,"supports_pdf_input":true,"supports_prompt_caching":true,"supports_reasoning":true,"supports_response_schema":true,"supports_tool_choice":true,"supports_vision":true,"supports_web_search":true},"openrouter/openai/gpt-6-sol":{"cache_creation_input_token_cost":2.5e-06,"cache_creation_input_token_cost_above_272k_tokens":5e-06,"cache_read_input_token_cost":2e-07,"cache_read_input_token_cost_above_272k_tokens":4e-07,"input_cost_per_token":2e-06,"input_cost_per_token_above_272k_tokens":4e-06,"litellm_provider":"openrouter","max_input_tokens":1050000,"max_output_tokens":128000,"max_tokens":128000,"mode":"chat","output_cost_per_token":1e-05,"output_cost_per_token_above_272k_tokens":1.5e-05,"source":"https://openrouter.ai/api/v1/models","supports_audio_input":false,"supports_function_calling":true,"supports_pdf_input":true,"supports_prompt_caching":true,"supports_reasoning":true,"supports_response_schema":true,"supports_tool_choice":true,"supports_vision":true,"supports_web_search":true}}''')

JOB_REQUIRED = ("task_file", "workdir", "out_dir", "home")
JOB_DEFAULTS = {
    "task_id": None,                          # the plug-in's id, "<domain>/<task id>"; derived when absent
    "domain": "banking_knowledge",
    "retrieval_variant": TR.VARIANT,          # tau2's alltools, the only variant (others refused)
    "sample": 0,
    "seed": None,                             # None: seed_of(task_id, sample)
    "simulation_id": None,                    # None: "<task_id>#s<sample>"
    "agent_time_s": 1200.0,                   # 20 minutes of agent time, the only limit
    "max_errors": 10,                         # tau2's default, kept
    "user": "user_simulator",                 # a tau2 registry user name
    "user_sim_model": "openrouter/openai/gpt-5-nano",
    "user_llm_args": {},                      # the customer's llm args (tau2 llm_args_user); {} = none
    "customer_timeout_s": 120.0,              # one attempt of one customer call
    "customer_attempts": 4,
    "customer_backoff_s": 5.0,                # the wait before attempt k+1 is customer_backoff_s * 3^(k-1)
    "key_fd": None,                           # the user simulator's OpenRouter key, read once and closed
    "token_fd": None,                         # Claude Code's credential fd, passed through untouched
    "token_kind": "oauth",                    # what token_fd carries: "oauth" (subscription token) or "api_key"
    "model": "claude-sonnet-5-5",
    "claude_cmd": ["claude"],                 # replaces argv[0] of stream_argv(); a test points it at a fake claude
    "container": None,                        # {"evoctl": [argv to evoctl], "cid": "cc-...", "cwd": "/evo/work"}
    "close_grace_s": 30.0,
    "mid_turn_grace_s": 3.0,
    # text sent BEFORE the first message, in the same stdin user message (e.g. a baseline method's preamble); "" = none,
    # the default: the first message is exactly the template
    "first_message_prefix": "",
}


class JobError(ValueError):
    """The job file is unusable (exit 2)."""


class LinkError(RuntimeError):
    """The link to claude failed: an infrastructure end."""


class WallReached(Exception):
    """The deadline passed while waiting for claude."""


class DenseOutage(RuntimeError):
    """KB_search_dense's embedding request failed every try (tau3_retrieval: 3 x 10 s): an infrastructure end, re-run,
    never the solver's 0, as for customer calls: the solver did nothing wrong."""


class CustomerFailed(RuntimeError):
    """Every attempt of one customer call failed: an infrastructure end."""


class Terminated(BaseException):
    """SIGTERM / SIGINT reached the harness (a BaseException: tau2's `except Exception` never swallows it)."""

    def __init__(self, signum: int):
        self.signum = signum
        self.name = signal.Signals(signum).name
        super().__init__(f"the harness received {self.name}")


# ───────────────────────────────────────────────────────────────────────────── pure helpers (no tau2)
def stream_argv(model: str = JOB_DEFAULTS["model"]) -> list:
    """cc_solver_lib.claude_argv without the positional prompt and without --max-turns (no turn cap), plus
    --input-format stream-json and --mcp-config for the in-process server. No --effort (default effort), no
    --append-system-prompt (stock system prompt)."""
    return ["claude", "-p", "--model", model, "--output-format", "stream-json",
            "--verbose", "--permission-mode", "bypassPermissions", "--strict-mcp-config",
            "--disallowedTools", ",".join(CC_DISALLOWED), "--thinking-display", THINKING_DISPLAY,
            "--input-format", "stream-json", "--mcp-config", MCP_CONFIG]


def claude_vars(home, token_fd=None, token_kind: str = "oauth") -> dict:
    """The variables claude itself must see on top of its base environment: HOME, CC_ENV, TAU2_ENV and, with a token
    fd, the fd variable of its kind (TOKEN_FD_VARS: CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR for a subscription token,
    CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR for an API key)."""
    env = {"HOME": str(home)}
    env.update(CC_ENV)
    env.update(TAU2_ENV)
    if token_fd is not None:
        env[TOKEN_FD_VARS[token_kind]] = str(int(token_fd))
    return env


def claude_env(home, token_fd=None, token_kind: str = "oauth") -> dict:
    """A local claude's whole environment: a few harmless names of this process's plus claude_vars. Nothing else of
    this process reaches claude."""
    env = {k: os.environ[k] for k in ENV_PASS if k in os.environ}
    env.setdefault("PATH", "/usr/bin:/bin")
    env.update(claude_vars(home, token_fd, token_kind))
    return env


def container_argv(container: dict, claude_cmd: list, model: str, home, token_fd=None,
                   token_kind: str = "oauth") -> list:
    """claude in the SSB solver's container: evoctl enter (nsenter + setpriv to evo, `env -i` + evoctl's base
    environment, cwd) then /usr/bin/env with claude's own variables, then claude_cmd + the stream argv."""
    pairs = [f"{k}={v}" for k, v in claude_vars(home, token_fd, token_kind).items()]
    return (list(container["evoctl"]) + ["enter", "--cid", container["cid"], "--cwd", container["cwd"]]
            + (["--keep-fd3"] if token_fd is not None else []) + ["--", "/usr/bin/env"] + pairs
            + list(claude_cmd) + stream_argv(model)[1:])


def launch_plan(job: dict) -> dict:
    """How claude starts for this job: argv, the Popen environment, the fds passed, the names claude itself sees."""
    home, token_fd, kind, model = job["home"], job["token_fd"], job["token_kind"], job["model"]
    pass_fds = (token_fd,) if token_fd is not None else ()
    if job["container"] is None:
        env = claude_env(home, token_fd, kind)
        return {"argv": list(job["claude_cmd"]) + stream_argv(model)[1:], "env": env, "pass_fds": pass_fds,
                "claude_env_names": sorted(env), "where": "local"}
    env = {k: os.environ[k] for k in ENV_PASS if k in os.environ}       # evoctl's own environment
    env.setdefault("PATH", "/usr/bin:/bin")
    return {"argv": container_argv(job["container"], job["claude_cmd"], model, home, token_fd, kind), "env": env,
            "pass_fds": pass_fds, "claude_env_names": sorted(claude_vars(home, token_fd, kind)),
            "where": f"container {job['container']['cid']}"}


def run_config(job: dict, seed: int) -> dict:
    """tau2's TextRunConfig for this job: the agent registered here, the customer with the job's user_llm_args (EMPTY
    by default; optional, e.g. {"reasoning": {"effort": "high"}} for an OpenRouter model), tau2's own max_steps (200:
    ends a conversation the customer never ends) and no tau2 wall (the agent clock is the only limit),
    tau2's max_errors."""
    return {"domain": job["domain"], "agent": AGENT, "llm_agent": f"claude-code:{job['model']}", "llm_args_agent": {},
            "user": job["user"], "llm_user": job["user_sim_model"], "llm_args_user": dict(job["user_llm_args"]),
            "max_steps": MAX_STEPS, "max_errors": int(job["max_errors"]), "timeout": None, "seed": seed,
            "retrieval_config": job["retrieval_variant"]}


def shell_bins() -> dict:
    """This process's PATH and where it finds the KB shell's binaries on it (shutil.which, as tau2's SandboxManager
    looks them up): the record of what the agent's `shell` ran on. No secret."""
    return {"PATH": os.environ.get("PATH"), **{b: shutil.which(b) for b in SHELL_BINS}}


def seed_of(task_id: str, sample: int = 0) -> int:
    """The first 31 bits of sha256('<task id>#s<k>')."""
    return int(hashlib.sha256(f"{task_id}#s{int(sample)}".encode()).hexdigest(), 16) >> 225


def first_message(policy: str, customer: str) -> str:
    """FIRST_MESSAGE_TEMPLATE filled in one pass (a brace in the policy or the customer's text is left alone)."""
    vals = {"policy": policy, "customer": customer}
    return re.sub(r"\{(policy|customer)\}", lambda m: vals[m.group(1)], FIRST_MESSAGE_TEMPLATE)


def first_message_sha256() -> str:
    return hashlib.sha256(FIRST_MESSAGE_TEMPLATE.encode("utf-8")).hexdigest()


def project_dir_name(cwd: str):
    """Claude Code's project directory name for a cwd (2.1.281: every non-alphanumeric character becomes '-'; a name
    over 200 characters is cut and gets a hash this file does not reproduce: None)."""
    s = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    return s if len(s) <= 200 else None


def transcript_path(home, cwd, session_id):
    name = project_dir_name(cwd) if cwd else None
    if not (name and session_id):
        return None
    return str(Path(home) / ".claude" / "projects" / name / f"{session_id}.jsonl")


def file_sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return (_is_int(v) or isinstance(v, float)) and math.isfinite(v)


def _is_str(v) -> bool:
    return isinstance(v, str) and v != "" and "\0" not in v


def _is_str_list(v) -> bool:
    return isinstance(v, list) and bool(v) and all(_is_str(a) for a in v)


# field -> (check, what it must be); every field of a job is checked
_CHECKS = {
    "task_file": (_is_str, "a non-empty path"),
    "workdir": (_is_str, "a non-empty path"),
    "out_dir": (_is_str, "a non-empty path"),
    "home": (lambda v: _is_str(v) and os.path.isabs(v), "an absolute path"),
    "task_id": (lambda v: v is None or _is_str(v), "null or a non-empty string"),
    "domain": (_is_str, "a non-empty string"),
    "retrieval_variant": (lambda v: v == TR.VARIANT, f"{TR.VARIANT!r} (tau2's alltools is the only variant; "
                                                     "a bm25_grep job is stale)"),
    "sample": (lambda v: _is_int(v) and v >= 0, "an integer >= 0"),
    "seed": (lambda v: v is None or (_is_int(v) and 0 <= v < 2 ** 31), "null or an integer in [0, 2^31)"),
    "simulation_id": (lambda v: v is None or _is_str(v), "null or a non-empty string"),
    "agent_time_s": (lambda v: _is_num(v) and v > 0, "a number > 0"),
    "max_errors": (lambda v: _is_int(v) and v >= 1, "an integer >= 1"),
    "user": (_is_str, "a non-empty string"),
    "user_sim_model": (_is_str, "a non-empty string"),
    "user_llm_args": (lambda v: isinstance(v, dict) and all(isinstance(k, str) and k for k in v),
                      "a JSON object (the customer's llm args; {} = the model's defaults)"),
    "customer_timeout_s": (lambda v: _is_num(v) and v > 0, "a number > 0"),
    "customer_attempts": (lambda v: _is_int(v) and v >= 1, "an integer >= 1"),
    "customer_backoff_s": (lambda v: _is_num(v) and v >= 0, "a number >= 0"),
    "key_fd": (lambda v: v is None or (_is_int(v) and v >= 3), "an fd number >= 3 or null"),
    "token_fd": (lambda v: v is None or (_is_int(v) and v >= 3), "an fd number >= 3 or null"),
    "token_kind": (lambda v: isinstance(v, str) and v in TOKEN_FD_VARS, f"one of {sorted(TOKEN_FD_VARS)}"),
    "model": (_is_str, "a non-empty string"),
    "claude_cmd": (_is_str_list, "a non-empty list of non-empty strings"),
    "container": (lambda v: v is None or (isinstance(v, dict) and set(v) == {"evoctl", "cid", "cwd"}
                                          and _is_str_list(v["evoctl"]) and _is_str(v["cid"])
                                          and bool(CID_RE.match(v["cid"])) and _is_str(v["cwd"])
                                          and os.path.isabs(v["cwd"])),
                  "null or {evoctl: [argv], cid: a container id, cwd: an absolute path}"),
    "close_grace_s": (lambda v: _is_num(v) and v >= 0, "a number >= 0"),
    "mid_turn_grace_s": (lambda v: _is_num(v) and v >= 0, "a number >= 0"),
    "first_message_prefix": (lambda v: isinstance(v, str) and "\0" not in v, "a string (\"\" = none)"),
}
assert set(_CHECKS) == set(JOB_REQUIRED) | set(JOB_DEFAULTS)


def load_job(path) -> dict:
    """The job, every field checked (JobError, exit 2, on anything unusable)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise JobError(f"job file {path}: {e}") from e
    if not isinstance(raw, dict):
        raise JobError("the job is a JSON object")
    unknown = sorted(set(raw) - set(JOB_REQUIRED) - set(JOB_DEFAULTS))
    missing = [k for k in JOB_REQUIRED if raw.get(k) is None]
    if unknown or missing:
        raise JobError(f"job keys: unknown {unknown}, missing {missing}")
    job = {**JOB_DEFAULTS, **raw}
    bad = [f"{k} is {what}" for k, (ok, what) in _CHECKS.items() if not ok(job[k])]
    if bad:
        raise JobError("; ".join(bad))
    if job["key_fd"] is not None and job["key_fd"] == job["token_fd"]:
        raise JobError("key_fd and token_fd are the same fd")
    if job["container"] is not None and job["token_fd"] not in (None, TOKEN_FD_IN_CONTAINER):
        raise JobError(f"token_fd is {TOKEN_FD_IN_CONTAINER} in a container (evoctl enter --keep-fd3)")
    return job


def read_key(fd: int, timeout_s: float = KEY_READ_S) -> str:
    """The key on fd: everything up to EOF or the first newline, stripped; the fd is closed. JobError when nothing
    usable arrives within timeout_s (a writer that keeps the pipe open without a newline) or the key is empty."""
    chunks, deadline = [], time.monotonic() + timeout_s
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise JobError(f"the key fd {fd} gave no newline or EOF within {timeout_s:g} s")
            try:
                ready, _, _ = select.select([fd], [], [], left)
            except (OSError, ValueError) as e:
                raise JobError(f"the key fd {fd} is unusable: {type(e).__name__}") from None
            if not ready:
                continue
            b = os.read(fd, 65536)
            if not b:
                break
            chunks.append(b)
            if b"\n" in b:
                break
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    key = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    if not key:
        raise JobError(f"the key fd {fd} carried no key")
    return key


# ───────────────────────────────────────────────────────────────────────────── the link to claude
class ClaudeLink:
    """One claude process and the CLI's stream-json control protocol over its stdin/stdout. Holds the in-process MCP
    server "tau2" (its tools are set by the agent from tau2's Tool list). No tau2 import here."""

    def __init__(self, argv, env, cwd, out_dir, pass_fds=(), redact=None):
        self.argv, self.env, self.cwd = list(argv), dict(env), str(cwd)
        self.out_dir = Path(out_dir)
        self.pass_fds = tuple(pass_fds)
        self.redact = redact or (lambda s: s)
        self.tools: list = []                   # MCP tool entries {name, description, inputSchema}
        self.proc = None
        self.rc = None
        self._fd = None
        self._buf = b""
        self._eof = False
        self._logs = {}
        self.closed = False
        self.n_req = 0
        self.init_response = None
        self.init = None                        # the system/init event (the first one)
        self.session_id = None
        self.results: list = []                 # a brief of every result event
        self.rate_limit = None                  # the last rate_limit_info seen
        self.rate_rejected = None               # a "rejected" rate_limit_info not yet followed by an assistant event
        self.retrying = None                    # the last api_retry not yet followed by an assistant event
        self.api_retries: list = []             # a brief of every api_retry event (the first 200)
        self.n_api_retries = 0
        self.queue: list = []                   # tools/call requests not yet handed to tau2
        self.turn_tool_uses: list = []          # tau2 tool_use blocks of the current turn, not yet matched to a call
        self.orphans: list = []                 # calls of this turn that got a counter id, waiting for their block
        self.used_ids: set = set()              # every id handed to tau2
        self.late_ids: dict = {}                # counter id -> the toolu_ id of the block that arrived after its call
        self.cancelled: set = set()
        self.in_turn = False
        self.tools_listed = False               # the CLI listed the tau2 server's tools (tools/list answered)
        self.end_conversation = False
        self.n_tool_calls = 0
        self.id_sources: dict = {}
        self.n_unparsed = 0
        self.mcp_methods: list = []             # every MCP method the CLI sent, in order
        self.started_at = None

    # ── process
    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for n in ("claude.stream.jsonl", "claude.stdin.jsonl"):
            self._logs[n] = open(self.out_dir / n, "ab")
        err = open(self.out_dir / "claude.stderr", "ab")
        try:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                         cwd=self.cwd, env=self.env, pass_fds=self.pass_fds, start_new_session=True)
        except OSError as e:
            raise LinkError(f"claude did not start: {type(e).__name__}: {e}") from e
        finally:
            err.close()
            for fd in self.pass_fds:            # claude holds its own copy now; this process never needs the token
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.started_at = time.time()
        self._fd = self.proc.stdout.fileno()
        rid = self._request({"subtype": "initialize", "hooks": None, "sdkMcpServers": [SERVER]})
        until = time.perf_counter() + INIT_TIMEOUT_S
        while self.init_response is None:
            try:
                ev = self._event(until)
            except WallReached:
                raise LinkError(f"no answer to initialize within {INIT_TIMEOUT_S:.0f} s") from None
            if ev is None:
                raise LinkError(f"claude exited (rc {self._exit_code()}) before answering initialize; "
                                f"stderr: {self.stderr_tail()!r}")
            if ev.get("type") == "control_response" and (ev.get("response") or {}).get("request_id") == rid:
                resp = ev["response"]
                if resp.get("subtype") != "success":
                    raise LinkError(f"initialize refused: {resp.get('error')!r}")
                self.init_response = resp.get("response") or {}
            else:
                self._dispatch(ev)

    def _exit_code(self):
        if self.proc is None:
            return None
        try:
            self.rc = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.rc = None
        return self.rc

    def stderr_tail(self, n: int = 400) -> str:
        try:
            return self.redact((self.out_dir / "claude.stderr").read_bytes()[-n:].decode("utf-8", "replace"))
        except OSError:
            return ""

    def close(self, grace: float) -> None:
        """Close stdin, read out the stream until EOF or `grace` seconds, then kill the process group. Idempotent; a
        close interrupted part-way (a signal) is done again in full by the next call."""
        if self.closed:
            return
        if self.proc is None:
            for fh in self._logs.values():
                fh.close()
            self.closed = True
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        until = time.perf_counter() + max(0.0, grace)
        try:
            while self._line(until) is not None:
                pass
        except (WallReached, OSError, ValueError):
            pass
        if self.proc.poll() is None:            # EOF comes as claude exits: let it finish within the grace left
            try:
                self.proc.wait(timeout=max(0.0, until - time.perf_counter()))
            except subprocess.TimeoutExpired:
                pass
        if self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 10.0)):
                try:
                    os.killpg(self.proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        else:                                   # claude is gone; anything it left in its group goes too
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.rc = self.proc.poll()
        log = self._logs.get("claude.stream.jsonl")
        if self._buf and log is not None and not log.closed:      # an unterminated tail, logged as it is
            log.write(self._buf + b"\n")
            self._buf = b""
        try:
            self.proc.stdout.close()
        except OSError:
            pass
        for fh in self._logs.values():
            fh.close()
        self.closed = True

    # ── stdin
    def _send(self, obj) -> None:
        line = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise LinkError(f"claude's stdin is closed ({type(e).__name__}); rc {self._exit_code()}; "
                            f"stderr: {self.stderr_tail()!r}") from e
        self._logs["claude.stdin.jsonl"].write(line)
        self._logs["claude.stdin.jsonl"].flush()

    def _request(self, request: dict) -> str:
        self.n_req += 1
        rid = f"req_{self.n_req}_{os.urandom(4).hex()}"
        self._send({"type": "control_request", "request_id": rid, "request": request})
        return rid

    def _reply(self, request_id, response: dict) -> None:
        self._send({"type": "control_response",
                    "response": {"subtype": "success", "request_id": request_id, "response": response}})

    def send_user(self, text: str) -> None:
        self.turn_tool_uses = []
        self.orphans = []
        self.retrying = None
        self.rate_rejected = None
        self.in_turn = True
        self._send({"type": "user", "session_id": "", "message": {"role": "user", "content": text},
                    "parent_tool_use_id": None})

    def answer(self, call: dict, content, is_error: bool) -> None:
        """The tau2 result of a tools/call, as the MCP reply (none when the CLI cancelled the request). The text passes
        through redact(): a tool's error text never carries the user simulator's key into the container."""
        if call["request_id"] in self.cancelled:
            return
        self._reply(call["request_id"], {"mcp_response": {
            "jsonrpc": "2.0", "id": call["jsonrpc_id"],
            "result": {"content": [{"type": "text", "text": "" if content is None else self.redact(str(content))}],
                       "isError": bool(is_error)}}})

    # ── stdout
    def _line(self, deadline: float):
        """The next stdout line (bytes, logged), None at EOF; WallReached when the deadline passes first."""
        while True:
            i = self._buf.find(b"\n")
            if i >= 0 or (self._eof and self._buf):
                if i < 0:
                    i = len(self._buf)
                raw, self._buf = self._buf[:i], self._buf[i + 1:]
                self._logs["claude.stream.jsonl"].write(raw + b"\n")
                self._logs["claude.stream.jsonl"].flush()
                return raw
            if self._eof:
                return None
            left = deadline - time.perf_counter()
            if left <= 0:
                raise WallReached()
            ready, _, _ = select.select([self._fd], [], [], left)
            if not ready:
                raise WallReached()
            chunk = os.read(self._fd, 1 << 16)
            if chunk:
                self._buf += chunk
            else:
                self._eof = True

    def _event(self, deadline: float):
        """The next JSON object on stdout; None at EOF."""
        while True:
            raw = self._line(deadline)
            if raw is None:
                return None
            s = raw.strip()
            if not s.startswith(b"{"):
                continue
            try:
                ev = json.loads(s)
            except ValueError:
                self.n_unparsed += 1
                continue
            if isinstance(ev, dict):
                return ev

    def wait(self, deadline: float):
        """Read until the agent has something for tau2: ("tool", call) | ("result", event) | ("exit", rc).
        WallReached once the deadline has passed, even while lines keep arriving."""
        while True:
            if time.perf_counter() >= deadline:
                raise WallReached()
            if self.queue:
                return "tool", self.queue.pop(0)
            ev = self._event(deadline)
            if ev is None:
                self.in_turn = False
                return "exit", self._exit_code()
            if self._dispatch(ev) == "result":
                self.in_turn = False
                return "result", ev

    def api_stall(self):
        """What Claude Code is waiting on when it is retrying the API or rate limited (None when it is not)."""
        if self.retrying is not None:
            return f"retrying the API (api_retry {self.retrying})"
        if self.rate_rejected is not None:
            return f"rate limited (rate_limit_info {self.rate_rejected})"
        return None

    def _dispatch(self, ev: dict):
        t = ev.get("type")
        if t == "control_request":
            self._control(ev)
        elif t == "control_cancel_request":
            self.cancelled.add(ev.get("request_id"))
        elif t == "system" and ev.get("subtype") == "init":
            if self.init is None:
                self.init = ev
                self._check_init(ev)
            self.session_id = self.session_id or ev.get("session_id")
        elif t == "system" and ev.get("subtype") == "api_retry":
            brief = {k: ev.get(k) for k in ("attempt", "max_retries", "retry_delay_ms", "error_status")}
            brief["error"] = self.redact(str(ev.get("error"))[:200])
            if ev.get("no_response") is not None:
                brief["no_response"] = ev.get("no_response")
            self.retrying = brief
            self.n_api_retries += 1
            if len(self.api_retries) < 200:
                self.api_retries.append(brief)
        elif t == "assistant":
            self.retrying = None                # the API answered
            self.rate_rejected = None
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    if c.get("name") == END_CONVERSATION:
                        self.end_conversation = True
                    if str(c.get("name") or "").startswith(TOOL_PREFIX):
                        self._block(c)
        elif t == "rate_limit_event" or isinstance(ev.get("rate_limit_info"), dict):
            info = ev.get("rate_limit_info")
            if isinstance(info, dict):
                self.rate_limit = info
                self.rate_rejected = info if info.get("status") == "rejected" else None
        elif t == "result":
            self.session_id = self.session_id or ev.get("session_id")
            self.results.append({k: ev.get(k) for k in ("subtype", "is_error", "api_error_status", "num_turns",
                                                        "terminal_reason", "stop_reason", "duration_ms")})
            return "result"
        return None

    def _check_init(self, ev: dict) -> None:
        servers = {s.get("name"): s.get("status") for s in ev.get("mcp_servers") or [] if isinstance(s, dict)}
        if servers.get(SERVER) not in ("connected", "pending"):
            raise LinkError(f"the {SERVER} MCP server is not connected in claude's session: {servers!r}")

    def _control(self, ev: dict) -> None:
        rid, req = ev.get("request_id"), ev.get("request") or {}
        if req.get("subtype") != "mcp_message":
            self._send({"type": "control_response", "response": {
                "subtype": "error", "request_id": rid,
                "error": f"the tau2 harness does not handle control request {req.get('subtype')!r}"}})
            return
        server, msg = req.get("server_name"), req.get("message") or {}
        method, mid = msg.get("method"), msg.get("id")
        self.mcp_methods.append(method)
        if server != SERVER:
            self._reply(rid, {"mcp_response": {"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": f"Server '{server}' not found"}}})
            return
        if method == "tools/call":
            params = msg.get("params") or {}
            self.queue.append(self._call(rid, mid, params))
            return
        if method == "initialize":
            res = {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": MCP_PROTOCOL, "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER, "version": "1.0.0"}}}
        elif method == "tools/list":
            res = {"jsonrpc": "2.0", "id": mid, "result": {"tools": self.tools}}
            self.tools_listed = True
        elif method == "ping":
            res = {"jsonrpc": "2.0", "id": mid, "result": {}}
        elif mid is None:                       # a notification (notifications/initialized, ...)
            res = {"jsonrpc": "2.0", "result": {}}
        else:
            res = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method '{method}' not found"}}
        self._reply(rid, {"mcp_response": res})

    def _block(self, c: dict) -> None:
        """A tau2 tool_use block from the stream: the oldest call of this turn that got a counter id for the same name
        and arguments takes it (its block came late); otherwise it waits for its call."""
        bid = c.get("id")
        if not bid or bid in self.used_ids or bid in self.late_ids.values():
            return
        for o in self.orphans:
            if TOOL_PREFIX + o["name"] == c.get("name") and o["arguments"] == (c.get("input") or {}):
                self.orphans.remove(o)
                self.late_ids[o["id"]] = bid
                return
        self.turn_tool_uses.append(c)

    def _call(self, rid, mid, params: dict) -> dict:
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        tool_use_id = (params.get("_meta") or {}).get("claudecode/toolUseId")
        source = "_meta"
        if not tool_use_id or tool_use_id in self.used_ids:
            tool_use_id, source = None, "stream"
            for c in self.turn_tool_uses:       # the oldest unmatched block of this turn with this name and input
                if c.get("name") == TOOL_PREFIX + name and (c.get("input") or {}) == args:
                    tool_use_id = c.get("id")
                    break
        self.turn_tool_uses = [c for c in self.turn_tool_uses if c.get("id") != tool_use_id]
        call = {"request_id": rid, "jsonrpc_id": mid, "name": name, "arguments": args}
        if not tool_use_id:
            source = "counter"
            k = self.n_tool_calls + 1
            while f"tau2_call_{k:04d}" in self.used_ids:
                k += 1
            tool_use_id = f"tau2_call_{k:04d}"
            self.orphans.append({**call, "id": tool_use_id})
        tool_use_id = str(tool_use_id)
        self.used_ids.add(tool_use_id)
        self.n_tool_calls += 1
        self.id_sources[source] = self.id_sources.get(source, 0) + 1
        return {**call, "id": tool_use_id, "id_source": source}

    def brief(self, home) -> dict:
        init = self.init or {}
        cwd = init.get("cwd")
        return {"session_id": self.session_id, "cwd": cwd, "transcript": transcript_path(home, cwd, self.session_id),
                "rc": self.rc, "claude_code_version": init.get("claude_code_version"), "model": init.get("model"),
                "permission_mode": init.get("permissionMode"), "mcp_servers": init.get("mcp_servers"),
                "tau2_tools_listed": sorted(t for t in init.get("tools") or [] if str(t).startswith(TOOL_PREFIX)),
                "mcp_tools_listed": self.tools_listed,
                "skills": init.get("skills"), "turns": len(self.results), "results": self.results,
                "tool_calls": self.n_tool_calls, "id_sources": self.id_sources, "late_ids": self.late_ids,
                "end_conversation": self.end_conversation,
                "mcp_methods": sorted(set(m for m in self.mcp_methods if m)), "unparsed_lines": self.n_unparsed,
                "api_retries": {"count": self.n_api_retries, "events": self.api_retries,
                                "delay_ms_total": sum(int(e.get("retry_delay_ms") or 0) for e in self.api_retries)},
                "rate_limit": self.rate_limit}


# ───────────────────────────────────────────────────────────────────────────── the customer's calls
class CustomerCalls:
    """Replaces the customer's generate_next_message: every call is RETRIED as that same call.
    Each attempt runs on a deep copy of the customer's state in a daemon thread and is bounded by timeout_s; an attempt
    that raises, outlives timeout_s (abandoned: it works on its own copy) or returns a message tau2's validate() refuses
    is followed, after backoff_s * 3^(k-1), by the next. CustomerFailed when all `attempts` fail."""

    def __init__(self, user, *, timeout_s: float, attempts: int, backoff_s: float, redact=None):
        self.inner = user.generate_next_message
        self.timeout_s, self.attempts, self.backoff_s = float(timeout_s), int(attempts), float(backoff_s)
        self.redact = redact or (lambda s: s)
        self.calls = 0
        self.seconds = 0.0
        self.failures: list = []
        user.generate_next_message = self

    def _attempt(self, message, state):
        box = {}
        st = copy.deepcopy(state)

        def run():
            try:
                msg, new_state = self.inner(message, st)
                msg.validate()
                box["out"] = (msg, new_state)
            except BaseException as e:          # noqa: BLE001  (recorded; the call is retried)
                box["err"] = e

        th = threading.Thread(target=run, name=f"customer-call-{self.calls}", daemon=True)
        th.start()
        th.join(self.timeout_s)
        if "out" in box:
            return box["out"], None
        if "err" in box:
            e = box["err"]
            return None, self.redact(f"{type(e).__name__}: {e}")[:300]
        return None, f"no answer within {self.timeout_s:g} s"

    def __call__(self, message, state):
        self.calls += 1
        t0 = time.perf_counter()
        try:
            errors = []
            for k in range(self.attempts):
                if k:
                    time.sleep(self.backoff_s * 3 ** (k - 1))
                out, err = self._attempt(message, state)
                if out is not None:
                    return out
                errors.append(err)
                self.failures.append({"call": self.calls, "attempt": k + 1, "error": err})
            raise CustomerFailed(f"the customer's call {self.calls} failed all {self.attempts} attempts: {errors}")
        finally:
            self.seconds += time.perf_counter() - t0

    def brief(self) -> dict:
        return {"calls": self.calls, "seconds": round(self.seconds, 3), "failed_attempts": len(self.failures),
                "failures": self.failures[:50], "timeout_s": self.timeout_s, "attempts": self.attempts,
                "backoff_s": self.backoff_s}


# ───────────────────────────────────────────────────────────────────────────── the tau2 agent
_PENDING: list = []                             # the link of the rollout being built (run_job -> the factory)
_CLASS: list = []


def agent_class():
    """ClaudeCodeAgent, a tau2 HalfDuplexAgent (built on first use: tau2 is imported here)."""
    if _CLASS:
        return _CLASS[0]
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

    class ClaudeCodeAgent(HalfDuplexAgent):
        """Claude Code in tau2's agent seat. Its state is the list of messages it was given and gave. It keeps the agent
        clock: the sum of its turns' times, a turn running from sending the customer's message to its result."""

        def __init__(self, tools, domain_policy: str, link: ClaudeLink):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.link = link
            link.tools = [{"name": s["name"], "description": s.get("description") or s["name"],
                           "inputSchema": s.get("parameters") or {"type": "object", "properties": {}}}
                          for s in (t.openai_schema["function"] for t in tools)]
            self.seed = None
            self.limit_s = float(JOB_DEFAULTS["agent_time_s"])    # set by run_job
            self.used_s = 0.0
            self.turns_s: list = []
            self.turn_t0 = None
            self.deadline = None
            self.started = False
            self.awaiting = None                # the tools/call whose ToolMessage tau2 hands back next
            self.calls = 0
            self.first_prefix = ""              # set by run_job (job first_message_prefix; "" = none)

        def get_init_state(self, message_history=None) -> list:
            return list(message_history or [])

        def set_seed(self, seed: int) -> None:
            self.seed = seed                    # Claude Code takes no seed; kept for the record

        @classmethod
        def is_stop(cls, message) -> bool:
            return bool(message.content) and STOP_TOKEN in message.content

        def clock(self) -> dict:
            return {"limit_s": self.limit_s, "used_s": round(self.used_s, 3), "turns": len(self.turns_s),
                    "turns_s": [round(x, 3) for x in self.turns_s[:500]]}

        def stop(self, message=None, state=None) -> None:
            """tau2 stops the agent (its _finalize, or _cleanup after an exception): a turn still running -- a
            signal, too_many_errors before tau2 hands back a result -- is counted up to now."""
            self._end_turn()

        def _end_turn(self) -> None:
            if self.turn_t0 is not None:
                dt = time.perf_counter() - self.turn_t0
                self.used_s += dt
                self.turns_s.append(dt)
                self.turn_t0 = None

        def generate_next_message(self, message, state):
            self.calls += 1
            t0 = time.perf_counter()
            state.append(message)
            try:
                out = self._respond(message)
            except WallReached:
                self._end_turn()
                stall = self.link.api_stall()
                if stall:
                    raise B.HarnessEnd(f"the {self.limit_s:g} s of agent time ran out while Claude Code was "
                                       f"{self.link.redact(stall)}", reason="infrastructure_error") from None
                raise B.HarnessEnd(f"Claude Code used its {self.limit_s:g} s of agent time", reason="timeout") from None
            except (LinkError, DenseOutage) as e:
                self._end_turn()
                raise B.HarnessEnd(self.link.redact(str(e)), reason="infrastructure_error") from None
            except B.HarnessEnd:
                self._end_turn()
                raise
            out.generation_time_seconds = round(time.perf_counter() - t0, 3)
            state.append(out)
            return out, state

        def _respond(self, message):
            if isinstance(message, UserMessage) and not message.is_tool_call():
                text = message.content or ""
                if not self.started:
                    self.started = True
                    self.link.start()           # bounded by INIT_TIMEOUT_S; never agent time
                    text = self.first_prefix + first_message(self.domain_policy, text)
                self.turn_t0 = time.perf_counter()
                self.deadline = self.turn_t0 + max(0.0, self.limit_s - self.used_s)
                self.link.send_user(text)
            elif isinstance(message, ToolMessage):
                call, self.awaiting = self.awaiting, None
                if call is None or message.id != call["id"]:
                    raise LinkError(f"tau2 returned the result of {message.id!r}, not of the pending call")
                outage = TR._STATE.get("dense_outage")
                if outage:                      # the dense search's embedding service failed every try: end, re-run
                    raise DenseOutage(f"KB_search_dense: the embedding request failed {outage['attempts']}x "
                                      f"({outage['error']}); an infrastructure end, re-run")
                self.link.answer(call, message.content, message.error)
            else:
                raise LinkError(f"the agent takes a customer message or one tool result, not {type(message).__name__}")
            kind, ev = self.link.wait(self.deadline)
            if kind == "tool":
                self.awaiting = ev
                return AssistantMessage(role="assistant", content=None, cost=0.0, tool_calls=[
                    ToolCall(id=ev["id"], name=ev["name"], arguments=ev["arguments"], requestor="assistant")])
            self._end_turn()
            if kind == "exit":
                if self.link.end_conversation and self.link.tools_listed:
                    raise B.HarnessEnd(f"Claude Code ended the session with {END_CONVERSATION} (rc {ev})")
                raise LinkError(f"claude exited (rc {ev}) before ending its turn; stderr: {self.link.stderr_tail()!r}")
            return self._from_result(ev)

        def _from_result(self, ev: dict):
            sub, text, why = ev.get("subtype"), ev.get("result"), ev.get("terminal_reason")
            if not self.link.tools_listed:
                servers = (self.link.init or {}).get("mcp_servers")
                raise LinkError(f"the {SERVER} MCP server never listed its tools before Claude Code answered "
                                f"(mcp_servers {servers!r}, MCP methods {sorted(set(filter(None, self.link.mcp_methods)))})")
            if self.link.end_conversation:
                raise B.HarnessEnd(f"Claude Code ended the session with {END_CONVERSATION}")
            if sub == "error_max_turns" or why == "max_turns":
                raise B.HarnessEnd(f"Claude Code reached a turn limit ({sub}: {ev.get('errors')})")
            if ev.get("is_error") or sub != "success":
                brief = {k: ev.get(k) for k in ("subtype", "api_error_status", "terminal_reason", "errors")}
                if why in SOLVER_TERMINAL_REASONS:
                    raise B.HarnessEnd(f"Claude Code's turn ended in its own error {brief}: {str(text or '')[:300]!r}")
                raise LinkError(f"Claude Code's turn ended in an error {brief}: {str(text or '')[:300]!r}")
            if not isinstance(text, str) or not text.strip():
                raise B.HarnessEnd("Claude Code ended its turn with no text")
            return AssistantMessage(role="assistant", content=text, cost=0.0)

    _CLASS.append(ClaudeCodeAgent)
    return ClaudeCodeAgent


def _factory(tools, domain_policy, task=None, **_ignored):
    """tau2's agent factory signature (build_agent passes tools, domain_policy, llm, llm_args, task, ...). The task is
    never used: the expected actions and the customer's scenario stay on tau2's side."""
    if not _PENDING:
        raise RuntimeError(f"{AGENT} is built by {NAME}.run_job, which hands it its claude link")
    return agent_class()(tools, domain_policy, _PENDING.pop())


def register() -> None:
    """Add claude_code_agent to tau2's registry (a no-op the second time in one process)."""
    from tau2.registry import registry
    if registry.get_agent_factory(AGENT) is None:
        registry.register_agent_factory(_factory, AGENT)


# ───────────────────────────────────────────────────────────────────────────── one rollout
_LIVE: dict = {}                                # what a failure outside run_job's own handling must clean up


def _clear(p: Path):
    """Remove whatever is at p (a link is removed, never followed); returns what was there."""
    if p.is_symlink() or (p.exists() and not p.is_dir()):
        kind = "link" if p.is_symlink() else "file"
        p.unlink()
        return kind
    if p.is_dir():
        shutil.rmtree(p)
        return "dir"
    return None


def _write_infra_record(orch, sim, workdir: Path, meta: dict) -> Path:
    """The record of an infrastructure end: tau2/ without simulation.json (ungradeable by construction)."""
    files = B.bundle(orch, sim, meta)
    out = workdir / B.BUNDLE_DIR
    out.mkdir()
    for name, data in ((INFRA_SIMULATION, files["simulation.json"]), ("run.json", files["run.json"]),
                       ("final_db.json", files["final_db.json"])):
        with open(out / name, "xb") as fh:
            fh.write(data)
    return out


def _finalize_after_exception(orch):
    """A SimulationRun ended infrastructure_error after an exception outside the agent, or None."""
    from tau2.data_model.simulation import TerminationReason
    try:
        orch.termination_reason = TerminationReason.INFRASTRUCTURE_ERROR
        orch.done = True
        return orch._finalize()
    except Exception:
        return None


def run_job(job: dict, *, key=None, scrubbed=()) -> tuple:
    """One rollout. Returns (exit code, summary). The key (or None) is the user simulator's, already read."""
    t_start = time.time()
    task_file = Path(job["task_file"]).absolute()
    workdir = Path(job["workdir"]).absolute()
    out_dir = Path(job["out_dir"]).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    _LIVE["out_dir"] = out_dir
    if not workdir.is_dir():
        raise JobError(f"workdir {workdir} is not a directory")
    _LIVE["workdir"] = workdir
    cleared_before = _clear(workdir / B.BUNDLE_DIR)          # an old or planted bundle never survives this rollout
    if not task_file.is_file():
        raise JobError(f"task_file {task_file} is missing")
    home = Path(job["home"])
    if job["container"] is None:
        if not home.is_dir():
            raise JobError(f"home {home} is not a directory")
        own = os.environ.get("HOME")
        if own and home.resolve() == Path(own).resolve():
            raise JobError(f"home {home} is this process's own HOME (a local claude gets a fresh one)")
    root = task_file.parents[4]

    def redact(s: str) -> str:
        return s.replace(key, "[the user simulator's key]") if key else s

    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["TAU2_DATA_DIR"] = str(root)
    from tau2.utils.utils import DATA_DIR       # the first tau2 import reads TAU2_DATA_DIR, once per process
    if Path(DATA_DIR).resolve() != root.resolve():
        raise JobError(f"tau2 was imported with data root {DATA_DIR}, not {root} (the task file's)")
    from tau2.data_model.simulation import TextRunConfig
    from tau2.config import DEFAULT_MAX_STEPS
    if DEFAULT_MAX_STEPS != MAX_STEPS:
        raise JobError(f"tau2's DEFAULT_MAX_STEPS is {DEFAULT_MAX_STEPS}, not {MAX_STEPS}")
    from tau2.data_model.tasks import Task
    from tau2.runner.build import build_text_orchestrator

    import litellm
    litellm.register_model(LITELLM_GPT6_ENTRIES)   # see LITELLM_GPT6_ENTRIES
    if key:
        litellm.openrouter_key = key             # memory only: litellm's OpenRouter key (main.py, openrouter branch)
    # KB_search_dense through OpenRouter with this key (memory only; None: offline), the documents'
    # embeddings read-only from <root>/.embeddings_cache, the shell sandboxes' extra denyRead. Before any environment is
    # built; a RetrievalError (no cache, a changed variant) escapes run_job: an infrastructure end, exit 75.
    retrieval = TR.install(root, key=key)
    try:
        task = Task.model_validate_json(task_file.read_text(encoding="utf-8"))
    except ValueError as e:
        raise JobError(f"task_file {task_file} is not a tau2 task: {str(e)[:300]}") from None
    domain, variant = job["domain"], job["retrieval_variant"]
    task_id = job["task_id"] or f"{domain}/{task.id}"
    if task_id != f"{domain}/{task.id}":
        raise JobError(f"task_id {task_id!r} is not the task file's {domain}/{task.id}")
    sample = int(job["sample"])
    seed = int(job["seed"]) if job["seed"] is not None else seed_of(task_id, sample)
    sim_id = job["simulation_id"] or f"{task_id}#s{sample}"
    agent_time_s = float(job["agent_time_s"])
    plan = launch_plan(job)
    link = ClaudeLink(plan["argv"], plan["env"], workdir, out_dir, pass_fds=plan["pass_fds"], redact=redact)
    _LIVE["link"] = link

    register()
    config = TextRunConfig(**run_config(job, seed))
    _PENDING[:] = [link]
    try:
        orch = build_text_orchestrator(config, task, seed=seed, simulation_id=sim_id)
    finally:
        _PENDING[:] = []
    orch.agent.limit_s = agent_time_s
    orch.agent.first_prefix = job["first_message_prefix"]
    customer = CustomerCalls(orch.user, timeout_s=job["customer_timeout_s"], attempts=job["customer_attempts"],
                             backoff_s=job["customer_backoff_s"], redact=redact)
    raised, sig = None, None
    try:
        sim, detail = B.run_conversation(orch)
    except Terminated as e:                     # SIGTERM / SIGINT: claude goes at once
        sig = e.name
        link.close(0.0)
        raised = str(e)
        sim, detail = _finalize_after_exception(orch), raised
    except Exception as e:                      # outside the agent: the customer, tau2, this harness
        raised = redact(f"{type(e).__name__}: {e}")[:2000]
        sim, detail = _finalize_after_exception(orch), raised
    link.close(job["mid_turn_grace_s"] if link.in_turn else job["close_grace_s"])

    cc = link.brief(home)
    clock = orch.agent.clock()
    cust = customer.brief()
    meta = {"task_file": str(task_file), "retrieval_variant": variant, "retrieval_kwargs": {},
            "end_detail": redact(detail) if isinstance(detail, str) else detail,
            "name": NAME, "harness_sha256": file_sha256(__file__), "bundle_sha256": file_sha256(B.__file__),
            "task_id": task_id, "sample": sample, "user": job["user"], "user_sim_model": job["user_sim_model"],
            "user_llm_args": dict(config.llm_args_user),
            "solver": {"harness": "claude_code", "model": job["model"], "cc_version_pin": CC_VERSION,
                       "argv": stream_argv(job["model"]), "where": plan["where"]},
            "agent_time": clock, "customer": cust, "retrieval": retrieval,
            "first_message_sha256": first_message_sha256(), "cc": cc}
    prefix = job["first_message_prefix"]
    if prefix:                                  # recorded only when set: a default job's records keep their keys
        meta["first_message_prefix"] = {"sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
                                        "chars": len(prefix)}
    reason = sim.termination_reason.value if sim is not None else "infrastructure_error"
    bundle_dir, graded = None, False
    cleared_after = _clear(workdir / B.BUNDLE_DIR)           # anything the solver planted during the conversation
    if sim is not None:
        if reason == "infrastructure_error":
            bundle_dir = _write_infra_record(orch, sim, workdir, meta)
        else:
            bundle_dir = B.write_bundle(orch, sim, workdir, meta)
            graded = True
    code = EXIT_GRADED if graded else EXIT_INFRA
    summary = {"name": NAME, "task_id": task_id, "sample": sample, "seed": seed, "simulation_id": sim_id,
               "end": {"reason": reason, "detail": meta["end_detail"]}, "raised": raised, "signal": sig,
               "exit_code": code, "bundle": str(bundle_dir) if bundle_dir else None, "graded_bundle": graded,
               "cleared_before": cleared_before, "cleared_after": cleared_after,
               "steps": orch.step_count, "tool_errors": orch.num_errors, "max_errors": orch.max_errors,
               "agent_time": clock, "customer": cust, "seconds": round(time.time() - t_start, 3),
               "user": job["user"], "user_sim_model": job["user_sim_model"], "user_llm_args": meta["user_llm_args"],
               "user_sim_key": "fd" if key else None, "dropped_env": list(scrubbed), "argv": plan["argv"],
               "env_names": sorted(link.env), "claude_env_names": plan["claude_env_names"], "where": plan["where"],
               "retrieval": retrieval, "shell_probe": TR._STATE.get("shell_probe"),
               "dense_outage": TR._STATE.get("dense_outage"), "shell_bins": shell_bins(), "cc": cc}
    if prefix:
        summary["first_message_prefix"] = dict(meta["first_message_prefix"])
    _write_summary(out_dir, summary)
    return code, summary


def _write_summary(out_dir: Path, summary: dict) -> None:
    (out_dir / "harness.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n",
                                          encoding="utf-8")


def _salvage(detail: str, sig=None) -> None:
    """After a failure outside run_job's own handling: kill claude, clear the bundle path, write harness.json."""
    link = _LIVE.get("link")
    if link is not None:
        try:
            link.close(0.0)
        except Exception:
            pass
    cleared = None
    if _LIVE.get("workdir") is not None:
        try:
            cleared = _clear(_LIVE["workdir"] / B.BUNDLE_DIR)
        except OSError:
            cleared = "failed"
    if _LIVE.get("out_dir") is not None:
        try:
            _write_summary(_LIVE["out_dir"], {
                "name": NAME, "end": {"reason": "infrastructure_error", "detail": detail}, "raised": detail,
                "signal": sig, "exit_code": EXIT_INFRA, "bundle": None, "graded_bundle": False,
                "cleared_after": cleared, "shell_bins": shell_bins(),
                "cc": {"rc": link.rc if link is not None else None}})
        except OSError:
            pass


def _on_signal(signum, _frame):
    for s in (signal.SIGTERM, signal.SIGINT):   # the first signal starts the cleanup; later ones never interrupt it
        signal.signal(s, signal.SIG_IGN)
    raise Terminated(signum)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="the rollout's job file (JSON)")
    args = ap.parse_args(argv)
    scrubbed = [k for k in KEY_ENV if k in os.environ]
    for k in scrubbed:                          # a key in the environment is never used: it is dropped, by name
        del os.environ[k]
    _LIVE.clear()
    previous = {}
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[s] = signal.signal(s, _on_signal)
        except ValueError:                      # not the main thread: no handler
            pass
    key = None

    def scrub(msg: str) -> str:
        return msg.replace(key, "[the user simulator's key]") if key else msg

    try:
        job = load_job(args.job)
        if job["key_fd"] is not None:
            key = read_key(job["key_fd"])
        code, _summary = run_job(job, key=key, scrubbed=scrubbed)
        return code
    except JobError as e:
        print(f"{NAME}: {scrub(str(e))}", file=sys.stderr)
        return EXIT_USAGE
    except Terminated as e:
        print(f"{NAME}: {e}", file=sys.stderr)
        _salvage(str(e), e.name)
        return EXIT_INFRA
    except Exception as e:                      # a harness fault: infrastructure, re-run
        msg = scrub(f"{type(e).__name__}: {e}")[:2000]
        print(f"{NAME}: {msg}", file=sys.stderr)
        _salvage(msg)
        return EXIT_INFRA
    finally:
        for s, h in previous.items():
            try:
                signal.signal(s, h)
            except (ValueError, TypeError):
                pass


if __name__ == "__main__":
    sys.exit(main())
