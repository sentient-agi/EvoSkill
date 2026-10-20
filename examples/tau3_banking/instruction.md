### 1. Benchmark, data and split

τ³-bench, the banking domain only (`banking_knowledge`). A task is a conversation between a bank's customer service agent (the solver) and a simulated customer who has a goal; the agent handles it according to the bank's policy, using the bank's tools and its knowledge base.

The data root is `/data/tau3_data_363133a`. `tau2/domains/banking_knowledge/` holds `db.json` (the bank's database), `documents/` (the knowledge base, 698 documents), `prompts/` (the agent's policy texts) and `tasks/` with one file per task, `task_NNN.json` (the customer's scenario, the database's starting state and how the task is judged). `tau2/user_simulator/` holds the customer simulator's guidelines. `.embeddings_cache/` at the data root holds the knowledge base's precomputed embeddings (OpenAI `text-embedding-3-large`, 698 documents), used by the dense search. A task id is `banking_knowledge/` plus the task file's name without `.json` (e.g. `banking_knowledge/task_004`).

Use the given split: `/data/split.json`, 19 train / 20 validation / 31 test.

### 2. What the solver gets and must produce

Per task, the solver is the bank's agent in one conversation. It gets the bank's policy for the `alltools` knowledge-base setup (search with `KB_search_bm25` (keyword) and `KB_search_dense` (embedding), and a read-only `shell` over the knowledge base) and the customer's messages, and it acts through the bank's tools. When the conversation ends, the conversation and the bank's final database are saved under `tau2/` in the solver's working directory.

### 3. Grader

The grader is `benchmark_grader.py`, given next to `plugin_api.py`. Its `grade(task, snapshot)` scores one rollout from 0 to 1; the snapshot is the solver's working directory when the conversation ended. It reads the task's `answer_files[0]` (the task file), `grading["domain"]` (`banking_knowledge`) and `grading["retrieval_variant"]` (`alltools`); a task has no input files.

### 4. Run size

3 epochs · 3 samples per task.
