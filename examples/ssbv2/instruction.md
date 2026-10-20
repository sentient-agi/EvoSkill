### 1. Benchmark, data and split

SpreadsheetBench v2, the three non-Visualization categories: Debugging, Financial_Model and Template (297 tasks: 100, 100 and 97). The data root also holds a fourth folder, Visualization, which is not part of this run. A task gives the solver a workbook and an instruction; the solver edits the workbook with Python and saves it.

The data root is `/data/spreadsheetbench-v2`, with one `dataset.json` per category; each record has `id`, `instruction`, `spreadsheet_path`, `golden_response_path` and `answer_position`; the two paths are relative to that category's folder. Several records share one golden_response_path (a Debugging project has ten tasks on one reference workbook, a Financial_Model project five); tasks that share a reference workbook share an answer, and no such group spans two of train, val and test. A task id is the category plus the record id (`<Category>/<NN_MM>`).

Use the given split: `/data/split.json`, 57 train / 57 val / 104 test.

### 2. What the solver gets and must produce

Per task, the solver gets the instruction and the input workbook in its working directory. It must save the modified workbook to the output path it is given.

### 3. Grader

The grader is `benchmark_grader.py`, given next to `plugin_api.py`. Its `grade(task, snapshot)` scores one rollout from 0 to 1; the snapshot is the solver's working directory when it exited. It reads the task's `inputs[0]` (the input workbook), `answer_files[0]` (the reference workbook) and `grading["answer_position"]`. The loop runs it; you do not need to.

### 4. Environment

The solver and the grader run in the image built from `scripts/daytona/Dockerfile.ssb_leg` in the same repository. The solver's Python is 3.11 with openpyxl 3.1.3, pandas 2.2.0 and numpy 1.26.4.

### 5. Run size

3 epochs · 3 samples per task.
