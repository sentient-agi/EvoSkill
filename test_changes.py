"""Quick local verification of changes before remote run."""
import sys
from pathlib import Path

errors = []

# 1. Test nested .git exclusion
print("1. Testing nested .git exclusion...")
from src.remote.sync import should_exclude_upload
assert should_exclude_upload(Path("examples/appworld2/.git/HEAD")), "Should exclude nested .git"
assert should_exclude_upload(Path(".git/config")), "Should exclude root .git"
assert not should_exclude_upload(Path("examples/appworld2/scripts/test_halo_agent.py")), "Should NOT exclude scripts"
print("   PASS")

# 2. Test upload_file_list doesn't include .git from nested repos
print("2. Testing upload_file_list excludes nested .git...")
from src.remote.sync import upload_file_list
project_root = Path(__file__).parent
files = upload_file_list(project_root)
git_files = [f for f in files if ".git" in f.parts]
if git_files:
    print(f"   FAIL — found {len(git_files)} .git files: {git_files[:3]}")
    errors.append("nested .git files in upload list")
else:
    print("   PASS")

# 3. Test LoopConfig accepts new params
print("3. Testing LoopConfig with new params...")
from src.loop.config import LoopConfig
config = LoopConfig(
    max_iterations=10,
    frontier_size=2,
    concurrency=1,
    failure_sample_count=5,
    no_improvement_limit=5,
    evolution_mode="prompt_only",
    samples_per_category=15,
)
assert config.samples_per_category == 15
assert config.failure_sample_count == 5
assert config.no_improvement_limit == 5
print("   PASS")

# 4. Test test_halo_agent.py has --runner-provider arg
print("4. Testing test_halo_agent.py CLI args...")
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "examples.appworld2.scripts.test_halo_agent", "--help"],
    capture_output=True, text=True, cwd=str(project_root),
)
if "--runner-provider" in result.stdout and "--experiment-name" in result.stdout:
    print("   PASS")
else:
    print(f"   FAIL — missing args in help output")
    errors.append("test_halo_agent missing CLI args")

# 5. Test run_evolution.py imports and difficulty split function
print("5. Testing run_evolution.py imports...")
try:
    from examples.appworld2.scripts.run_evolution import build_evolution_loop, main
    print("   PASS")
except Exception as e:
    print(f"   FAIL — {e}")
    errors.append(f"run_evolution import: {e}")

# 6. Test frontier display fix
print("6. Testing get_frontier_with_scores handles errors...")
from src.registry.manager import ProgramManager
# Just verify the method exists and the code path is correct
import inspect
source = inspect.getsource(ProgramManager.get_frontier_with_scores)
if "scored.append((name, 0.0, index))" in source:
    print("   PASS")
else:
    print("   FAIL — fallback not found in source")
    errors.append("frontier display fix missing")

print()
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
