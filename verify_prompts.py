import os, json
from pathlib import Path
from daytona import Daytona, DaytonaConfig

run_info = json.loads(Path('.evoskill/remote_run.json').read_text())
client = Daytona(DaytonaConfig(api_key=os.environ['DAYTONA_API_KEY']))
sandbox = client.get(run_info['extra']['sandbox_id'])

# Check current instructions.txt (should be base after RESTORING step)
current = sandbox.fs.download_file('/workspace/examples/appworld2/.claude/prompts/instructions.txt')
evolved = sandbox.fs.download_file('/workspace/examples/appworld2/.claude/prompts/evolved_instructions.txt')
backup = sandbox.fs.download_file('/workspace/examples/appworld2/.claude/prompts/instructions.txt.bak')

print(f"Base prompt size:    {len(backup)} bytes")
print(f"Evolved prompt size: {len(evolved)} bytes")
print(f"Current prompt size: {len(current)} bytes")
print(f"Base == Evolved: {backup == evolved}")
print(f"Base == Current: {backup == current}")
print()

if backup == evolved:
    print("WARNING: Base and evolved prompts are IDENTICAL — no difference was tested!")
else:
    print("OK: Base and evolved prompts are DIFFERENT")
    # Show what's different
    base_lines = backup.decode().splitlines()
    evolved_lines = evolved.decode().splitlines()
    print(f"\nBase: {len(base_lines)} lines")
    print(f"Evolved: {len(evolved_lines)} lines")
    added = [l for l in evolved_lines if l not in base_lines]
    if added:
        print(f"\nLines in evolved but not base:")
        for l in added:
            print(f"  + {l.strip()}")
