import os, json
from pathlib import Path
from daytona import Daytona, DaytonaConfig

run_info = json.loads(Path('.evoskill/remote_run.json').read_text())
client = Daytona(DaytonaConfig(api_key=os.environ['DAYTONA_API_KEY']))
sandbox = client.get(run_info['extra']['sandbox_id'])

sandbox.process.exec('cd /workspace/examples/appworld2 && git checkout -f program/iter-prompt-1')
content = sandbox.fs.download_file('/workspace/examples/appworld2/.claude/prompts/instructions.txt')
Path('evolved_prompt.txt').write_bytes(content)
print('Saved evolved_prompt.txt')

sandbox.process.exec('cd /workspace/examples/appworld2 && git checkout -f program/base')
content = sandbox.fs.download_file('/workspace/examples/appworld2/.claude/prompts/instructions.txt')
Path('base_prompt.txt').write_bytes(content)
print('Saved base_prompt.txt')
