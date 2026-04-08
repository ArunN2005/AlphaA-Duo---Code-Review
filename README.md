---
title: CodeReview ENV
emoji: "\ud83e\udde0"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# CodeReview-ENV

An OpenENV environment that trains AI agents to review code like a
senior engineer. The agent reads pull request diffs and learns to
identify bugs, classify their severity, and suggest fixes.

## Teammate quick start

### 1) Clone and enter the project

```bash
git clone https://github.com/ArunN2005/AlphaA-Duo---Code-Review.git
cd AlphaA-Duo---Code-Review
```

### 2) Set up Python environment

Use Python 3.10+.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -U pip
pip install -e .
pip install -r server/requirements.txt
```

### 4) Run the local API server

```bash
python server/app.py
```

Server URL: `http://localhost:8000`

Web demo URLs:
- `http://localhost:8000/web`
- `http://localhost:8000/web/cve`

### 5) Run local checks/tests

Smoke test:

```bash
python test_local.py
```

Interactive tester:

```bash
python interactive_tester.py --url http://localhost:8000
```

Policy evaluation:

```bash
python evaluate_metrics.py --policy heuristic --episodes 200
python evaluate_metrics.py --policy random --episodes 200
```

Full train/eval artifact pipeline:

```bash
python train_eval.py --policy both --episodes 300 --out results
```

Real CVE seed verification:

```bash
python real_cve_check.py
```

## What the agent learns
- Finding the exact line containing a bug
- Classifying severity: critical, medium, or style
- Suggesting a correct fix
- Avoiding false positives (penalized for flagging clean code)

## Reward structure
| Signal | Points |
|--------|--------|
| Correct line number | +1.0 |
| Close line number (within 2) | +0.4 |
| Correct severity | +0.5 |
| Valid fix suggestion | +0.3 |
| Security bug found exactly | +0.5 bonus |
| False positive penalty | -0.5 |

## How to use
### Remote (no setup needed)
from openenv import CodeReviewEnv
env = CodeReviewEnv(url="https://YOUR-SPACE-URL.hf.space")
obs = env.reset()
obs, reward, done, info = env.step(action)

### Local Docker
docker pull registry.hf.space/YOUR-USERNAME/codereview-env:latest
docker run -p 8000:8000 registry.hf.space/YOUR-USERNAME/codereview-env:latest

## Dataset
50 real-world style PRs across 10 bug categories including SQL injection,
XSS, hardcoded secrets, off-by-one errors, and path traversal.

Current project includes an expanded dataset with 650 entries:
- Multi-bug diffs (episodes can require more than one action)
- Clean calibration diffs for false-positive control
- Mixed Python, JavaScript, and Go patches

Real-world CVE-inspired seed cases are included with provenance metadata
(`source_type=real_cve_seed`), covering:
- Django SQL injection fixes (CVE-2024-53908, CVE-2026-1287)
- python-sql SQL injection (CVE-2024-9774)
- Twisted XSS (CVE-2024-41810)
- aiohttp and youtube-dl traversal patterns
- python-jwt auth bypass (CVE-2022-39227)
- joblib eval() RCE class (CVE-2022-21797)
- GitPython command-sanitization/RCE class (CVE-2023-40267)
- python-multipart ReDoS (CVE-2024-24762)

Ingestion format and loader for real CVE seeds:
- Manifest file: data/cve_manifest.json
- Loader script: ingest_cve_manifest.py

To ingest from manifest:

python ingest_cve_manifest.py

Legacy helper (also idempotent):

python add_real_world_cve_cases.py

## Live web demo
Start the server and open:

http://localhost:8000/web

For judge-facing CVE side-by-side baseline vs reviewer comparison:

http://localhost:8000/web/cve

This UI lets you paste Python/JavaScript/Go snippets and get color-coded
severity badges from the review endpoint.

## Advanced evaluation
Run local smoke test:

python test_local.py

Run interactive HTTP tester:

python interactive_tester.py --url http://localhost:8000

Run policy evaluation with per-class precision/recall/F1:

python evaluate_metrics.py --policy heuristic --episodes 200
python evaluate_metrics.py --policy random --episodes 200

Run full train/eval artifact pipeline (curves + confusion matrices + CSVs):

python train_eval.py --policy both --episodes 300 --out results

Run real CVE seed verification (measured before/after baseline):

python real_cve_check.py

## Action space
line_number: int - which line contains the bug
severity: str - "critical", "medium", or "style"
message: str - description of the bug
suggested_fix: str - how to fix it
rationale: str - optional reasoning text used in reward shaping

## Observation space
diff: str - the pull request diff
filename: str - the file being reviewed
episode_id: int - the PR identifier
file_context: str - local file-level context
repo_summary: str - repository/module-level context

## State space
current_pr_index: int - active PR index in dataset
steps_taken: int - actions taken in the current episode
max_actions: int - action budget for this episode
reviewed_lines: list[int] - lines reviewed so far
session_history: list[dict] - action and reward trail within episode
