# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Codereview Env Environment.

This module creates an HTTP server that exposes the CodereviewEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import json
from pathlib import Path
from uuid import uuid4
from fastapi import Body

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Observation
    from pydantic import Field
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ReviewAction, ReviewObservation
    from .environment import CodeReviewEnvironment
except (ModuleNotFoundError, ImportError):
    from models import ReviewAction, ReviewObservation
    from server.environment import CodeReviewEnvironment


class APIReviewObservation(Observation):
    """HTTP-compatible observation that includes OpenENV-required fields."""

    diff: str
    filename: str
    episode_id: int
    file_context: str = ""
    repo_summary: str = ""
    total_bugs: int = 0
    remaining_bugs: int = 0
    is_clean: bool = False
    bug_categories: list[str] = Field(default_factory=list)


class HTTPCodeReviewEnvironment(Environment):
    """Adapter to expose CodeReviewEnvironment through OpenENV HTTP server."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = CodeReviewEnvironment()

    def _coerce_observation(self, raw_obs):
        """Normalize environment outputs to a full observation shape.

        Some OpenENV versions can surface state-like objects during reset.
        This adapter converts those into the expected observation fields.
        """
        if hasattr(raw_obs, "diff") and hasattr(raw_obs, "episode_id"):
            return raw_obs

        if hasattr(raw_obs, "current_pr_index"):
            idx = int(getattr(raw_obs, "current_pr_index"))
            pr = self._env.prs[idx]
            return ReviewObservation(
                diff=pr["diff"],
                filename=pr["filename"],
                episode_id=pr["id"],
                file_context=getattr(self._env, "current_file_context", ""),
                repo_summary=getattr(self._env, "current_repo_summary", ""),
                total_bugs=len(getattr(self._env, "current_bugs", [])),
                remaining_bugs=len(getattr(self._env, "current_bugs", []))
                - len(getattr(self._env, "found_bug_indices", set())),
                is_clean=len(getattr(self._env, "current_bugs", [])) == 0,
                bug_categories=sorted(
                    {b.get("category", "logic_bug") for b in getattr(self._env, "current_bugs", [])}
                )
                if getattr(self._env, "current_bugs", [])
                else ["clean"],
            )

        raise TypeError(f"Unsupported observation type returned by environment: {type(raw_obs)!r}")

    def reset(self, **kwargs):
        obs = self._coerce_observation(self._env.reset(**kwargs))
        return APIReviewObservation(
            diff=obs.diff,
            filename=obs.filename,
            episode_id=obs.episode_id,
            file_context=obs.file_context,
            repo_summary=obs.repo_summary,
            total_bugs=obs.total_bugs,
            remaining_bugs=obs.remaining_bugs,
            is_clean=obs.is_clean,
            bug_categories=obs.bug_categories,
            reward=0.0,
            done=False,
        )

    def step(self, action: ReviewAction, **kwargs):
        obs, reward, done, info = self._env.step(action)
        obs = self._coerce_observation(obs)
        return APIReviewObservation(
            diff=obs.diff,
            filename=obs.filename,
            episode_id=obs.episode_id,
            file_context=obs.file_context,
            repo_summary=obs.repo_summary,
            total_bugs=obs.total_bugs,
            remaining_bugs=obs.remaining_bugs,
            is_clean=obs.is_clean,
            bug_categories=obs.bug_categories,
            reward=reward,
            done=done,
            metadata=info,
        )

    @property
    def state(self):
        return self._env.state


# Create the app with web interface and README integration
app = create_app(
    HTTPCodeReviewEnvironment,
    ReviewAction,
    APIReviewObservation,
    env_name="codereview_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


class SnippetReviewRequest(BaseModel):
    code: str
    language: str = "python"
    repo_summary: str = ""


class CveCompareRequest(BaseModel):
    cve: str


class WebResetRequest(BaseModel):
    session_id: str | None = None


class WebStepRequest(BaseModel):
    session_id: str
    action: ReviewAction


class LegacyStepRequest(BaseModel):
    action: ReviewAction
    session_id: str | None = None


_WEB_SESSIONS: dict[str, CodeReviewEnvironment] = {}
_LEGACY_SESSION_ID = "legacy-default"


def _get_or_create_web_env(session_id: str | None) -> tuple[str, CodeReviewEnvironment]:
    sid = session_id or str(uuid4())
    env = _WEB_SESSIONS.get(sid)
    if env is None:
        env = CodeReviewEnvironment()
        _WEB_SESSIONS[sid] = env
    return sid, env


def _replace_legacy_route(path: str, method: str) -> None:
    method = method.upper()
    app.router.routes = [
        r
        for r in app.router.routes
        if not (getattr(r, "path", None) == path and method in (getattr(r, "methods", set()) or set()))
    ]


def _review_snippet(code: str) -> list[dict]:
    findings: list[dict] = []
    lines = code.splitlines()
    for idx, line in enumerate(lines, start=1):
        lower = line.lower()

        if "select" in lower and ("%" in line or "{" in line or "+" in line) and "where" in lower:
            findings.append(
                {
                    "line": idx,
                    "severity": "critical",
                    "category": "SQL Injection",
                    "message": "Query appears to concatenate untrusted input into SQL.",
                    "fix": "Use parameterized placeholders and bound parameters.",
                }
            )
        if "innerhtml" in lower or "dangerouslysetinnerhtml" in lower:
            findings.append(
                {
                    "line": idx,
                    "severity": "critical",
                    "category": "XSS",
                    "message": "HTML sink receives untrusted content.",
                    "fix": "Escape or sanitize user content and use text-only rendering APIs.",
                }
            )
        if "eval(" in lower or "exec(" in lower:
            findings.append(
                {
                    "line": idx,
                    "severity": "critical",
                    "category": "Code Execution",
                    "message": "Dynamic code execution on variable input is dangerous.",
                    "fix": "Replace eval/exec with a safe parser or explicit dispatch table.",
                }
            )
        if "os.path.join" in lower and ".." in lower:
            findings.append(
                {
                    "line": idx,
                    "severity": "critical",
                    "category": "Path Traversal",
                    "message": "Path assembly may allow directory traversal.",
                    "fix": "Normalize input and enforce path remains under trusted base directory.",
                }
            )
        if "random.randint" in lower or "math.random" in lower:
            findings.append(
                {
                    "line": idx,
                    "severity": "medium",
                    "category": "Insecure Random",
                    "message": "Non-cryptographic randomness used where security may matter.",
                    "fix": "Use a cryptographically secure random source for tokens and nonces.",
                }
            )
        if "api_key" in lower and ("=" in line) and ("\"" in line or "'" in line):
            findings.append(
                {
                    "line": idx,
                    "severity": "critical",
                    "category": "Hardcoded Secret",
                    "message": "Possible secret literal in source code.",
                    "fix": "Load secrets from environment or a secret manager.",
                }
            )

    return findings


def _load_cve_manifest() -> list[dict]:
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "cve_manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _line_for_marker(diff: str, marker: str) -> int:
    for idx, line in enumerate(diff.splitlines(), start=1):
        if marker in line:
            return idx
    return 0


def _baseline_for_case(case: dict) -> dict:
    return {
        "line_number": 0,
        "severity": "style",
        "message": "Looks good.",
        "suggested_fix": "No changes.",
        "rationale": "baseline",
    }


def _reviewer_for_case(case: dict) -> dict:
    diff = case.get("diff", "")
    category = case.get("category", "logic_bug")
    line_number = _line_for_marker(diff, case.get("marker", ""))
    severity = case.get("severity", "medium")

    message = "Potential issue in modified code path."
    fix = case.get("correct_fix", "Add validation and use safer primitives.")
    rationale = f"cve-focused reviewer, category={category}"

    if category == "sql_injection":
        message = "SQL injection risk from unsafe query composition."
        fix = "Use parameterized SQL placeholders and bound parameters."
    elif category == "xss":
        message = "XSS risk from unescaped content in HTML output."
        fix = "Escape/sanitize user content before rendering in HTML sinks."
    elif category == "path_traversal":
        message = "Path traversal risk from untrusted path segments."
        fix = "Normalize input and enforce base-directory confinement checks."
    elif category in {"missing_auth", "auth_bypass"}:
        message = "Authentication/authorization bypass in sensitive path."
        fix = "Verify auth claims/signature before privileged operations."
    elif category == "redos":
        message = "Regex pattern may enable catastrophic backtracking (ReDoS)."
        fix = "Use linear-time parsing pattern and bound input lengths."

    return {
        "line_number": line_number,
        "severity": severity,
        "message": message,
        "suggested_fix": fix,
        "rationale": rationale,
    }


def _score_action(case: dict, action: dict) -> dict:
    bug_line = _line_for_marker(case.get("diff", ""), case.get("marker", ""))
    expected_sev = case.get("severity", "medium")
    reward = 0.0
    if action["line_number"] == bug_line:
        reward += 1.0
        if expected_sev == "critical":
            reward += 0.5
    elif abs(action["line_number"] - bug_line) <= 2:
        reward += 0.4

    if action["severity"] == expected_sev:
        reward += 0.5

    target = set((case.get("correct_fix", "")).lower().split())
    got = set((action.get("suggested_fix", "")).lower().split())
    if len(target.intersection(got)) >= 3:
        reward += 0.3

    if action["line_number"] == 0:
        reward -= 0.5

    return {
        "reward": round(reward, 4),
        "line_hit": action["line_number"] == bug_line,
        "severity_hit": action["severity"] == expected_sev,
        "bug_line": bug_line,
    }


@app.post("/review-snippet")
def review_snippet(payload: SnippetReviewRequest) -> dict:
    findings = _review_snippet(payload.code)
    return {
        "language": payload.language,
        "repo_summary": payload.repo_summary,
        "finding_count": len(findings),
        "findings": findings,
    }


@app.post("/rl/reset")
def rl_reset(payload: WebResetRequest) -> dict:
    sid, env = _get_or_create_web_env(payload.session_id)
    obs = env.reset()
    return {
        "session_id": sid,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/rl/step")
def rl_step(payload: WebStepRequest) -> dict:
    sid, env = _get_or_create_web_env(payload.session_id)
    obs, reward, done, info = env.step(payload.action)
    out = obs.model_dump()
    out["metadata"] = info
    return {
        "session_id": sid,
        "observation": out,
        "reward": reward,
        "done": done,
    }


_replace_legacy_route("/reset", "POST")
_replace_legacy_route("/step", "POST")
_replace_legacy_route("/web", "GET")
_replace_legacy_route("/web/cve", "GET")
_replace_legacy_route("/web/reset", "POST")
_replace_legacy_route("/web/step", "POST")


@app.post("/reset")
def legacy_reset(payload: dict = Body(default={})) -> dict:
    sid = payload.get("session_id") if isinstance(payload, dict) else None
    sid = sid or _LEGACY_SESSION_ID
    sid, env = _get_or_create_web_env(sid)
    obs = env.reset()
    return {
        "session_id": sid,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/step")
def legacy_step(payload: LegacyStepRequest) -> dict:
    sid = payload.session_id or _LEGACY_SESSION_ID
    sid, env = _get_or_create_web_env(sid)
    obs, reward, done, info = env.step(payload.action)
    out = obs.model_dump()
    out["metadata"] = info
    return {
        "session_id": sid,
        "observation": out,
        "reward": reward,
        "done": done,
    }


@app.post("/web/reset")
def web_reset(payload: dict = Body(default={})) -> dict:
    return legacy_reset(payload)


@app.post("/web/step")
def web_step(payload: LegacyStepRequest) -> dict:
    return legacy_step(payload)


@app.get("/cve-cases")
def cve_cases() -> dict:
    items = _load_cve_manifest()
    return {
        "count": len(items),
        "cases": [
            {
                "cve": x.get("cve", ""),
                "repo": x.get("repo", ""),
                "category": x.get("category", ""),
                "severity": x.get("severity", ""),
                "filename": x.get("filename", ""),
            }
            for x in items
        ],
    }


@app.post("/review-cve-compare")
def review_cve_compare(payload: CveCompareRequest) -> dict:
    items = _load_cve_manifest()
    case = next((x for x in items if x.get("cve") == payload.cve), None)
    if not case:
        return {"error": f"CVE not found: {payload.cve}"}

    baseline = _baseline_for_case(case)
    reviewer = _reviewer_for_case(case)
    baseline_score = _score_action(case, baseline)
    reviewer_score = _score_action(case, reviewer)

    return {
        "cve": case.get("cve"),
        "repo": case.get("repo"),
        "category": case.get("category"),
        "severity": case.get("severity"),
        "filename": case.get("filename"),
        "diff": case.get("diff"),
        "baseline": {"action": baseline, "result": baseline_score},
        "reviewer": {"action": reviewer, "result": reviewer_score},
    }


@app.get("/web", response_class=HTMLResponse)
def web_ui() -> str:
    return """
<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>CodeReview-ENV RL Interface</title>
    <style>
        :root {
            --bg: #f5f7fb;
            --ink: #1a1f2e;
            --panel: #ffffff;
            --critical: #b42318;
            --medium: #b54708;
            --style: #175cd3;
                        --accent: #0f766e;
                        --muted: #667085;
        }
        body { margin: 0; font-family: 'Segoe UI', Tahoma, sans-serif; background: radial-gradient(circle at top right, #d7f7ef 0%, var(--bg) 55%); color: var(--ink); }
        .wrap { max-width: 980px; margin: 24px auto; padding: 0 16px; }
        .card { background: var(--panel); border-radius: 14px; padding: 18px; box-shadow: 0 10px 32px rgba(16,24,40,0.08); }
        h1 { margin: 0 0 8px 0; font-size: 28px; }
        p { margin: 4px 0 14px 0; }
                textarea, input, select { width: 100%; border: 1px solid #d4d7e1; border-radius: 10px; padding: 10px; font-size: 14px; box-sizing: border-box; }
                textarea { min-height: 100px; font-family: Consolas, monospace; }
                pre { white-space: pre-wrap; background: #f8f9fc; border: 1px solid #e4e7ec; border-radius: 10px; padding: 10px; max-height: 320px; overflow: auto; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        button { margin-top: 12px; background: var(--accent); color: white; border: none; border-radius: 10px; padding: 11px 14px; font-weight: 600; cursor: pointer; }
                .meta { font-size: 12px; color: var(--muted); }
                .result { margin-top: 12px; border: 1px solid #e4e7ec; border-radius: 10px; padding: 10px; background: #fcfcfd; }
                .pill { display: inline-block; margin-right: 8px; padding: 4px 10px; border-radius: 999px; font-size: 12px; }
                .ok { background: #ecfdf3; color: #027a48; }
                .warn { background: #fff6ed; color: #b54708; }
                .bad { background: #fef3f2; color: #b42318; }
    </style>
</head>
<body>
    <div class='wrap'>
        <div class='card'>
                        <h1>CodeReview-ENV</h1>
                        <p>RL environment loop: reset a PR, submit structured review action, inspect reward and done.</p>

                        <button onclick='resetEnv()'>Reset - Load New PR</button>

                        <h3 style='margin-bottom:6px;'>Current PR</h3>
                        <div class='meta' id='obsMeta'>No PR loaded yet.</div>
                        <pre id='diffBox'>Click Reset to load a diff from the dataset.</pre>

                        <h3 style='margin-bottom:6px;'>Your Review Action</h3>
                        <div class='grid'>
                            <div>
                                <label>line_number</label>
                                <input id='line_number' type='number' value='0'>
                            </div>
                            <div>
                                <label>severity</label>
                                <select id='severity'>
                                    <option value='critical'>critical</option>
                                    <option value='medium'>medium</option>
                                    <option value='style'>style</option>
                                </select>
                            </div>
                        </div>
                        <label style='margin-top:8px; display:block;'>message</label>
                        <textarea id='message' placeholder='Describe the bug briefly'></textarea>
                        <label style='margin-top:8px; display:block;'>suggested_fix</label>
                        <textarea id='suggested_fix' placeholder='Describe the correct fix'></textarea>

                        <button onclick='submitStep()'>Submit Review</button>

                        <div class='result'>
                            <div id='scoreLine' class='meta'>Reward: - | Done: -</div>
                            <div id='statusPills' style='margin-top:8px;'></div>
                            <pre id='infoBox' style='margin-top:8px;'>Step info will appear here.</pre>
                        </div>
        </div>
    </div>
<script>
let latestObs = null;
let sessionId = null;

function esc(text) {
    return String(text ?? '');
}

function renderObs(payload) {
    const obs = payload.observation || {};
    latestObs = obs;
    document.getElementById('obsMeta').innerText = `filename: ${esc(obs.filename)} | episode_id: ${esc(obs.episode_id)} | total_bugs: ${esc(obs.total_bugs)} | remaining_bugs: ${esc(obs.remaining_bugs)} | clean: ${esc(obs.is_clean)}`;
    document.getElementById('diffBox').innerText = esc(obs.diff);
}

function renderStep(payload) {
    document.getElementById('scoreLine').innerText = `Reward: ${payload.reward} | Done: ${payload.done}`;

    const info = (payload.observation && payload.observation.metadata) || {};
    document.getElementById('infoBox').innerText = JSON.stringify(info, null, 2);

    const pills = document.getElementById('statusPills');
    pills.innerHTML = '';
    const donePill = document.createElement('span');
    donePill.className = `pill ${payload.done ? 'warn' : 'ok'}`;
    donePill.innerText = payload.done ? 'Episode Done' : 'Episode Active';
    pills.appendChild(donePill);

    if (typeof payload.reward === 'number') {
        const rewardPill = document.createElement('span');
        rewardPill.className = `pill ${payload.reward > 0 ? 'ok' : (payload.reward < 0 ? 'bad' : 'warn')}`;
        rewardPill.innerText = payload.reward > 0 ? 'Positive Reward' : (payload.reward < 0 ? 'Negative Reward' : 'Neutral Reward');
        pills.appendChild(rewardPill);
    }

    if (payload.observation) {
        renderObs({observation: payload.observation});
    }
}

async function resetEnv() {
    const res = await fetch('/rl/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId})
    });
    const data = await res.json();
    sessionId = data.session_id;
    renderObs(data);
    document.getElementById('scoreLine').innerText = 'Reward: 0.0 | Done: false';
    document.getElementById('infoBox').innerText = 'Step info will appear here.';
    document.getElementById('statusPills').innerHTML = '<span class="pill ok">Episode Active</span>';
}

async function submitStep() {
    if (!sessionId) {
        await resetEnv();
    }

    const action = {
        line_number: Number(document.getElementById('line_number').value || 0),
        severity: document.getElementById('severity').value,
        message: document.getElementById('message').value,
        suggested_fix: document.getElementById('suggested_fix').value
    };

    const res = await fetch('/rl/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, action})
    });

    const data = await res.json();
    sessionId = data.session_id;
    renderStep(data);
}
</script>
</body>
</html>
"""


@app.get("/web/cve", response_class=HTMLResponse)
def web_cve_ui() -> str:
    return """
<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>CodeReview-ENV CVE Compare</title>
    <style>
        :root { --bg:#f5f7fb; --ink:#1a1f2e; --card:#fff; --critical:#b42318; --medium:#b54708; --style:#175cd3; --ok:#027a48; }
        body { margin:0; font-family:'Segoe UI',Tahoma,sans-serif; color:var(--ink); background: linear-gradient(135deg,#ecfeff,#f8fafc); }
        .wrap { max-width:1100px; margin:24px auto; padding:0 16px; }
        .card { background:var(--card); border-radius:14px; box-shadow:0 10px 30px rgba(16,24,40,.08); padding:16px; }
        .row { display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center; }
        select, button { border:1px solid #d0d5dd; border-radius:10px; padding:10px; font-size:14px; }
        button { background:#0f766e; color:#fff; border:none; cursor:pointer; font-weight:600; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
        .panel { border:1px solid #e4e7ec; border-radius:10px; padding:12px; }
        .badge { display:inline-block; border-radius:999px; padding:2px 8px; color:#fff; font-size:12px; margin-right:6px; }
        .critical { background:var(--critical); } .medium { background:var(--medium); } .style { background:var(--style); }
        pre { white-space:pre-wrap; background:#f8f9fc; border:1px solid #e4e7ec; border-radius:10px; padding:10px; max-height:260px; overflow:auto; }
        .score { font-weight:700; }
        .win { color:var(--ok); }
        .meta { color:#475467; font-size:13px; }
    </style>
</head>
<body>
    <div class='wrap'>
        <div class='card'>
            <h1 style='margin:0 0 8px 0;'>CVE Side-by-Side Review Demo</h1>
            <p class='meta' style='margin-top:0;'>Pick a CVE from the manifest and compare baseline vs reviewer outputs live.</p>
            <div class='row'>
                <select id='cveSelect'></select>
                <button onclick='runCompare()'>Run Comparison</button>
            </div>
            <div id='meta' class='meta' style='margin-top:10px;'></div>
            <pre id='diff'></pre>
            <div class='grid'>
                <div class='panel'>
                    <h3 style='margin-top:0;'>Baseline</h3>
                    <div id='baseline'></div>
                </div>
                <div class='panel'>
                    <h3 style='margin-top:0;'>Reviewer</h3>
                    <div id='reviewer'></div>
                </div>
            </div>
            <div id='winner' class='score' style='margin-top:10px;'></div>
        </div>
    </div>
<script>
async function loadCases() {
    const res = await fetch('/cve-cases');
    const data = await res.json();
    const sel = document.getElementById('cveSelect');
    sel.innerHTML = '';
    for (const c of data.cases) {
        const opt = document.createElement('option');
        opt.value = c.cve;
        opt.text = `${c.cve} | ${c.category} | ${c.repo}`;
        sel.appendChild(opt);
    }
}

function renderAction(containerId, obj) {
    const sev = (obj.action.severity || 'style').toLowerCase();
    const el = document.getElementById(containerId);
    el.innerHTML = `
        <div><span class='badge ${sev}'>${sev}</span><strong>line ${obj.action.line_number}</strong></div>
        <div class='meta'>${obj.action.message}</div>
        <div class='meta'>Fix: ${obj.action.suggested_fix}</div>
        <div class='meta'>Line hit: ${obj.result.line_hit} | Severity hit: ${obj.result.severity_hit}</div>
        <div class='score'>Reward: ${obj.result.reward}</div>
    `;
}

async function runCompare() {
    const cve = document.getElementById('cveSelect').value;
    const res = await fetch('/review-cve-compare', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({cve})
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    document.getElementById('meta').innerText = `${data.cve} | ${data.repo} | ${data.category} | ${data.filename}`;
    document.getElementById('diff').innerText = data.diff;
    renderAction('baseline', data.baseline);
    renderAction('reviewer', data.reviewer);

    const win = data.reviewer.result.reward > data.baseline.result.reward;
    document.getElementById('winner').innerHTML = win
        ? '<span class="win">Reviewer outperformed baseline on this CVE.</span>'
        : 'Reviewer did not outperform baseline for this case.';
}

loadCases();
</script>
</body>
</html>
"""


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m codereview_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn codereview_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
