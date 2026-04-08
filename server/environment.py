"""Code review reinforcement-learning environment for pull request diff analysis."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ReviewAction, ReviewObservation, ReviewState
except ImportError:
    from models import ReviewAction, ReviewObservation, ReviewState


class CodeReviewEnvironment(Environment):
    """Environment that supports multi-action code-review episodes with class-level metrics."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Load dataset and initialize episode state and running evaluation metrics."""
        data_path = Path(__file__).resolve().parents[1] / "data" / "prs.json"
        with data_path.open("r", encoding="utf-8") as f:
            self.prs: list[dict[str, Any]] = json.load(f)

        if not self.prs:
            raise ValueError("Dataset is empty. Expected entries in data/prs.json.")

        self.categories = sorted(
            {
                b.get("category", item.get("bug_category", "logic_bug"))
                for item in self.prs
                for b in item.get("bugs", [])
                if b.get("category")
            }
            | {"clean"}
        )
        self.class_stats: dict[str, dict[str, int]] = {
            c: {"tp": 0, "fp": 0, "fn": 0} for c in self.categories
        }

        self.current_pr_index: int = 0
        self.done: bool = False
        self.steps_taken: int = 0
        self.max_actions: int = 1
        self.found_bug_indices: set[int] = set()
        self.current_bugs: list[dict[str, Any]] = []
        self.reviewed_lines: set[int] = set()
        self.session_history: list[dict[str, Any]] = []
        self.current_file_context: str = ""
        self.current_repo_summary: str = ""

    def _normalize_pr(self, pr: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        """Normalize each PR into a bug-list representation and clean flag."""
        bugs = pr.get("bugs")
        if isinstance(bugs, list) and bugs:
            out = []
            for b in bugs:
                out.append(
                    {
                        "line": int(b["line"]),
                        "severity": str(b["severity"]),
                        "description": str(b["description"]),
                        "correct_fix": str(b["correct_fix"]),
                        "category": str(b.get("category", pr.get("bug_category", "logic_bug"))),
                    }
                )
            return out, bool(pr.get("is_clean", False))

        if int(pr.get("bug_line", 0)) == 0:
            return [], True

        return [
            {
                "line": int(pr["bug_line"]),
                "severity": str(pr["severity"]),
                "description": str(pr["bug_description"]),
                "correct_fix": str(pr["correct_fix"]),
                "category": str(pr.get("bug_category", "logic_bug")),
            }
        ], False

    def reset(self, seed: int | None = None, forced_index: int | None = None, **kwargs: Any) -> ReviewObservation:
        """Start a new episode and return the first observation for the selected PR."""
        if seed is not None:
            random.seed(seed)

        self.current_pr_index = forced_index if forced_index is not None else random.randrange(len(self.prs))
        self.done = False
        self.steps_taken = 0
        self.found_bug_indices = set()
        self.reviewed_lines = set()
        self.session_history = []

        pr = self.prs[self.current_pr_index]
        self.current_bugs, is_clean = self._normalize_pr(pr)
        self.max_actions = max(1, len(self.current_bugs) + 1)
        self.current_file_context = str(pr.get("file_context", pr["diff"]))
        self.current_repo_summary = str(
            pr.get(
                "repo_summary",
                f"Repository module containing {pr['filename']}; review focuses on security and correctness.",
            )
        )

        return ReviewObservation(
            diff=pr["diff"],
            filename=pr["filename"],
            episode_id=pr["id"],
            file_context=self.current_file_context,
            repo_summary=self.current_repo_summary,
            total_bugs=len(self.current_bugs),
            remaining_bugs=len(self.current_bugs),
            is_clean=is_clean,
            bug_categories=sorted({b["category"] for b in self.current_bugs}) if self.current_bugs else ["clean"],
        )

    def _keyword_set(self, text: str) -> set[str]:
        """Normalize text into lowercase keywords for simple overlap scoring."""
        return {w for w in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(w) > 2}

    def _infer_predicted_category(self, action: ReviewAction) -> str:
        """Infer predicted bug class from free-form rationale and fix text."""
        text = f"{action.message} {action.suggested_fix} {action.rationale}".lower()
        clean_markers = ["no bug", "clean", "approve", "no issue", "no fix required"]
        if action.line_number == 0 and any(m in text for m in clean_markers):
            return "clean"

        rules = {
            "sql_injection": ["sql", "parameterized", "query", "injection"],
            "xss": ["xss", "innerhtml", "sanitize", "escape"],
            "hardcoded_secret": ["secret", "apikey", "token", "credential", "hardcoded"],
            "off_by_one": ["off", "boundary", "index", "length"],
            "null_dereference": ["null", "none", "nil", "dereference"],
            "missing_auth": ["auth", "authorization", "admin", "permission"],
            "insecure_random": ["random", "predictable", "crypto", "nonce"],
            "path_traversal": ["path", "traversal", "basename", "directory"],
            "integer_overflow": ["overflow", "bounds", "int32", "wrap"],
            "logic_bug": ["logic", "condition", "branch", "retry"],
        }
        best = ("logic_bug", 0)
        for category, kws in rules.items():
            score = sum(1 for k in kws if k in text)
            if score > best[1]:
                best = (category, score)
        if action.line_number == 0 and best[1] == 0:
            return "clean"
        return best[0]

    def _reasoning_reward(self, action: ReviewAction, expected_category: str) -> tuple[float, dict[str, float]]:
        """Score message quality for conciseness, correctness, and hallucination resistance."""
        details = {"conciseness": 0.0, "correctness": 0.0, "non_hallucination": 0.0}
        text = f"{action.message} {action.suggested_fix} {action.rationale}".strip()
        words = text.split()

        if 6 <= len(words) <= 70:
            details["conciseness"] = 0.1
        else:
            details["conciseness"] = -0.05

        predicted = self._infer_predicted_category(action)
        details["correctness"] = 0.15 if predicted == expected_category else -0.05

        banned = ["buffer overflow", "race condition", "deserialization"]
        if any(k in text.lower() for k in banned) and expected_category not in {"integer_overflow", "logic_bug"}:
            details["non_hallucination"] = -0.1
        else:
            details["non_hallucination"] = 0.05

        return sum(details.values()), details

    def _update_class_stats(self, predicted: str, actual: str) -> None:
        """Update one-vs-rest TP/FP/FN counters for each bug class."""
        for c in self.categories:
            pred_c = predicted == c
            actual_c = actual == c
            if pred_c and actual_c:
                self.class_stats[c]["tp"] += 1
            elif pred_c and not actual_c:
                self.class_stats[c]["fp"] += 1
            elif actual_c and not pred_c:
                self.class_stats[c]["fn"] += 1

    def _class_metrics(self) -> dict[str, dict[str, float]]:
        """Return precision/recall/F1 by bug class from accumulated counters."""
        metrics: dict[str, dict[str, float]] = {}
        for c, s in self.class_stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            metrics[c] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return metrics

    def step(self, action: ReviewAction) -> tuple[ReviewObservation, float, bool, dict[str, Any]]:
        """Score one review action and continue until max actions or all bugs are found."""
        if self.done:
            raise ValueError("Episode is done. Call reset().")

        pr = self.prs[self.current_pr_index]
        is_clean = len(self.current_bugs) == 0
        self.steps_taken += 1

        reward = 0.0
        matched_idx = None
        line_distance = -1
        expected_category = "clean" if is_clean else "logic_bug"
        expected_severity = "style"

        if is_clean:
            if int(action.line_number) == 0:
                reward += 1.0
            else:
                reward -= 1.0
        else:
            open_bugs = [
                (idx, bug) for idx, bug in enumerate(self.current_bugs) if idx not in self.found_bug_indices
            ]
            if open_bugs:
                matched_idx, best_bug = min(
                    open_bugs,
                    key=lambda x: abs(int(action.line_number) - int(x[1]["line"])),
                )
                line_distance = abs(int(action.line_number) - int(best_bug["line"]))
                expected_category = str(best_bug["category"])
                expected_severity = str(best_bug["severity"])
                is_close_match = line_distance <= 2

                if int(action.line_number) == int(best_bug["line"]):
                    reward += 1.0
                    if expected_severity == "critical":
                        reward += 0.5
                    self.found_bug_indices.add(matched_idx)
                elif is_close_match:
                    reward += 0.4
                else:
                    # Far misses should not get neutral line credit in final scoring.
                    reward -= 0.3

                # Tighten final scoring: quality signals only count when localization is correct/near.
                if is_close_match and str(action.severity) == expected_severity:
                    reward += 0.5

                if is_close_match:
                    fix_keywords = self._keyword_set(str(best_bug["correct_fix"]))
                    action_keywords = self._keyword_set(str(action.suggested_fix))
                    overlap = len(fix_keywords.intersection(action_keywords))
                    if overlap >= 3:
                        reward += 0.3
                    elif overlap >= 1:
                        reward += 0.1

            if int(action.line_number) == 0:
                reward -= 0.5

        reasoning_score, reasoning_breakdown = self._reasoning_reward(action, expected_category)
        reward += reasoning_score

        predicted_category = self._infer_predicted_category(action)
        self._update_class_stats(predicted_category, expected_category)

        self.done = self.steps_taken >= self.max_actions or len(self.found_bug_indices) == len(self.current_bugs)

        if self.done and not is_clean:
            for idx, bug in enumerate(self.current_bugs):
                if idx not in self.found_bug_indices:
                    self._update_class_stats("clean", str(bug["category"]))

        self.reviewed_lines.add(int(action.line_number))
        self.session_history.append(
            {
                "step": self.steps_taken,
                "line_number": int(action.line_number),
                "severity": str(action.severity),
                "message": str(action.message),
                "suggested_fix": str(action.suggested_fix),
                "rationale": str(action.rationale),
                "reward": round(float(reward), 4),
                "predicted_category": predicted_category,
                "expected_category": expected_category,
            }
        )

        obs = ReviewObservation(
            diff=pr["diff"],
            filename=pr["filename"],
            episode_id=pr["id"],
            file_context=self.current_file_context,
            repo_summary=self.current_repo_summary,
            total_bugs=len(self.current_bugs),
            remaining_bugs=len(self.current_bugs) - len(self.found_bug_indices),
            is_clean=is_clean,
            bug_categories=sorted({b["category"] for b in self.current_bugs}) if self.current_bugs else ["clean"],
        )
        info = {
            "expected_bug_line": self.current_bugs[matched_idx]["line"] if matched_idx is not None else 0,
            "line_distance": line_distance,
            "expected_severity": expected_severity,
            "expected_category": expected_category,
            "predicted_category": predicted_category,
            "reasoning_breakdown": reasoning_breakdown,
            "found_bug_count": len(self.found_bug_indices),
            "remaining_bugs": len(self.current_bugs) - len(self.found_bug_indices),
            "steps_taken": self.steps_taken,
            "max_actions": self.max_actions,
            "reviewed_lines": sorted(self.reviewed_lines),
            "session_history": self.session_history,
            "is_clean": is_clean,
            "class_metrics": self._class_metrics(),
            "current_pr_index": self.current_pr_index,
        }
        return obs, reward, self.done, info

    @property
    def state(self) -> ReviewState:
        """Return current environment state including progress through multi-action episode."""
        return ReviewState(
            current_pr_index=self.current_pr_index,
            done=self.done,
            steps_taken=self.steps_taken,
            max_actions=self.max_actions,
            found_bug_count=len(self.found_bug_indices),
            total_bug_count=len(self.current_bugs),
            reviewed_lines=sorted(self.reviewed_lines),
            session_history=self.session_history,
        )
