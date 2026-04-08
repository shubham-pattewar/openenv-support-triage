"""Deterministic graders for the customer support triage tasks."""

from __future__ import annotations

from typing import Any


def _extract_score(candidate: Any) -> float | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        if "grader_score" in candidate and candidate["grader_score"] is not None:
            return float(candidate["grader_score"])
        if "metadata" in candidate and isinstance(candidate["metadata"], dict):
            value = candidate["metadata"].get("final_score")
            if value is not None:
                return float(value)
        observation = candidate.get("observation")
        if observation is not None:
            return _extract_score(observation)
        return None

    for attr in ("grader_score", "final_score"):
        value = getattr(candidate, attr, None)
        if value is not None:
            return float(value)

    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("final_score")
        if value is not None:
            return float(value)

    observation = getattr(candidate, "observation", None)
    if observation is not None:
        return _extract_score(observation)

    return None


def trajectory(state: Any, submission: Any) -> float:
    """Return the task score for the finished episode."""

    score = _extract_score(submission)
    if score is None:
        score = _extract_score(state)
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))
