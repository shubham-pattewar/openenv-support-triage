"""Deterministic graders for the customer support triage tasks."""

from __future__ import annotations

from typing import Any

SCORE_EPSILON = 0.001


def _extract_score(candidate: Any) -> float | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        if "metadata" in candidate and isinstance(candidate["metadata"], dict):
            raw_score = candidate["metadata"].get("raw_score")
            if raw_score is not None:
                return float(raw_score)
            value = candidate["metadata"].get("final_score")
            if value is not None:
                return float(value)
        if "grader_score" in candidate and candidate["grader_score"] is not None:
            return float(candidate["grader_score"])
        observation = candidate.get("observation")
        if observation is not None:
            return _extract_score(observation)
        return None

    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        raw_score = metadata.get("raw_score")
        if raw_score is not None:
            return float(raw_score)

    for attr in ("grader_score", "final_score"):
        value = getattr(candidate, attr, None)
        if value is not None:
            return float(value)

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
        return SCORE_EPSILON
    score = max(0.0, min(1.0, float(score)))
    return SCORE_EPSILON + ((1.0 - (2 * SCORE_EPSILON)) * score)
