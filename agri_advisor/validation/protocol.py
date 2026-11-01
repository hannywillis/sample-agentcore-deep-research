"""Expert Validation Protocol (WP1 Task 1.7 / D1.7).

Decides whether a recommendation can be delivered as-is or must be escalated
to a human expert, records the reasons, and queues escalated items for review.
Expert verdicts are stored alongside so inter-rater agreement and the effect of
validation on farmer trust (WP3 comparative design) can be analysed later.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from agri_advisor.config import Settings, get_settings


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ValidationResult:
    requires_expert: bool
    risk_level: RiskLevel
    reasons: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    out_of_domain: bool = False
    review_id: str | None = None
    protocol_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d


class ExpertValidationProtocol:
    def __init__(
        self, config_path: Path | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        path = config_path or self.settings.expert_validation_config
        self.config: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        self.version = str(self.config.get("version", "unversioned"))
        self.t = self.config.get("thresholds", {})
        self._risk_patterns = {
            group: self._compile(words)
            for group, words in self.config.get("high_risk_keywords", {}).items()
        }
        self._domain = self._compile(self.config.get("domain_terms", []))

    @staticmethod
    def _compile(words: list[str]) -> re.Pattern[str] | None:
        if not words:
            return None
        alt = "|".join(
            re.escape(w.lower()) for w in sorted(words, key=len, reverse=True)
        )
        return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)

    # --------------------------------------------------------------- checks
    def is_agricultural(self, text: str) -> bool:
        return bool(self._domain and self._domain.search(text))

    def risk_keywords(self, text: str) -> dict[str, list[str]]:
        hits: dict[str, list[str]] = {}
        for group, pat in self._risk_patterns.items():
            if pat:
                found = sorted({m.group(0).lower() for m in pat.finditer(text)})
                if found:
                    hits[group] = found
        return hits

    def evaluate(
        self,
        query: str,
        retrieval_query: str,
        evidence_scores: list[float],
        predictions: dict[str, dict[str, Any]] | None = None,
        advisory_text: str = "",
        insufficient_evidence: bool = False,
    ) -> ValidationResult:
        """Apply every rule; return the combined decision.

        ``predictions`` maps task -> ``PredictionResult.to_dict()``.
        """
        reasons: list[str] = []
        rules: list[str] = []
        risk = RiskLevel.LOW

        def bump(level: RiskLevel) -> None:
            nonlocal risk
            order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
            if order.index(level) > order.index(risk):
                risk = level

        # 1. Evidence sufficiency ------------------------------------------
        strong = [s for s in evidence_scores if s >= self.t.get("min_top_score", 0.15)]
        weak_evidence = len(strong) < self.t.get("min_evidence_count", 1)
        if weak_evidence or insufficient_evidence:
            rules.append("insufficient_evidence")
            reasons.append("No sufficiently relevant validated evidence was found.")
            bump(RiskLevel.MEDIUM)

        # 2. Out-of-domain ---------------------------------------------------
        out_of_domain = not (
            self.is_agricultural(query) or self.is_agricultural(retrieval_query)
        ) and (weak_evidence or not predictions)
        if out_of_domain:
            rules.append("out_of_domain")
            reasons.append(
                "Question appears to be outside agricultural advisory scope."
            )
            bump(RiskLevel.MEDIUM)

        # 3. High-risk topics ------------------------------------------------
        hits = self.risk_keywords(f"{query} {retrieval_query} {advisory_text}")
        for group, words in hits.items():
            rules.append(f"high_risk_keyword:{group}")
            reasons.append(
                f"High-risk topic ({group.replace('_', ' ')}): {', '.join(words)}."
            )
            bump(RiskLevel.HIGH)

        # 4. Predictive uncertainty & risk outcomes --------------------------
        min_conf = self.t.get("min_prediction_confidence", 0.35)
        esc_levels = set(self.t.get("escalate_confidence_levels", ["low"]))
        high_risk_outcomes = self.config.get("high_risk_predictions", {})
        for task, pred in (predictions or {}).items():
            unc = pred.get("uncertainty", {})
            if unc.get("level") in esc_levels or unc.get("score", 1.0) < min_conf:
                rules.append(f"low_confidence:{task}")
                reasons.append(
                    f"Low confidence in {task.replace('_', ' ')} prediction "
                    f"({unc.get('level')}, score {unc.get('score')})."
                )
                bump(RiskLevel.MEDIUM)
            if pred.get("value") in high_risk_outcomes.get(task, []):
                rules.append(f"high_risk_outcome:{task}")
                reasons.append(
                    f"Predicted {task.replace('_', ' ')} is {pred.get('value')}."
                )
                bump(RiskLevel.HIGH)
            if (
                self.t.get("escalate_unvalidated_models", True)
                and pred.get("model_status") != "validated"
            ):
                rules.append(f"unvalidated_model:{task}")
                reasons.append(
                    f"{task.replace('_', ' ').capitalize()} model is not yet validated "
                    "on field data."
                )
                bump(RiskLevel.MEDIUM)

        requires = bool(rules)
        return ValidationResult(
            requires_expert=requires,
            risk_level=risk,
            reasons=list(dict.fromkeys(reasons)),
            triggered_rules=list(dict.fromkeys(rules)),
            out_of_domain=out_of_domain,
            protocol_version=self.version,
        )

    # ---------------------------------------------------------- review queue
    def enqueue(self, recommendation: dict[str, Any]) -> str:
        """Append an escalated recommendation to the expert review queue."""
        review_id = str(uuid.uuid4())
        if not self.config.get("review", {}).get("queue_enabled", True):
            return review_id
        path = self.settings.review_queue_path
        path.parent.mkdir(parents=True, exist_ok=True)
        item = {
            "review_id": review_id,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "protocol_version": self.version,
            "status": "pending",
            "validators_required": self.config.get("review", {}).get(
                "validators_per_item", 2
            ),
            "rubric": self.config.get("review", {}).get("rubric", []),
            "recommendation": recommendation,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return review_id

    def record_verdict(
        self,
        review_id: str,
        validator_id: str,
        verdict: str,
        scores: dict[str, int] | None = None,
        comment: str = "",
    ) -> dict[str, Any]:
        allowed = self.config.get("review", {}).get("verdicts", [])
        if allowed and verdict not in allowed:
            raise ValueError(f"verdict must be one of {allowed}")
        entry = {
            "type": "verdict",
            "review_id": review_id,
            "validator_id": validator_id,
            "verdict": verdict,
            "scores": scores or {},
            "comment": comment,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.settings.review_queue_path.with_name("expert_verdicts.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry
