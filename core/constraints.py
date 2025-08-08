from __future__ import annotations

# core/constraints.py - Complete Constitutional System
"""
Constitutional DSL Parser and Constraint Enforcement Engine
Transforms human-readable emotional boundaries into executable constraints and enforces them in real-time
"""
# ------------------------------------

# imports (cleaned)
import re
import json
import yaml
import time
import math
import logging
import threading
import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Deque, Pattern  # <-- add Deque, Pattern
from dsl.parser import ConditionEngine
import jsonschema


class ConstraintType(Enum):
    EMOTIONAL_BOUNDARY = "emotional_boundary"
    PRIVACY_RULE = "privacy_rule"
    INTERACTION_LIMIT = "interaction_limit"
    BEHAVIORAL_GUIDELINE = "behavioral_guideline"


class ActionType(Enum):
    BLOCK = "block"
    ACKNOWLEDGE = "acknowledge_emotion"
    ANONYMIZE = "anonymize"
    SUGGEST_BREAK = "suggest_break"
    REDIRECT = "redirect"
    LOG_ONLY = "log_only"


@dataclass
class ConstraintViolation:
    constraint_name: str
    constraint_type: ConstraintType
    violation_reason: str
    suggested_action: ActionType
    severity: str
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class CompiledConstraint:
    name: str
    constraint_type: ConstraintType
    # keep original string for auditability + add compiled Pattern
    pattern_str: Optional[str] = None
    pattern: Optional[re.Pattern] = None
    condition: Optional[str] = None
    action: ActionType = ActionType.LOG_ONLY
    response_template: Optional[str] = None
    severity: str = "medium"
    enabled: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationContext:
    """Context information for constraint evaluation"""
    user_id: str
    user_input: str
    ai_response: Optional[str] = None
    session_duration: float = 0.0
    emotional_intensity: float = 0.0
    privacy_score: float = 0.0
    interaction_count: int = 0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class _SessionState:
    start_time: datetime
    last_interaction: datetime
    interaction_count: int = 0
    daily_interactions: int = 0
    total_emotional_intensity: float = 0.0
    last_reset_date: datetime.date = field(default_factory=lambda: datetime.now().date())
    # Rolling/EMA metrics
    ema_emotional_intensity: float = 0.0
    ema_alpha: float = 0.2  # 20% weight to the latest point by default
    # Sliding window of interaction timestamps for rate metrics
    events: Deque[datetime] = field(default_factory=lambda: deque(maxlen=4096))

@dataclass
class ConstraintEvaluationResult:
    """Result of constraint evaluation"""
    is_allowed: bool
    violations: List[ConstraintViolation]
    actions: List[ActionType]
    suggested_response: Optional[str] = None
    processing_route: str = "local"
    confidence_score: float = 1.0
    evaluation_time_ms: float = 0.0
    session_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PIIMatch:
    kind: str
    value: str
    span: Tuple[int, int]

class ConstitutionalParser:
    """
    Parses and compiles constitutional constraints from YAML into executable rules
    """

    def __init__(self):
        self.schema = self._create_constraint_schema()
        self.compiled_constraints: List[CompiledConstraint] = []
        self.emotion_patterns = self._load_emotion_patterns()

    def _create_constraint_schema(self) -> Dict:
        """JSON Schema for validating constraint YAML"""
        return {
            "type": "object",
            "properties": {
                "constraints": {
                    "type": "object",
                    "properties": {
                        "emotional_boundaries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "pattern": {"type": "string"},
                                    "response": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                                    "enabled": {"type": "boolean"}
                                },
                                "required": ["name", "pattern", "response"]
                            }
                        },
                        "privacy_rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "condition": {"type": "string"},
                                    "action": {"type": "string"},
                                    "severity": {"type": "string"}
                                },
                                "required": ["name", "condition", "action"]
                            }
                        },
                        "interaction_limits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "condition": {"type": "string"},
                                    "action": {"type": "string"},
                                    "threshold": {"type": "number"}
                                },
                                "required": ["name", "condition", "action"]
                            }
                        },
                        "behavioral_guidelines": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "required": {"type": "boolean"},
                                    "validation": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["name", "description"]
                            }
                        }
                    }
                }
            },
            "required": ["constraints"]
        }

    def _load_emotion_patterns(self) -> Dict[str, str]:
        """Pre-defined emotion detection patterns"""
        return {
            "anger": r"\b(angry|furious|rage|mad|pissed|irritated|annoyed)\b",
            "sadness": r"\b(sad|depressed|down|miserable|heartbroken|devastated)\b",
            "anxiety": r"\b(anxious|worried|nervous|stressed|overwhelmed|panic)\b",
            "joy": r"\b(happy|joyful|excited|thrilled|elated|cheerful)\b",
            "fear": r"\b(afraid|scared|terrified|frightened|worried|anxious)\b",
            "disgust": r"\b(disgusted|revolted|sick|appalled|repulsed)\b",
            "surprise": r"\b(surprised|shocked|amazed|astonished|stunned)\b"
        }

    def parse_constraints(self, yaml_str: str) -> List[CompiledConstraint]:
        """Parse YAML constraints and compile into executable format"""
        try:
            constraints_data = yaml.safe_load(yaml_str)
            jsonschema.validate(constraints_data, self.schema)
            compiled = self._compile_constraints(constraints_data)
            self.compiled_constraints = compiled
            return compiled
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
        except jsonschema.ValidationError as e:
            raise ValueError(f"Constraint validation failed: {e.message}")
        except Exception as e:
            raise ValueError(f"Failed to parse constraints: {e}")

    def _compile_constraints(self, data: Dict) -> List[CompiledConstraint]:
        """Compile parsed YAML into executable constraint objects"""
        compiled: List[CompiledConstraint] = []
        constraints = data.get("constraints", {})

        # Emotional boundaries
        for boundary in constraints.get("emotional_boundaries", []):
            pat_str = boundary.get("pattern")
            pat = None
            if pat_str:
                try:
                    pat = re.compile(pat_str, re.IGNORECASE)
                except re.error as e:
                    raise ValueError(f"[{boundary.get('name', '<unnamed>')}] invalid regex pattern: {e}")
            compiled.append(CompiledConstraint(
                name=boundary["name"],
                constraint_type=ConstraintType.EMOTIONAL_BOUNDARY,
                pattern_str=pat_str,
                pattern=pat,
                action=ActionType(boundary.get("response", "acknowledge_emotion")),
                severity=boundary.get("severity", "medium"),
                enabled=boundary.get("enabled", True),
                response_template=boundary.get("template"),
                metadata={"category": "emotional", "auto_generated": False}
            ))

            # Privacy rules (unchanged structure)
        for rule in constraints.get("privacy_rules", []):
            compiled.append(CompiledConstraint(
                name=rule["name"],
                constraint_type=ConstraintType.PRIVACY_RULE,
                condition=rule["condition"],
                action=ActionType(rule["action"]),
                severity=rule.get("severity", "high"),
                enabled=rule.get("enabled", True),
                metadata={"category": "privacy", "compliance_required": True}
            ))

        # Compile interaction limits
        for limit in constraints.get("interaction_limits", []):
            compiled.append(CompiledConstraint(
                name=limit["name"],
                constraint_type=ConstraintType.INTERACTION_LIMIT,
                condition=limit["condition"],
                action=ActionType(limit["action"]),
                severity=limit.get("severity", "medium"),
                enabled=limit.get("enabled", True),
                metadata={"category": "interaction", "threshold": limit.get("threshold")}
            ))

        # Compile behavioral guidelines
        for guideline in constraints.get("behavioral_guidelines", []):
            compiled.append(CompiledConstraint(
                name=guideline["name"],
                constraint_type=ConstraintType.BEHAVIORAL_GUIDELINE,
                condition=guideline.get("description"),
                action=ActionType.LOG_ONLY,
                severity="low",
                enabled=guideline.get("required", False),
                metadata={
                    "category": "behavioral",
                    "validation_rules": guideline.get("validation", [])
                }
            ))

        return compiled

    def get_constraint_by_name(self, name: str) -> Optional[CompiledConstraint]:
        """Retrieve a specific constraint by name"""
        for constraint in self.compiled_constraints:
            if constraint.name == name:
                return constraint
        return None

    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[CompiledConstraint]:
        """Get all constraints of a specific type"""
        return [c for c in self.compiled_constraints if c.constraint_type == constraint_type]

    def validate_constraint_syntax(self, yaml_str: str) -> Tuple[bool, List[str]]:
        """Validate constraint syntax without compiling"""
        errors = []
        try:
            data = yaml.safe_load(yaml_str)
            jsonschema.validate(data, self.schema)
            return len(errors) == 0, errors
        except Exception as e:
            errors.append(str(e))
            return False, errors


class PrivacyAnalyzer:
    """
    Analyzes privacy sensitivity of text input.
    - Precompiles regexes for performance
    - Supports detection, scoring, extraction, and redaction
    """

    def __init__(
        self,
        *,
        # Weights for scoring (sum can exceed 1; we cap later)
        weight_per_pii_hit: float = 0.30,
        weight_per_sensitive_keyword: float = 0.20,
        max_score: float = 1.0,
        # Allow caller to extend/override
        extra_keywords: Optional[List[str]] = None,
        extra_patterns: Optional[Dict[str, str]] = None
    ) -> None:
        self.max_score = max_score
        self.weight_per_pii_hit = weight_per_pii_hit
        self.weight_per_sensitive_keyword = weight_per_sensitive_keyword

        # Base patterns (reasonably strict, not perfect)
        base_patterns: Dict[str, str] = {
            "email": r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
            # Accept common separators and country codes; keep it moderate to reduce FPs
            "phone": r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}))\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            # 13–19 digits with common separators; validated via Luhn below
            "credit_card": r"\b(?:\d[ -]?){13,19}\b",
            # Basic IPv4
            "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            # IBAN (very rough: 15–34 alnum starting with 2 letters)
            "iban": r"\b[A-Z]{2}[0-9A-Z]{13,32}\b",
        }
        if extra_patterns:
            base_patterns.update(extra_patterns)

        # Precompile once
        flags = re.IGNORECASE
        self._compiled_patterns: Dict[str, Pattern[str]] = {
            kind: re.compile(pattern, flags) for kind, pattern in base_patterns.items()
        }

        # Keywords (lowercased matching)
        default_keywords = [
            "password",
            "social security",
            "credit card",
            "bank account",
            "routing number",
            "medical",
            "diagnosis",
            "prescription",
            "passport",
            "driver license",
            "dob",
            "date of birth",
        ]
        if extra_keywords:
            default_keywords.extend(extra_keywords)
        # normalize + de-dupe
        self._keywords = sorted(set(k.lower() for k in default_keywords))

    # ---------- Public API ----------

    def analyze_privacy_score(self, text: str) -> float:
        """
        Calculate privacy sensitivity score (0..max_score).
        Score is additive across:
          - distinct PII matches (by type)
          - sensitive keyword hits
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.0

        # PII pattern hits
        hits = self.detect_pii(text)
        if hits:
            # count distinct matches, not just types, to reflect density
            score += self.weight_per_pii_hit * len(hits)

        # Keyword hits (unique keywords)
        keyword_hits = sum(1 for kw in self._keywords if kw in text_lower)
        score += self.weight_per_sensitive_keyword * keyword_hits

        return min(score, self.max_score)

    def contains_pii(self, text: str) -> bool:
        """True if any PII is detected."""
        return bool(self.detect_pii(text))

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        Return a list of PIIMatch(kind, value, span).
        Applies extra validation for credit cards (Luhn).
        """
        matches: List[PIIMatch] = []
        if not text:
            return matches

        for kind, regex in self._compiled_patterns.items():
            for m in regex.finditer(text):
                val = m.group(0)
                # Extra validation to curb obvious false positives
                if kind == "credit_card":
                    digits = re.sub(r"[^\d]", "", val)
                    # Typical card length + Luhn
                    if not (13 <= len(digits) <= 19 and self._luhn_ok(digits)):
                        continue
                if kind == "ipv4":
                    if not self._ipv4_ok(val):
                        continue
                matches.append(PIIMatch(kind=kind, value=val, span=m.span()))
        return matches

    def redact(self, text: str, replacement: str = "<redacted>") -> str:
        """
        Redact detected PII in the text, returning a new string.
        """
        if not text:
            return text

        matches = self.detect_pii(text)
        if not matches:
            return text

        # Replace from right to left to keep spans stable
        redacted = text
        for m in sorted(matches, key=lambda x: x.span[0], reverse=True):
            start, end = m.span
            redacted = redacted[:start] + replacement + redacted[end:]
        return redacted

    # ---------- Helpers ----------

    @staticmethod
    def _luhn_ok(number: str) -> bool:
        """Luhn checksum for credit-card-like numbers."""
        s = 0
        alt = False
        for ch in reversed(number):
            d = ord(ch) - 48  # '0' -> 0
            if d < 0 or d > 9:
                return False
            if alt:
                d *= 2
                if d > 9:
                    d -= 9
            s += d
            alt = not alt
        return (s % 10) == 0

    @staticmethod
    def _ipv4_ok(addr: str) -> bool:
        """Basic IPv4 sanity check (0-255 per octet)."""
        try:
            parts = addr.split(".")
            if len(parts) != 4:
                return False
            for p in parts:
                if not p.isdigit():
                    return False
                n = int(p)
                if n < 0 or n > 255:
                    return False
            return True
        except Exception:
            return False


class EmotionAnalyzer:
    """Analyzes emotional content and intensity"""

    def __init__(self):
        self.emotion_patterns = {
            "anger": r"\b(angry|furious|rage|mad|pissed|irritated|annoyed|hate)\b",
            "sadness": r"\b(sad|depressed|down|miserable|heartbroken|devastated|crying)\b",
            "anxiety": r"\b(anxious|worried|nervous|stressed|overwhelmed|panic|afraid)\b",
            "joy": r"\b(happy|joyful|excited|thrilled|elated|cheerful|love)\b",
            "fear": r"\b(afraid|scared|terrified|frightened|worried|anxious)\b"
        }

        self.intensity_multipliers = {
            "very": 1.5, "extremely": 2.0, "incredibly": 1.8, "really": 1.3,
            "so": 1.4, "absolutely": 1.7, "completely": 1.6, "totally": 1.5
        }

    def analyze_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity score (0-1)"""
        text_lower = text.lower()
        base_intensity = 0.0
        multiplier = 1.0

        # Check for emotional patterns
        for emotion, pattern in self.emotion_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                base_intensity += 0.2 * len(matches)

        # Check for intensity multipliers
        for intensifier, mult in self.intensity_multipliers.items():
            if intensifier in text_lower:
                multiplier = max(multiplier, mult)

        # Check for caps (indicates shouting/intensity)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            multiplier *= 1.3

        # Check for exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            multiplier += 0.1 * min(exclamation_count, 5)

        final_intensity = min(base_intensity * multiplier, 1.0)
        return final_intensity

    def detect_primary_emotion(self, text: str) -> Optional[str]:
        """Detect the primary emotion in text"""
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, pattern in self.emotion_patterns.items():
            matches = re.findall(pattern, text_lower)
            emotion_scores[emotion] = len(matches)

        if not emotion_scores or max(emotion_scores.values()) == 0:
            return None

        return max(emotion_scores, key=emotion_scores.get)


class SessionTracker:
    """
    Tracks user session information for interaction limits and analytics.

    Upgrades:
      - Thread-safe with RLock
      - Configurable daily reset hour (default midnight local)
      - EMAs for emotional intensity
      - Rolling rate metrics (1m/5m/1h) using timestamp deque
      - Pruning for inactive sessions and global cap
      - Clean end_session + touch + manual reset
    """

    def __init__(
        self,
        day_reset_hour: int = 0,
        max_sessions: int = 10000,
        event_window_seconds: int = 3600,
        ema_alpha: float = 0.2,
    ):
        """
        :param day_reset_hour: hour of day (0–23) when daily counters reset
        :param max_sessions: maximum sessions kept in memory
        :param event_window_seconds: size of the sliding window for rate metrics
        :param ema_alpha: EMA smoothing factor for emotional intensity
        """
        if not (0 <= day_reset_hour <= 23):
            raise ValueError("day_reset_hour must be 0..23")

        self._lock = threading.RLock()
        self._sessions: Dict[str, _SessionState] = {}
        self._day_reset_hour = day_reset_hour
        self._max_sessions = max_sessions
        self._event_window = timedelta(seconds=event_window_seconds)
        self._ema_alpha = ema_alpha

    # ---------- internal utilities ----------

    def _now(self) -> datetime:
        # Naive local time to match rest of your codebase
        return datetime.now()

    def _maybe_reset_daily(self, session: _SessionState, now: datetime) -> None:
        """
        Reset daily counters if past the configured reset boundary.
        Boundary is the most recent day at `day_reset_hour`.
        """
        # Determine current 'reset day' cutoff
        cutoff = now.replace(hour=self._day_reset_hour, minute=0, second=0, microsecond=0)
        if now.hour < self._day_reset_hour:
            # before reset hour -> today's cutoff is actually yesterday's reset moment
            cutoff = cutoff - timedelta(days=1)

        # If last reset happened before the active cutoff day, reset daily counters
        if session.last_reset_date < cutoff.date():
            session.daily_interactions = 0
            session.last_reset_date = cutoff.date()

    def _update_ema(self, session: _SessionState, value: float) -> None:
        a = session.ema_alpha
        session.ema_emotional_intensity = (
            a * value + (1.0 - a) * session.ema_emotional_intensity
        )

    def _prune_old_events(self, session: _SessionState, now: datetime) -> None:
        """Remove timestamps outside the rolling window."""
        cutoff = now - self._event_window
        ev = session.events
        while ev and ev[0] < cutoff:
            ev.popleft()

    def _compute_rates(self, session: _SessionState, now: datetime) -> Dict[str, float]:
        """
        Compute interaction rates over 1m/5m/1h windows.
        We reuse the event deque (which already prunes > event_window).
        """
        # Ensure events are pruned to the configured window
        self._prune_old_events(session, now)

        def count_since(delta: timedelta) -> int:
            start = now - delta
            # events are in chronological order; linear scan from right is okay
            # for small windows; for large loads you could binary search.
            c = 0
            for t in reversed(session.events):
                if t >= start:
                    c += 1
                else:
                    break
            return c

        rate_1m = count_since(timedelta(minutes=1))
        rate_5m = count_since(timedelta(minutes=5))
        rate_1h = count_since(timedelta(hours=1))

        return {
            "rate_1m": float(rate_1m),
            "rate_5m": float(rate_5m),
            "rate_1h": float(rate_1h),
        }

    def _ensure_capacity(self) -> None:
        """Evict least-recently-touched sessions if over capacity."""
        if len(self._sessions) <= self._max_sessions:
            return
        # Evict by oldest last_interaction first
        victims = sorted(self._sessions.items(), key=lambda kv: kv[1].last_interaction)
        to_evict = len(self._sessions) - self._max_sessions
        for i in range(to_evict):
            self._sessions.pop(victims[i][0], None)

    def _get_or_create(self, user_id: str, now: datetime) -> _SessionState:
        sess = self._sessions.get(user_id)
        if sess is None:
            sess = _SessionState(
                start_time=now,
                last_interaction=now,
                ema_alpha=self._ema_alpha,
            )
            self._sessions[user_id] = sess
            self._ensure_capacity()
        return sess

    # ---------- public API ----------

    def update_session(self, user_id: str, context: "EvaluationContext") -> None:
        """
        Update session information for a user and reflect results back into context.
        """
        with self._lock:
            now = self._now()
            sess = self._get_or_create(user_id, now)

            # Daily reset if needed
            self._maybe_reset_daily(sess, now)

            # Increment counters and aggregates
            sess.last_interaction = now
            sess.interaction_count += 1
            sess.daily_interactions += 1

            # Record event for rate metrics
            sess.events.append(now)
            self._prune_old_events(sess, now)

            # Defensive: context may not have emotional_intensity *yet*; treat missing as 0
            emotional_intensity = getattr(context, "emotional_intensity", 0.0) or 0.0
            sess.total_emotional_intensity += float(emotional_intensity)
            self._update_ema(sess, float(emotional_intensity))

            # Reflect session info back to context
            context.session_duration = (now - sess.start_time).total_seconds() / 60.0
            context.interaction_count = sess.interaction_count

    def touch(self, user_id: str) -> None:
        """Bump last_interaction without counting as a full interaction."""
        with self._lock:
            now = self._now()
            sess = self._get_or_create(user_id, now)
            self._maybe_reset_daily(sess, now)
            sess.last_interaction = now

    def reset_daily(self, user_id: str) -> None:
        """Force daily counter reset (rarely needed, but handy for admin/testing)."""
        with self._lock:
            if user_id in self._sessions:
                sess = self._sessions[user_id]
                sess.daily_interactions = 0
                sess.last_reset_date = self._now().date()

    def get_session_info(self, user_id: str) -> Dict[str, Any]:
        """Get current session information for a user. Returns {} if session not found."""
        with self._lock:
            sess = self._sessions.get(user_id)
            if not sess:
                return {}

            now = self._now()
            self._maybe_reset_daily(sess, now)
            rates = self._compute_rates(sess, now)

            duration_min = (now - sess.start_time).total_seconds() / 60.0
            time_since_last = (now - sess.last_interaction).total_seconds()
            avg_intensity = (
                sess.total_emotional_intensity / max(sess.interaction_count, 1)
            )

            return {
                "session_duration_min": duration_min,
                "interaction_count": sess.interaction_count,
                "daily_interactions": sess.daily_interactions,
                "avg_emotional_intensity": avg_intensity,
                "ema_emotional_intensity": sess.ema_emotional_intensity,
                "time_since_last_sec": time_since_last,
                **rates,
            }

    def end_session(self, user_id: str) -> Dict[str, Any]:
        """
        End and remove a session; returns a summary snapshot for logging/audit.
        Returns {} if session does not exist.
        """
        with self._lock:
            sess = self._sessions.pop(user_id, None)
            if not sess:
                return {}

            now = self._now()
            duration_min = (now - sess.start_time).total_seconds() / 60.0
            avg_intensity = sess.total_emotional_intensity / max(sess.interaction_count, 1)

            return {
                "session_duration_min": duration_min,
                "interaction_count": sess.interaction_count,
                "daily_interactions": sess.daily_interactions,
                "avg_emotional_intensity": avg_intensity,
                "ema_emotional_intensity": sess.ema_emotional_intensity,
                "ended_at": now.isoformat(),
            }

    def prune_inactive(self, inactive_seconds: int = 86_400) -> int:
        """
        Remove sessions that haven't been touched for `inactive_seconds`.
        Returns number of pruned sessions.
        """
        with self._lock:
            now = self._now()
            cutoff = now - timedelta(seconds=inactive_seconds)
            victims = [uid for uid, s in self._sessions.items() if s.last_interaction < cutoff]
            for uid in victims:
                self._sessions.pop(uid, None)
            return len(victims)


class ConstraintEnforcer:
    """Main constraint enforcement engine"""

    def __init__(
            self,
            constitutional_parser: ConstitutionalParser,
            foundation_root: Optional[Union[str, Path]] = None,
            *,
            privacy_analyzer: Optional[PrivacyAnalyzer] = None,
            emotion_analyzer: Optional[EmotionAnalyzer] = None,
            session_tracker: Optional[SessionTracker] = None,
            logger: Optional[logging.Logger] = None,
    ) -> None:
        self.parser = constitutional_parser

        # Dependencies (DI-friendly for tests)
        self.privacy_analyzer = privacy_analyzer or PrivacyAnalyzer()
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.session_tracker = session_tracker or SessionTracker()

        # DSL condition engine for constraint evaluation
        self._cond = ConditionEngine()

        self.violation_log: List[ConstraintViolation] = []

        # Logger (don’t double-add handlers; let caller configure level/handlers)
        self.logger = logger or logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Resolve foundation root
        if foundation_root is None:
            # project root = two levels up from this file
            self.foundation_root = Path(__file__).resolve().parents[1]
        else:
            self.foundation_root = Path(foundation_root).expanduser().resolve()
            if not self.foundation_root.exists():
                raise FileNotFoundError(f"foundation_root not found: {self.foundation_root}")

        # Load responses after root is known
        self.response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, str]:
        """
        Load response templates for different constraint violations.
        Falls back to internal defaults if no external config is found.
        """
        default_templates = {
            # keep both "block" and "privacy_block" keys so ActionType.BLOCK works
            "block": "I can't process that request as it contains sensitive or unsafe information.",
            "privacy_block": "I can't process that request as it contains sensitive information that should remain private.",
            "acknowledge_emotion": "I can sense you're experiencing strong emotions. I'm here to help you work through this.",
            "suggest_break": "It looks like you've been in a long session. Would you like to take a break?",
            "anonymize": "I'll help you with that, but I need to anonymize some sensitive information first.",
            "redirect": "Let me help you approach this topic in a different way.",
            "log_only": "Noted.",
            "default": "I need to handle this request carefully to ensure your wellbeing and privacy."
        }

        config_path = self.foundation_root / "data" / "response_templates.json"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    external = json.load(f)
                if isinstance(external, dict):
                    # normalize keys -> lower; only accept string values
                    external_norm = {str(k).lower(): v for k, v in external.items() if isinstance(v, str)}
                    default_templates.update(external_norm)
                    self.logger.info(f"Loaded external response templates from {config_path}")
                else:
                    self.logger.warning(f"Ignoring templates at {config_path}: not a JSON object")
            except Exception as e:
                self.logger.warning(f"Failed to load external response templates: {e}")

        return default_templates

    def evaluate_constraints(self, context: EvaluationContext) -> ConstraintEvaluationResult:
        """Evaluate all active constraints against the given context"""
        start_time = time.time()
        violations: List[ConstraintViolation] = []

        # 1) Analyze context first so session metrics include this interaction’s values
        context.privacy_score = self.privacy_analyzer.analyze_privacy_score(context.user_input)
        context.emotional_intensity = self.emotion_analyzer.analyze_emotional_intensity(context.user_input)

        # 2) Then update session tracking (EMAs/rates see fresh intensity)
        self.session_tracker.update_session(context.user_id, context)

        # 3) Evaluate each constraint type
        violations.extend(self._evaluate_emotional_boundaries(context))
        violations.extend(self._evaluate_privacy_rules(context))
        violations.extend(self._evaluate_interaction_limits(context))

        # 4) Log violations
        if violations:
            self.violation_log.extend(violations)

        # 5) Rank actions (most restrictive first)
        action_order = ["block", "anonymize", "redirect", "suggest_break", "acknowledge_emotion", "log_only"]
        actions = sorted(
            {v.suggested_action for v in violations},
            key=lambda a: action_order.index(a.value) if a.value in action_order else len(action_order)
        )

        # Allowed unless we hit a BLOCK
        is_allowed = ActionType.BLOCK not in {a for a in actions}

        suggested_response = self._generate_suggested_response(violations)
        processing_route = self._determine_processing_route(context, violations)

        evaluation_time = (time.time() - start_time) * 1000.0

        return ConstraintEvaluationResult(
            is_allowed=is_allowed,
            violations=violations,
            actions=list(actions),
            suggested_response=suggested_response,
            processing_route=processing_route,
            confidence_score=self._calculate_confidence_score(context, violations),
            evaluation_time_ms=evaluation_time
        )

    def _evaluate_emotional_boundaries(self, context: EvaluationContext) -> List[ConstraintViolation]:
        """Evaluate emotional boundary constraints with precompiled regex patterns."""
        violations: List[ConstraintViolation] = []
        emotional_constraints = self.parser.get_constraints_by_type(ConstraintType.EMOTIONAL_BOUNDARY)
        if not emotional_constraints:
            return violations

        user_text = context.user_input or ""

        # Precompute once per evaluation
        primary_emotion = self.emotion_analyzer.detect_primary_emotion(user_text)

        for constraint in emotional_constraints:
            if not constraint.enabled:
                continue

            # Expect precompiled pattern on constraint.pattern; be defensive if string slipped through
            compiled = None
            pattern_str = None

            if isinstance(constraint.pattern, re.Pattern):
                compiled = constraint.pattern
                pattern_str = getattr(constraint, "pattern_str", constraint.pattern.pattern)
            elif isinstance(constraint.pattern, str) and constraint.pattern.strip():
                # Defensive fallback: compile on the fly (still case-insensitive)
                compiled = re.compile(constraint.pattern, re.IGNORECASE)
                pattern_str = constraint.pattern
            else:
                continue  # malformed/empty

            match = compiled.search(user_text)
            if not match:
                continue

            violations.append(ConstraintViolation(
                constraint_name=constraint.name,
                constraint_type=ConstraintType.EMOTIONAL_BOUNDARY,
                violation_reason=f"Emotional pattern detected: {pattern_str}",
                suggested_action=constraint.action,
                severity=constraint.severity,
                timestamp=datetime.now(),
                context={
                    "matched_pattern": pattern_str,
                    "matched_span": match.span(),
                    "matched_text": match.group(0),
                    "emotional_intensity": context.emotional_intensity,
                    "primary_emotion": primary_emotion
                }
            ))

        return violations

    def _evaluate_privacy_rules(self, context: EvaluationContext) -> List[ConstraintViolation]:
        """Evaluate privacy rule constraints using the DSL condition engine."""
        violations: List[ConstraintViolation] = []
        privacy_constraints = self.parser.get_constraints_by_type(ConstraintType.PRIVACY_RULE)

        if not privacy_constraints:
            return violations

        # Lazy-init the condition engine (wired to our PrivacyAnalyzer)
        if not hasattr(self, "_cond"):
            try:
                from dsl.parser import ConditionEngine
            except Exception as e:
                self.logger.error(f"Failed to import ConditionEngine: {e}")
                return violations
            self._cond = ConditionEngine(contains_pii_hook=self.privacy_analyzer.contains_pii)

        # Build a single env for this evaluation pass
        env = {
            "privacy_score": context.privacy_score,
            "session_duration": context.session_duration,
            "emotional_intensity": context.emotional_intensity,
            "interaction_count": context.interaction_count,
            "user_input": context.user_input,
        }

        for constraint in privacy_constraints:
            if not constraint.enabled:
                continue

            cond = (constraint.condition or "").strip()
            if not cond:
                # No condition means nothing to evaluate
                continue

            try:
                condition_met = self._cond.evaluate(cond, env)
            except Exception as e:
                self.logger.warning(f"Invalid privacy rule condition '{cond}' for '{constraint.name}': {e}")
                continue

            if not condition_met:
                continue

            # Enrich context with concrete PII hits (if any)
            pii_hits = self.privacy_analyzer.detect_pii(context.user_input or "")
            hit_payload = [
                {"kind": m.kind, "value": m.value, "span": m.span}
                for m in pii_hits
            ]

            violations.append(ConstraintViolation(
                constraint_name=constraint.name,
                constraint_type=ConstraintType.PRIVACY_RULE,
                violation_reason=f"Privacy condition met: {cond}",
                suggested_action=constraint.action,
                severity=constraint.severity,
                timestamp=datetime.now(),
                context={
                    "privacy_score": context.privacy_score,
                    "contains_pii": bool(pii_hits),
                    "pii_hits": hit_payload if pii_hits else [],
                }
            ))

        return violations

    def _evaluate_interaction_limits(self, context: EvaluationContext) -> List[ConstraintViolation]:
        """Evaluate interaction limit constraints via the DSL condition engine."""
        violations: List[ConstraintViolation] = []
        limit_constraints = self.parser.get_constraints_by_type(ConstraintType.INTERACTION_LIMIT)
        if not limit_constraints:
            return violations

        # Pull session metrics (rates/EMA) to make limits expressive
        sess_info = self.session_tracker.get_session_info(context.user_id) or {}
        env = {
            "session_duration": context.session_duration,
            "emotional_intensity": context.emotional_intensity,
            "interaction_count": context.interaction_count,
            # optional/richer metrics
            "rate_1m": sess_info.get("rate_1m", 0.0),
            "rate_5m": sess_info.get("rate_5m", 0.0),
            "rate_1h": sess_info.get("rate_1h", 0.0),
            "avg_emotional_intensity": sess_info.get("avg_emotional_intensity", 0.0),
            "ema_emotional_intensity": sess_info.get("ema_emotional_intensity", 0.0),
        }

        for constraint in limit_constraints:
            if not constraint.enabled:
                continue

            cond = (constraint.condition or "").strip()
            if not cond:
                continue

            try:
                condition_met = self._cond.evaluate(cond, env)
            except Exception as e:
                self.logger.warning(
                    f"Invalid interaction limit condition '{cond}' for '{constraint.name}': {e}"
                )
                continue

            if not condition_met:
                continue

            # Include a compact snapshot of the env in the violation
            env_snapshot = {
                "session_duration": env["session_duration"],
                "emotional_intensity": env["emotional_intensity"],
                "interaction_count": env["interaction_count"],
                "rate_1m": env["rate_1m"],
                "rate_5m": env["rate_5m"],
                "rate_1h": env["rate_1h"],
                "avg_emotional_intensity": env["avg_emotional_intensity"],
                "ema_emotional_intensity": env["ema_emotional_intensity"],
            }

            violations.append(ConstraintViolation(
                constraint_name=constraint.name,
                constraint_type=ConstraintType.INTERACTION_LIMIT,
                violation_reason=f"Interaction limit exceeded: {cond}",
                suggested_action=constraint.action,
                severity=constraint.severity,
                timestamp=datetime.now(),
                context=env_snapshot,
            ))

        return violations

    def _generate_suggested_response(self, violations: List[ConstraintViolation]) -> Optional[str]:
        """Generate appropriate response based on violations"""
        if not violations:
            return None

        # Prioritize by severity
        high = [v for v in violations if v.severity == "high"]
        primary = high[0] if high else violations[0]

        # Map action to template key
        action_key = primary.suggested_action.value
        # allow both "block" and legacy "privacy_block"
        if action_key == "block" and "block" not in self.response_templates:
            action_key = "privacy_block"

        return self.response_templates.get(action_key, self.response_templates["default"])

    def _determine_processing_route(self, context: EvaluationContext, violations: List[ConstraintViolation]) -> str:
        """Determine how the request should be processed"""
        if any(v.severity == "high" for v in violations):
            return "blocked"

        if any(v.constraint_type.name == "PRIVACY_RULE" for v in violations) or context.privacy_score > 0.7:
            return "local_only"

        return "local_preferred"

    def _calculate_confidence_score(self, context: EvaluationContext, violations: List[ConstraintViolation]) -> float:
        """Calculate confidence score for the evaluation"""
        base = 1.0
        if context.emotional_intensity > 0.8:
            base -= 0.1
        if context.privacy_score > 0.9:
            base -= 0.15
        return max(base, 0.1)


# Example constraint definitions for testing
EXAMPLE_CONSTRAINTS = """
constraints:
  emotional_boundaries:
    - name: "anger_detection"
      pattern: "angry|furious|rage|mad|pissed"
      response: "acknowledge_emotion"
      severity: "high"
      template: "I understand you're feeling frustrated. Let me help you work through this constructively."

    - name: "sadness_support"
      pattern: "sad|depressed|down|heartbroken"
      response: "acknowledge_emotion"
      severity: "medium"
      template: "I can sense you're going through a difficult time. I'm here to listen and support you."

  privacy_rules:
    - name: "pii_protection"
      condition: "contains_pii == true"
      action: "anonymize"
      severity: "high"

    - name: "sensitive_data_cloud_block"
      condition: "privacy_score > 0.8"
      action: "block"
      severity: "high"

  interaction_limits:
    - name: "session_duration_limit"
      condition: "session_duration > 60"
      action: "suggest_break"
      severity: "medium"
      threshold: 60

    - name: "emotional_intensity_limit"
      condition: "emotional_intensity > 0.9"
      action: "suggest_break"
      severity: "high"
      threshold: 0.9
"""


def test_constraint_system():
    """Test the complete constraint system"""
    print("🧪 Testing Complete Constraint System...")

    # Initialize components
    parser = ConstitutionalParser()
    constraints = parser.parse_constraints(EXAMPLE_CONSTRAINTS)
    print(f"✅ Loaded {len(constraints)} constraints")

    # Initialize enforcer
    enforcer = ConstraintEnforcer(parser)

    # Test cases
    test_cases = [
        {
            "description": "Angry user input",
            "input": "I am so angry and furious about this situation!"
        },
        {
            "description": "Privacy-sensitive input",
            "input": "My email is john.doe@example.com and my SSN is 123-45-6789"
        },
        {
            "description": "Normal conversation",
            "input": "How are you doing today?"
        }
    ]

    print("\n🧪 Running Test Cases...")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")

        context = EvaluationContext(
            user_id=f"test_user_{i}",
            user_input=test_case["input"],
            session_duration=5.0
        )

        result = enforcer.evaluate_constraints(context)

        print(f"Allowed: {result.is_allowed}")
        print(f"Processing Route: {result.processing_route}")
        print(f"Evaluation Time: {result.evaluation_time_ms:.2f}ms")
        print(f"Confidence: {result.confidence_score:.2f}")

        if result.violations:
            print("Violations:")
            for violation in result.violations:
                print(f"  - {violation.constraint_name} ({violation.severity})")
                print(f"    Action: {violation.suggested_action.value}")

        if result.suggested_response:
            print(f"Suggested Response: {result.suggested_response}")

    return True


if __name__ == "__main__":
    test_constraint_system()