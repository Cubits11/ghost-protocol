# core/config.py
"""
Ghost Protocol v0.1 - System Configuration Module
Centralized configuration for the entire Ghost Protocol system
"""

from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class GhostProtocolConfig:
    """
    Configuration for Ghost Protocol system.
    You can modify values here or override via environment variables in the future.
    """
    db_path: str = os.getenv("GHOST_DB_PATH", "ghost_protocol.db")
    encryption_password: str = os.getenv("GHOST_ENCRYPTION_PASSWORD", "ghost_protocol_secret_key")
    privacy_budget_epsilon: float = float(os.getenv("GHOST_PRIVACY_EPSILON", 8.0))
    privacy_budget_delta: float = float(os.getenv("GHOST_PRIVACY_DELTA", 1e-5))
    session_timeout_minutes: int = int(os.getenv("GHOST_SESSION_TIMEOUT", 60))
    memory_decay_enabled: bool = os.getenv("GHOST_MEMORY_DECAY", "true").lower() == "true"
    audit_logging_enabled: bool = os.getenv("GHOST_AUDIT_LOGGING", "true").lower() == "true"
    default_privacy_level: int = int(os.getenv("GHOST_DEFAULT_PRIVACY_LEVEL", 2))

    def as_dict(self) -> dict:
        """Return config as dictionary (for UI/status display)"""
        return {
            "db_path": self.db_path,
            "encryption_password": "••••••••••••••",
            "privacy_budget_epsilon": self.privacy_budget_epsilon,
            "privacy_budget_delta": self.privacy_budget_delta,
            "session_timeout_minutes": self.session_timeout_minutes,
            "memory_decay_enabled": self.memory_decay_enabled,
            "audit_logging_enabled": self.audit_logging_enabled,
            "default_privacy_level": self.default_privacy_level
        }

    def summary(self) -> str:
        """Return short config summary for logs/UI"""
        return (
            f"📁 DB: {self.db_path}, "
            f"🔐 Privacy ε={self.privacy_budget_epsilon}, "
            f"🕒 Session Timeout: {self.session_timeout_minutes}min, "
            f"🧠 Memory Decay: {self.memory_decay_enabled}, "
            f"📜 Audit Logging: {self.audit_logging_enabled}"
        )