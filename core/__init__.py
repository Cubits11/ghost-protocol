# core/__init__.py - Ghost Protocol Core Module Exports
"""
Ghost Protocol v0.1 - Core Module
Exports all essential components for emotionally sovereign AI
"""

# Constitutional Constraints System
from .constraints import (
    ConstitutionalParser,
    ConstraintEnforcer,
    EvaluationContext,
    ConstraintEvaluationResult,
    ConstraintType,
    ActionType,
    ConstraintViolation,
    CompiledConstraint,
    PrivacyAnalyzer,
    EmotionAnalyzer,
    SessionTracker,
    EXAMPLE_CONSTRAINTS
)

# Encrypted Memory Vault
from .vault import (
    EmotionalMemoryVault,
    EmotionalContext,
    MemoryQuery,
    EncryptionManager,
    SearchableEncryption
)

# Version info
__version__ = "0.1.0"
__author__ = "Ghost Protocol Team"
__description__ = "The first technically enforceable framework for emotionally sovereign AI"

# Core system components for easy import
__all__ = [
    # Main classes
    "ConstitutionalParser",
    "ConstraintEnforcer",
    "EmotionalMemoryVault",

    # Data structures
    "EvaluationContext",
    "ConstraintEvaluationResult",
    "EmotionalContext",
    "MemoryQuery",
    "CompiledConstraint",
    "ConstraintViolation",

    # Enums
    "ConstraintType",
    "ActionType",

    # Analyzers
    "PrivacyAnalyzer",
    "EmotionAnalyzer",
    "SessionTracker",

    # Encryption
    "EncryptionManager",
    "SearchableEncryption",

    # Examples
    "EXAMPLE_CONSTRAINTS",

    # Metadata
    "__version__",
    "__author__",
    "__description__"
]