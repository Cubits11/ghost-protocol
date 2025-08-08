# 📘 Ghost Protocol v0.1 – API Documentation

This document provides an overview of the core API for the Ghost Protocol emotionally sovereign AI system.

---

## 🔧 `GhostProtocolConfig`

Configuration settings for the entire system.

```python
@dataclass
class GhostProtocolConfig:
    db_path: str = 'ghost_protocol.db'
    encryption_password: str = 'ghost_protocol_secret_key'
    privacy_budget_epsilon: float = 8.0
    privacy_budget_delta: float = 1e-05
    session_timeout_minutes: int = 60
    memory_decay_enabled: bool = True
    audit_logging_enabled: bool = True
    default_privacy_level: int = 2
```

---

## 📦 `ProcessingResult`

Result of processing a user request.

```python
class ProcessingResult:
    response: str
    processing_route: str
    constraints_applied: List[str]
    privacy_budget_used: float
    emotional_context_stored: bool
    interaction_id: str
    confidence_score: float
    processing_time_ms: float
    system_status: Dict[str, Any]
```

---

## 🔒 `PrivacyBudgetManager`

Manages the differential privacy budget for system operations.

```python
class PrivacyBudgetManager:
    def get_remaining_budget(self) -> Tuple[float, float]
    def can_afford(self, epsilon_cost: float, delta_cost: float) -> bool
    def spend_budget(self, operation: str, epsilon_cost: float, delta_cost: float) -> bool
    def get_budget_status(self) -> Dict[str, Any]
```

---

## 🤖 `LocalAISimulator`

Simulates AI processing locally.

```python
class LocalAISimulator:
    async def generate_response(
        self, user_input: str,
        context: EvaluationContext,
        constraint_result: ConstraintEvaluationResult
    ) -> str
```

---

## ☁️ `CloudAPISimulator`

Simulates AI response generation using cloud infrastructure.

```python
class CloudAPISimulator:
    async def generate_response(
        self, anonymized_input: str,
        privacy_budget_used: float
    ) -> str
```

---

## 🧠 `GhostProtocolSystem`

Main orchestration class for the system.

```python
class GhostProtocolSystem:
    async def process_user_input(self, user_input: str, user_id: str = 'default_user') -> ProcessingResult
    def get_system_status(self) -> Dict[str, Any]
    def load_user_constraints(self, constraints_yaml: str) -> bool
    def close(self)
```

---

## 🧪 `test_ghost_protocol_system()`

Function for full pipeline testing.

```python
def test_ghost_protocol_system():
    """Runs an integration test of the full system pipeline."""
```
