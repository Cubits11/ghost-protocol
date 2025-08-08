# ghost_protocol_main.py - Main System Integration
"""
Ghost Protocol v0.1 - Emotionally Sovereign AI System
Integrates all components: Constitutional DSL, Constraint Enforcement, Encrypted Memory
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

# Import our components from core modules
from core.constraints import (
    ConstitutionalParser,
    ConstraintEnforcer,
    EvaluationContext,
    ConstraintEvaluationResult,
    EXAMPLE_CONSTRAINTS
)
from core.vault import EmotionalMemoryVault, EmotionalContext, MemoryQuery
from core.privacy import PrivacyBudgetManager
from core.router import HybridRouter
from core.audit import AuditLogger
from core.config import GhostProtocolConfig

class LocalAISimulator:
    """Simulates local AI processing"""

    def __init__(self):
        self.model_loaded = True
        self.response_templates = {
            "acknowledge_emotion": [
                "I can sense you're experiencing strong emotions. I'm here to listen and support you.",
                "It sounds like you're going through something difficult. How can I help?",
                "I understand this is important to you. Let's work through this together."
            ],
            "general": [
                "I understand what you're saying. Let me help you with that.",
                "That's an interesting point. Here's how I can assist:",
                "I'm here to help. Let me think about the best way to approach this."
            ],
            "privacy_sensitive": [
                "I'll handle this sensitively and keep your information private.",
                "I understand this is personal. I'll process this locally to protect your privacy.",
                "Your privacy is important. I'm processing this securely on your device."
            ]
        }

    async def generate_response(self, user_input: str, context: EvaluationContext,
                                constraint_result: ConstraintEvaluationResult) -> str:
        """Generate AI response based on input and constraints"""
        # Simulate processing time
        await asyncio.sleep(0.1 + np.random.random() * 0.1)

        # Choose response based on constraints
        if constraint_result.violations:
            violation_actions = [v.suggested_action.value for v in constraint_result.violations]
            if "acknowledge_emotion" in violation_actions:
                templates = self.response_templates["acknowledge_emotion"]
            else:
                templates = self.response_templates["privacy_sensitive"]
        else:
            templates = self.response_templates["general"]

        # Select random template and customize
        base_response = np.random.choice(templates)

        # Add context-specific information
        if context.emotional_intensity > 0.7:
            base_response += " I can see this is really important to you."

        if context.privacy_score > 0.5:
            base_response += " I'm keeping this conversation private and secure."

        return base_response


class CloudAPISimulator:
    """Simulates cloud AI processing"""

    def __init__(self):
        self.api_available = True

    async def generate_response(self, anonymized_input: str, privacy_budget_used: float) -> str:
        """Generate response using cloud API"""
        # Simulate network latency
        await asyncio.sleep(0.5 + np.random.random() * 0.3)

        responses = [
            "Based on your query, here's what I can help you with...",
            "I've analyzed your request and here's my recommendation...",
            "After processing your input, I think the best approach would be...",
            "Here's how I can assist you with that..."
        ]

        base_response = np.random.choice(responses)
        return f"[CLOUD] {base_response} (Privacy budget used: {privacy_budget_used:.3f}ε)"


class GhostProtocolSystem:
    """Main Ghost Protocol system orchestrator"""

    def __init__(self, config: Optional[GhostProtocolConfig] = None):
        self.config = config or GhostProtocolConfig()

        # Initialize components
        self.parser = ConstitutionalParser()
        self.memory_vault = EmotionalMemoryVault(
            self.config.db_path,
            self.config.encryption_password
        )
        self.constraint_enforcer = ConstraintEnforcer(self.parser)
        self.privacy_manager = PrivacyBudgetManager(
            self.config.privacy_budget_epsilon,
            self.config.privacy_budget_delta
        )

        # AI processors
        self.local_ai = LocalAISimulator()
        self.cloud_ai = CloudAPISimulator()

        # System state
        self.system_initialized = False
        self.active_sessions = {}

        # Initialize with default constraints
        self._initialize_system()

    def _initialize_system(self):
        """Initialize Ghost Protocol system"""
        try:
            # Load default constraints
            constraints = self.parser.parse_constraints(EXAMPLE_CONSTRAINTS)
            print(f"✅ Loaded {len(constraints)} constitutional constraints")

            # Verify memory vault
            integrity = self.memory_vault.verify_data_integrity()
            print(f"✅ Memory vault integrity: {integrity.get('integrity_score', 0):.1%}")

            # Check privacy budget
            budget_status = self.privacy_manager.get_budget_status()
            print(f"✅ Privacy budget: {budget_status['remaining_epsilon']:.1f}ε remaining")

            self.system_initialized = True
            print("🔥 Ghost Protocol v0.1 initialized successfully!")

        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            self.system_initialized = False

    async def process_user_input(self, user_input: str, user_id: str = "default_user") -> ProcessingResult:
        """Main entry point for processing user requests"""
        start_time = time.time()
        interaction_id = f"{user_id}_{int(time.time() * 1000)}"

        if not self.system_initialized:
            return ProcessingResult(
                response="System not initialized. Please try again.",
                processing_route="error",
                constraints_applied=[],
                privacy_budget_used=0.0,
                emotional_context_stored=False,
                interaction_id=interaction_id,
                confidence_score=0.0,
                processing_time_ms=0.0,
                system_status={"error": "system_not_initialized"}
            )

        try:
            # Create evaluation context
            context = EvaluationContext(
                user_id=user_id,
                user_input=user_input,
                timestamp=datetime.now(),
                metadata={"interaction_id": interaction_id}
            )

            # Evaluate constraints
            constraint_result = self.constraint_enforcer.evaluate_constraints(context)

            # Determine processing route
            processing_route = self._determine_processing_route(context, constraint_result)

            # Generate response based on route
            if processing_route == "blocked":
                response = constraint_result.suggested_response or "I can't process that request due to safety constraints."
                privacy_budget_used = 0.0
            elif processing_route == "local_only":
                response = await self.local_ai.generate_response(user_input, context, constraint_result)
                privacy_budget_used = 0.0
            elif processing_route == "cloud_anonymized":
                # Anonymize input and use cloud
                anonymized_input = self._anonymize_input(user_input, context)
                budget_cost = 0.2  # Cost for cloud processing

                if self.privacy_manager.spend_budget("cloud_processing", budget_cost, 1e-6):
                    response = await self.cloud_ai.generate_response(anonymized_input, budget_cost)
                    privacy_budget_used = budget_cost
                else:
                    # Fall back to local processing
                    response = await self.local_ai.generate_response(user_input, context, constraint_result)
                    privacy_budget_used = 0.0
                    processing_route = "local_fallback"
            else:
                # Default to local processing
                response = await self.local_ai.generate_response(user_input, context, constraint_result)
                privacy_budget_used = 0.0
                processing_route = "local_default"

            # Store emotional context
            emotional_context_stored = await self._store_emotional_context(
                user_id, user_input, response, context, constraint_result
            )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            return ProcessingResult(
                response=response,
                processing_route=processing_route,
                constraints_applied=[v.constraint_name for v in constraint_result.violations],
                privacy_budget_used=privacy_budget_used,
                emotional_context_stored=emotional_context_stored,
                interaction_id=interaction_id,
                confidence_score=constraint_result.confidence_score,
                processing_time_ms=processing_time_ms,
                system_status=self.get_system_status()
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return ProcessingResult(
                response=f"I encountered an error processing your request: {str(e)}",
                processing_route="error",
                constraints_applied=[],
                privacy_budget_used=0.0,
                emotional_context_stored=False,
                interaction_id=interaction_id,
                confidence_score=0.0,
                processing_time_ms=processing_time_ms,
                system_status=self.get_system_status()
            )

    def _determine_processing_route(self, context: EvaluationContext,
                                    constraint_result: ConstraintEvaluationResult) -> str:
        """Determine how to process the request"""
        # Check for blocking violations
        high_severity_violations = [v for v in constraint_result.violations if v.severity == "high"]
        if high_severity_violations:
            return "blocked"

        # Check privacy sensitivity
        if context.privacy_score > 0.7:
            return "local_only"

        # Check emotional intensity
        if context.emotional_intensity > 0.8:
            return "local_only"

        # Check privacy budget availability
        if context.privacy_score > 0.3 and self.privacy_manager.can_afford(0.2, 1e-6):
            return "cloud_anonymized"

        return "local_only"

    def _anonymize_input(self, user_input: str, context: EvaluationContext) -> str:
        """Anonymize user input for cloud processing"""
        # Simple anonymization - would be more sophisticated in production
        anonymized = user_input

        # Remove potential PII patterns
        import re
        anonymized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', anonymized)
        anonymized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', anonymized)
        anonymized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', anonymized)

        # Add noise if high emotional intensity
        if context.emotional_intensity > 0.6:
            anonymized = f"[EMOTIONAL_CONTENT] {anonymized}"

        return anonymized

    async def _store_emotional_context(self, user_id: str, user_input: str, response: str,
                                       context: EvaluationContext,
                                       constraint_result: ConstraintEvaluationResult) -> bool:
        """Store emotional context in encrypted memory"""
        try:
            # Create emotion vector (simplified)
            emotion_vector = np.array([
                context.emotional_intensity,
                context.privacy_score,
                len(constraint_result.violations),
                constraint_result.confidence_score,
                0.5  # placeholder for additional features
            ], dtype=np.float32)

            # Determine primary emotion
            primary_emotion = "neutral"
            if context.emotional_intensity > 0.5:
                # This would be more sophisticated emotion detection
                if "angry" in user_input.lower() or "mad" in user_input.lower():
                    primary_emotion = "anger"
                elif "sad" in user_input.lower() or "depressed" in user_input.lower():
                    primary_emotion = "sadness"
                elif "anxious" in user_input.lower() or "worried" in user_input.lower():
                    primary_emotion = "anxiety"
                elif "happy" in user_input.lower() or "excited" in user_input.lower():
                    primary_emotion = "joy"

            # Create emotional context
            emotional_context = EmotionalContext(
                context_id=context.metadata.get("interaction_id"),
                user_id=user_id,
                emotion_vector=emotion_vector,
                primary_emotion=primary_emotion,
                emotional_intensity=context.emotional_intensity,
                conversation_summary=f"User: {user_input[:100]}... AI: {response[:100]}...",
                tags=self._extract_tags(user_input, constraint_result),
                privacy_level=self._determine_privacy_level(context),
                metadata={
                    "processing_route": constraint_result.processing_route,
                    "constraints_triggered": [v.constraint_name for v in constraint_result.violations],
                    "confidence_score": constraint_result.confidence_score
                }
            )

            return self.memory_vault.store_emotional_context(emotional_context)

        except Exception as e:
            print(f"Error storing emotional context: {e}")
            return False

    def _extract_tags(self, user_input: str, constraint_result: ConstraintEvaluationResult) -> List[str]:
        """Extract relevant tags from user input"""
        tags = []

        # Add emotion-based tags
        emotion_keywords = {
            "work": ["work", "job", "office", "boss", "colleague"],
            "family": ["family", "mother", "father", "parent", "child"],
            "relationship": ["relationship", "partner", "boyfriend", "girlfriend"],
            "health": ["health", "sick", "doctor", "medical"],
            "money": ["money", "financial", "budget", "debt", "salary"]
        }

        user_lower = user_input.lower()
        for tag, keywords in emotion_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                tags.append(tag)

        # Add constraint-based tags
        for violation in constraint_result.violations:
            tags.append(f"constraint_{violation.constraint_type.value}")

        return tags

    def _determine_privacy_level(self, context: EvaluationContext) -> int:
        """Determine privacy level based on context"""
        if context.privacy_score > 0.8:
            return 3  # High privacy
        elif context.privacy_score > 0.4:
            return 2  # Medium privacy
        else:
            return 1  # Low privacy

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            budget_status = self.privacy_manager.get_budget_status()
            vault_stats = self.memory_vault.get_memory_statistics("system")

            return {
                "system_initialized": self.system_initialized,
                "privacy_budget": budget_status,
                "memory_vault": {
                    "total_contexts": vault_stats.get("total_contexts", 0),
                    "storage_health": vault_stats.get("storage_health", "unknown")
                },
                "constraints": {
                    "total_loaded": len(self.parser.compiled_constraints),
                    "enabled": len([c for c in self.parser.compiled_constraints if c.enabled])
                },
                "system_health": "healthy" if self.system_initialized else "error",
                "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
            }
        except Exception as e:
            return {"error": str(e), "system_health": "error"}

    def load_user_constraints(self, constraints_yaml: str) -> bool:
        """Load custom user constraints"""
        try:
            constraints = self.parser.parse_constraints(constraints_yaml)
            print(f"✅ Loaded {len(constraints)} user constraints")
            return True
        except Exception as e:
            print(f"❌ Failed to load user constraints: {e}")
            return False

    def close(self):
        """Cleanup system resources"""
        if hasattr(self, 'memory_vault'):
            self.memory_vault.close()


def test_ghost_protocol_system():
    """Test the complete Ghost Protocol system"""
    print("👻 Testing Complete Ghost Protocol System...")

    # Initialize system
    ghost = GhostProtocolSystem()

    if not ghost.system_initialized:
        print("❌ System failed to initialize")
        return False

    # Test cases
    test_cases = [
        {
            "description": "Normal conversation",
            "input": "Hello, how are you doing today?",
            "user_id": "test_user_1"
        },
        {
            "description": "Angry emotional input",
            "input": "I am so angry and frustrated about this situation!",
            "user_id": "test_user_2"
        },
        {
            "description": "Privacy-sensitive input",
            "input": "My email is john.doe@example.com and I need help with something personal",
            "user_id": "test_user_3"
        },
        {
            "description": "Mixed emotional and privacy content",
            "input": "I'm really worried about my health diagnosis from doctor.smith@hospital.com",
            "user_id": "test_user_4"
        }
    ]

    print("\n🧪 Running Test Cases...")

    async def run_tests():
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['description']} ---")
            print(f"Input: {test_case['input']}")

            result = await ghost.process_user_input(test_case["input"], test_case["user_id"])

            print(f"✅ Response: {result.response}")
            print(f"🔄 Processing Route: {result.processing_route}")
            print(f"⚖️ Constraints Applied: {result.constraints_applied}")
            print(f"🔒 Privacy Budget Used: {result.privacy_budget_used}ε")
            print(f"💾 Emotional Context Stored: {result.emotional_context_stored}")
            print(f"⏱️ Processing Time: {result.processing_time_ms:.1f}ms")
            print(f"🎯 Confidence: {result.confidence_score:.2f}")

    # Run async tests
    asyncio.run(run_tests())

    print("\n📊 System Status:")
    status = ghost.get_system_status()
    print(json.dumps(status, indent=2, default=str))

    # Cleanup
    ghost.close()

    print("\n✅ Ghost Protocol system tests completed!")
    return True


if __name__ == "__main__":
    test_ghost_protocol_system()