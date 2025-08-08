# core/router.py

"""
Hybrid Router for Ghost Protocol
Determines whether to process input locally or route to cloud
"""

from typing import Dict, Any

class HybridRouter:
    def __init__(self, default_mode: str = "local_preferred"):
        self.default_mode = default_mode

    def decide_route(self, context: Dict[str, Any]) -> str:
        if context.get("constraint_violations"):
            return "blocked"

        if self.default_mode == "local_only":
            return "local_only"

        # Simulate privacy-aware complexity scoring
        if context.get("complexity_score", 0) < 0.5:
            return "local_preferred"

        return "cloud_required"