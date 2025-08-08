# core/privacy.py

"""
Privacy Budget Manager for Ghost Protocol v0.1
Manages differential privacy budget (ε, δ) with operation logging
"""

from datetime import datetime
from typing import Dict, Any
import Tuple

class PrivacyBudgetManager:
    """Manages differential privacy budget"""

    def __init__(self, total_epsilon: float = 8.0, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.expenditure_log = []
        self.daily_reset_time = datetime.now().date()

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        self._check_daily_reset()
        return (self.total_epsilon - self.spent_epsilon, self.total_delta - self.spent_delta)

    def can_afford(self, epsilon_cost: float, delta_cost: float) -> bool:
        """Check if operation can be afforded"""
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        return remaining_epsilon >= epsilon_cost and remaining_delta >= delta_cost

    def spend_budget(self, operation: str, epsilon_cost: float, delta_cost: float) -> bool:
        """Spend privacy budget for an operation"""
        if not self.can_afford(epsilon_cost, delta_cost):
            return False

        self.spent_epsilon += epsilon_cost
        self.spent_delta += delta_cost

        self.expenditure_log.append({
            "operation": operation,
            "epsilon_cost": epsilon_cost,
            "delta_cost": delta_cost,
            "timestamp": datetime.now().isoformat(),
            "remaining_epsilon": self.total_epsilon - self.spent_epsilon
        })

        return True

    def _check_daily_reset(self):
        """Reset budget daily"""
        current_date = datetime.now().date()
        if current_date > self.daily_reset_time:
            self.spent_epsilon = 0.0
            self.spent_delta = 0.0
            self.daily_reset_time = current_date
            self.expenditure_log = []

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        utilization = (self.spent_epsilon / self.total_epsilon) * 100

        return {
            "total_epsilon": self.total_epsilon,
            "remaining_epsilon": remaining_epsilon,
            "utilization_percent": utilization,
            "operations_today": len(self.expenditure_log),
            "status": "healthy" if utilization < 80 else "approaching_limit" if utilization < 95 else "critical"
        }