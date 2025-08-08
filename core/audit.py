# core/audit.py

"""
Tamper-evident audit logger for Ghost Protocol
Logs decisions, constraint violations, and privacy actions
"""

import json
import os
import hashlib
import time
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_path: str = "data/logs/audit_log.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                json.dump([], f)

    def _generate_hash(self, record: Dict[str, Any]) -> str:
        raw = json.dumps(record, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def log_event(self, event: Dict[str, Any]):
        event["timestamp"] = time.time()
        event["record_hash"] = self._generate_hash(event)

        with open(self.log_path, "r+") as f:
            logs = json.load(f)
            logs.append(event)
            f.seek(0)
            json.dump(logs, f, indent=2)