"""Model fallback priorities after the zero-token pass."""
from dataclasses import dataclass, field
import json
import math

# Weighted tokens. Weak priors, updated only from calls with recorded usage.
PRIORS = {"math": 350, "data_analysis": 450, "search": 500, "coding": 950}


@dataclass
class Planner:
    max_task_tenths: int
    queue: str = "value"
    adaptive: bool = True
    costs: dict[str, float] = field(default_factory=lambda: dict(PRIORS))
    success: dict[str, float] = field(default_factory=dict)

    def load(self, path):
        data = json.loads(path.read_text())
        if data.get("version") != 1 or set(data.get("kinds", {})) - PRIORS.keys():
            raise ValueError("invalid calibration version or task kind")
        for kind, stats in data["kinds"].items():
            cost, success = stats["weighted_tokens"], stats["success_rate"]
            if not (math.isfinite(cost) and 20 <= cost <= 10_000 and 0 < success <= 1):
                raise ValueError("invalid calibration estimates")
            self.costs[kind] = cost
            self.success[kind] = success

    def estimate(self, task):
        instruction_cost = min(150, len(task.instruction.encode()) / 20)
        return self.costs[task.kind] + instruction_cost

    def priority(self, task):
        if self.queue == "reward":
            return (-task.reward, task.root.name)
        return (-task.reward * self.success.get(task.kind, 0.75) / self.estimate(task), -task.reward, task.root.name)

    def limit(self, task, remaining, pending_count):
        # Small qualification runs can use the full cap for both tasks.
        soft = self.max_task_tenths if pending_count <= 3 else int(22 * self.estimate(task))
        # A cheap learned mean must not truncate structured actions while the
        # remaining budget could afford a larger allowance for every task.
        soft = max(soft, remaining // max(1, pending_count))
        return min(self.max_task_tenths, soft, remaining)

    def observe(self, task, cost_tenths, route):
        if not self.adaptive or cost_tenths <= 0 or route in {"budget_or_proxy", "proxy_error"}:
            return
        previous = self.costs[task.kind]
        cost = min(3 * previous, max(previous / 3, cost_tenths / 10))
        self.costs[task.kind] = 0.8 * previous + 0.2 * cost
