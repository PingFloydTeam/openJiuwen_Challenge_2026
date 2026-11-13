"""Bounded deterministic task state; no model-generated summaries."""
from collections import OrderedDict


class TaskMemory:
    def __init__(self, limit=2000):
        self.limit = limit
        self.facts = OrderedDict()
        self.error = ''

    def success(self, signature, message):
        self.facts.pop(signature, None)
        self.facts[signature] = message[:1000]
        while len(self.facts) > 4:
            self.facts.popitem(last=False)
        self.error = ''

    def failure(self, message):
        self.error = message[:500]

    def render(self):
        remaining = self.limit - len(self.error)
        selected = []
        for value in reversed(self.facts.values()):
            if len(value) <= remaining:
                selected.append(value)
                remaining -= len(value) + 1
        return '\n'.join([*reversed(selected), *([self.error] if self.error else [])])
