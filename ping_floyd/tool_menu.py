"""Small menus for the first attempt, drawn from the existing tool allowlist."""

TOOL_PROFILES = ("full", "compact", "body", "shortlist")

PREFERRED = {
    "math": ("balance", "calc", "linear", "congruences", "grid_paths", "graph"),
    "data_analysis": ("query", "sql", "schema", "read", "calc"),
    "search": ("lookup", "grep", "read", "kb_route", "kb_select", "schema", "query", "calc"),
}


def shortlist(kind, tools, limit=3):
    order = PREFERRED.get(kind)
    if order is None or len(tools) <= limit:
        return list(tools)
    # Task-specific tools already enabled by the compact profile get first priority.
    return sorted(tools, key=lambda tool: order.index(tool) if tool in order else -1)[:limit]
