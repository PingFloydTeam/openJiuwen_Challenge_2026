"""Queries over exact fields and directed links in bounded Markdown records."""
from collections import deque

from .domain import NoUniqueAnswer
from .tools import markdown_records


def index_records(root, strict):
    records = {}
    for i, (name, record) in enumerate(markdown_records(root, strict, max_total=2_000_000)):
        if i >= 512:
            raise ValueError("knowledge graph has too many records")
        entry = records.setdefault(name.casefold(), (name, {}))
        fields = entry[1]
        for label, value in record["facts"].items():
            key = label.casefold()
            choices = fields.setdefault(key, set())
            if value is None or value:
                choices.add(value)
    return records


def kb_select(root, filters, strict=False):
    """Return the unique heading matching all field/value pairs."""
    if not 1 <= len(filters) <= 8 or any(not name.strip() or not value.strip() for name, value in filters.items()):
        raise ValueError("supply up to eight nonempty exact field filters")
    matches = []
    uncertain = False
    for name, fields in index_records(root, strict).values():
        ambiguous = False
        for field, value in filters.items():
            choices = fields.get(field.casefold(), set())
            known = {item.casefold() for item in choices if item is not None}
            if value.casefold() not in known and None not in choices:
                break
            ambiguous |= None in choices or len(known) != 1
        else:
            if ambiguous:
                uncertain = True
            else:
                matches.append(name)
    if uncertain or len(matches) != 1:
        raise NoUniqueAnswer("missing_information", "the filters do not identify a unique heading")
    return matches[0]


def kb_route(root, start, end, fields, mode="distance", strict=False):
    """Return shortest directed distance or its uniquely determined first hop."""
    if not 1 <= len(fields) <= 6 or mode not in {"distance", "first"}:
        raise ValueError("provide one to six link fields and distance or first mode")
    records = index_records(root, strict)
    start, end = start.strip().casefold(), end.strip().casefold()
    if start not in records or end not in records:
        raise NoUniqueAnswer("missing_information", "route endpoint is missing")
    labels = {label.casefold() for label in fields}
    distance = {start: 0}
    first = {start: set()}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            if mode == "distance":
                return distance[node]
            if len(first[node]) != 1:
                raise NoUniqueAnswer("missing_information", "the shortest route has no unique first hop")
            return records[next(iter(first[node]))][0]
        next_distance = distance[node] + 1
        for label in labels:
            choices = records[node][1].get(label, set())
            if len(choices) > 1 or None in choices:
                raise NoUniqueAnswer("logical_error", "a relevant link is contradictory")
            if not choices:
                continue
            neighbor = next(iter(choices)).casefold()
            if neighbor not in records:
                raise NoUniqueAnswer("missing_information", "a reachable link has no record")
            hops = {neighbor} if node == start else first[node]
            if neighbor not in distance:
                distance[neighbor] = next_distance
                first[neighbor] = set(hops)
                queue.append(neighbor)
            elif distance[neighbor] == next_distance:
                first[neighbor].update(hops)
    raise NoUniqueAnswer("logical_error", "there is no directed route between the endpoints")
