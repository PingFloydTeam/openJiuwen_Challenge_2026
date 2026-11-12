"""Bounded exact algorithms for structured local tasks."""
from collections import defaultdict, deque
import csv
import heapq
from io import StringIO
from math import comb

from .domain import NoUniqueAnswer, local_file, read_text


def grid_paths(rows, cols, blocked=()):
    """Count right/down paths between opposite corners of a cell grid."""
    if not (1 <= rows <= 1000 and 1 <= cols <= 1000 and len(blocked) <= 2000):
        raise ValueError("grid dimensions or number of blocked cells exceed limits")
    obstacles = set()
    for cell in blocked:
        if len(cell) != 2 or not (1 <= cell[0] <= rows and 1 <= cell[1] <= cols):
            raise ValueError("blocked cells must have valid 1-based coordinates")
        obstacles.add(tuple(cell))
    if not obstacles:
        return comb(rows + cols - 2, cols - 1)
    if rows * cols > 150_000:
        raise ValueError("grid with obstacles exceeds work limit")
    paths = [1] + [0] * (cols - 1)
    for row in range(1, rows + 1):
        for col in range(cols):
            if (row, col + 1) in obstacles:
                paths[col] = 0
            elif col:
                paths[col] += paths[col - 1]
    return paths[-1]


def weighted_path(edges, start, end, mode="shortest"):
    """Return a shortest distance or the longest distance in a DAG."""
    if len(edges) > 2500 or mode not in {"shortest", "longest_dag"}:
        raise ValueError("supply up to 2500 edges and shortest or longest_dag mode")
    neighbors = defaultdict(list)
    indegree = dict.fromkeys([start, end], 0)
    for left, right, weight in edges:
        if any(type(name) is not str or not 1 <= len(name) <= 80 for name in (left, right)):
            raise ValueError("vertices must be short nonempty labels")
        if type(weight) is not int or abs(weight) > 10**12 or (mode == "shortest" and weight < 0):
            raise ValueError("invalid weight; shortest mode requires nonnegative weights")
        neighbors[left].append((right, weight))
        indegree.setdefault(left, 0)
        indegree[right] = indegree.get(right, 0) + 1
    distances = {start: 0}
    if mode == "shortest":
        queue = [(0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            if node == end:
                return distance
            for target, weight in neighbors[node]:
                candidate = distance + weight
                if target not in distances or candidate < distances[target]:
                    distances[target] = candidate
                    heapq.heappush(queue, (candidate, target))
    else:
        queue = deque(node for node, count in indegree.items() if count == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for target, weight in neighbors[node]:
                if node in distances:
                    candidate = distances[node] + weight
                    if target not in distances or candidate > distances[target]:
                        distances[target] = candidate
                indegree[target] -= 1
                if not indegree[target]:
                    queue.append(target)
        if visited != len(indegree):
            raise NoUniqueAnswer("logical_error", "the graph contains a directed cycle")
        if end in distances:
            return distances[end]
    raise NoUniqueAnswer("logical_error", "no route exists between the requested endpoints")


def graph_distance(root, start, end, mode="shortest", file="", edges=None):
    if bool(file) == (edges is not None):
        raise ValueError("supply either an edge list or a workspace CSV/TSV file")
    if file:
        path = local_file(root, file, {".csv", ".tsv"})
        stream = StringIO(read_text(path, 200_000))
        rows = csv.reader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",", strict=True)
        if next(rows, []) != ["source", "target", "weight"]:
            raise ValueError("the graph table must have source,target,weight columns in order")
        edges = []
        for source, target, weight in rows:
            edges.append((source, target, int(weight)))
            if len(edges) > 2500:
                raise ValueError("the graph table has too many edges")
    return weighted_path(edges, start, end, mode)
