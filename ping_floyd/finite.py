"""Bounded exhaustive search over explicit integer domains and a safe expression AST."""
import ast
from fractions import Fraction
from itertools import product
import math
import operator
import re

from .domain import NoUniqueAnswer
from .constraints import affine

BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def bounded(value):
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 512:
        raise ValueError("arithmetic exceeds 512-bit limit")
    return value


def expression(source, names, predicate=False):
    if not isinstance(source, str) or not 1 <= len(source) <= 300:
        raise ValueError("expressions must be short strings")
    source = re.sub(r"(?<![<>=!])=(?!=)", "==", source)
    tree = ast.parse(source, mode="eval").body
    if predicate and not (isinstance(tree, (ast.Compare, ast.BoolOp)) or
                          isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not)):
        raise ValueError("constraints must be comparisons or Boolean expressions")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("expression exceeds AST limit")

    def compile_node(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10**12:
            return lambda env: Fraction(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return lambda env: Fraction(env[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            arg = compile_node(node.operand)
            if isinstance(node.op, ast.Not):
                return lambda env: not arg(env)
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return lambda env: sign * arg(env)
        if isinstance(node, ast.BinOp) and type(node.op) in BIN:
            left, right, op = compile_node(node.left), compile_node(node.right), BIN[type(node.op)]
            return lambda env: bounded(op(left(env), right(env)))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            if type(node.right.value) is not int or not 0 <= node.right.value <= 6:
                raise ValueError("only integer powers 0..6 are allowed")
            left, power = compile_node(node.left), node.right.value
            return lambda env: bounded(left(env) ** power)
        if isinstance(node, ast.Compare) and all(type(op) in CMP for op in node.ops):
            values = [compile_node(n) for n in [node.left, *node.comparators]]
            operators = [CMP[type(op)] for op in node.ops]
            def compare(env):
                previous = values[0](env)
                for op, value in zip(operators, values[1:], strict=True):
                    current = value(env)
                    if not op(previous, current):
                        return False
                    previous = current
                return True
            return compare
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            parts = [compile_node(n) for n in node.values]
            return lambda env: (all if isinstance(node.op, ast.And) else any)(part(env) for part in parts)
        raise ValueError("use only domain variables, bounded arithmetic, comparisons, and/or/not")

    return compile_node(tree)


def reduced_assignments(domains, constraints):
    """Eliminate affine equalities over Q, then enumerate only free coordinates.

    Every reconstructed point is checked for integrality and original bounds.
    Nonlinear and Boolean conditions remain with the ordinary exact evaluator.
    """
    names = sorted(domains, key=lambda n: -(domains[n][1] - domains[n][0]))
    matrix = []
    for text in constraints:
        match = re.fullmatch(r"([^<>=!]+)==?([^<>=!]+)", text)
        if not match:
            continue
        try:
            terms, constant = affine(f"({match[1].strip()})-({match[2].strip()})")
        except (ValueError, SyntaxError):
            continue
        if terms.keys() <= domains.keys():
            matrix.append([terms.get(n, Fraction(0)) for n in names] + [-constant])
    pivots = {}
    for col in range(len(names)):
        rank = len(pivots)
        found = next((i for i in range(rank, len(matrix)) if matrix[i][col]), None)
        if found is None:
            continue
        matrix[rank], matrix[found] = matrix[found], matrix[rank]
        divisor = matrix[rank][col]
        matrix[rank] = [v / divisor for v in matrix[rank]]
        for i, row in enumerate(matrix):
            if i != rank and row[col]:
                scale = row[col]
                matrix[i] = [a - scale*b for a, b in zip(row, matrix[rank], strict=True)]
        pivots[col] = rank
    inconsistent = any(not any(row[:-1]) and row[-1] for row in matrix)
    free = [i for i in range(len(names)) if i not in pivots]
    ranges = [range(domains[names[i]][0], domains[names[i]][1]+1) for i in free]
    candidates = 0 if inconsistent else math.prod(map(len, ranges))

    def assignments():
        if inconsistent:
            return
        for values in product(*ranges):
            env = dict(zip((names[i] for i in free), values, strict=True))
            for col, row in pivots.items():
                value = matrix[row][-1] - sum(matrix[row][j]*env[names[j]] for j in free)
                low, high = domains[names[col]]
                if value.denominator != 1 or not low <= value <= high:
                    break
                env[names[col]] = int(value)
            else:
                yield env
    return candidates, assignments()


def finite_search(domains, constraints, target="", mode="unique"):
    if not 1 <= len(domains) <= 6 or len(constraints) > 12 or mode not in {"unique", "count", "min", "max", "sum"}:
        raise ValueError("finite search requires 1..6 variables, <=12 constraints, and a supported mode")
    names, ranges = [], []
    for name, bounds in domains.items():
        if (not re.fullmatch(r"[A-Za-z][A-Za-z_0-9]{0,15}", name) or len(bounds) != 2
                or any(type(n) is not int or abs(n) > 10000 for n in bounds) or bounds[0] > bounds[1]):
            raise ValueError("each domain is a named inclusive integer interval within [-10000,10000]")
        names.append(name)
        ranges.append(range(bounds[0], bounds[1]+1))
    # Validate the complete input before symbolic preprocessing; no ignored names.
    conditions = [expression(text, domains, predicate=True) for text in constraints]
    candidates, assignments = reduced_assignments(domains, constraints)
    if candidates > 100000:
        raise ValueError("finite search exceeds 100000 assignments after equality elimination")
    work = sum(len(text) for text in [*constraints, target]) * candidates
    if work > 8_000_000:
        raise ValueError("finite search exceeds expression-work limit")
    if mode != "count" and not target:
        raise ValueError("this mode requires a target expression")
    objective = expression(target, domains) if target else None
    total, count, best = Fraction(0), 0, None
    for env in assignments:
        try:
            if not all(condition(env) for condition in conditions):
                continue
            count += 1
            if mode == "count":
                continue
            value = objective(env)
            if type(value) is bool:
                raise ValueError("target must be numeric")
            value = Fraction(value)
            if max(value.numerator.bit_length(), value.denominator.bit_length()) > 512:
                raise ValueError("intermediate result exceeds numeric limit")
            if mode == "unique" and best is not None and value != best:
                raise NoUniqueAnswer("missing_information", "feasible assignments give different target values")
            if mode == "min":
                best = value if best is None else min(best, value)
            elif mode == "max":
                best = value if best is None else max(best, value)
            else:
                best = value
            total += value
        except ZeroDivisionError:
            raise NoUniqueAnswer("undefined_math", "expression is undefined for a candidate; add an explicit domain guard") from None
    if mode == "count":
        return count
    if not count:
        raise NoUniqueAnswer("logical_error", "no assignment satisfies all constraints")
    return str(total if mode == "sum" else best)
