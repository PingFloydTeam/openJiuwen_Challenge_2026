"""Exact bounded constraint solvers. The input language has no executable expressions."""
import ast
from fractions import Fraction
from math import gcd
import re

from .domain import NoUniqueAnswer

VARIABLE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,15}\Z")


def affine(expression):
    if not 1 <= len(expression) <= 180:
        raise ValueError("invalid expression length")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10**18:
            return {}, Fraction(node.value)
        if isinstance(node, ast.Name) and VARIABLE.fullmatch(node.id):
            return {node.id: Fraction(1)}, Fraction(0)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            terms, constant = visit(node.operand)
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return {key: value * sign for key, value in terms.items()}, constant * sign
        if not isinstance(node, ast.BinOp):
            raise ValueError("use integer constants, variables and linear arithmetic")
        left, x = visit(node.left)
        right, y = visit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            terms = {key: left.get(key, 0) + sign * right.get(key, 0) for key in left.keys() | right.keys()}
            return {key: value for key, value in terms.items() if value}, x + sign * y
        if isinstance(node.op, ast.Mult) and not (left and right):
            return ({key: value * y for key, value in left.items()} if left
                    else {key: value * x for key, value in right.items()}), x * y
        if isinstance(node.op, ast.Div) and not right:
            if not y:
                raise NoUniqueAnswer("undefined_math", "division by zero in a constraint")
            return {key: value / y for key, value in left.items()}, x / y
        raise ValueError("constraints must be linear and denominators constant")

    terms, value = visit(ast.parse(expression.strip(), mode="eval").body)
    return {key: term for key, term in terms.items() if term}, value


def solve_linear(equations, variable):
    """Return the requested coordinate if all solutions give it the same value."""
    if not 1 <= len(equations) <= 12 or not VARIABLE.fullmatch(variable):
        raise ValueError("supply up to twelve equations and one variable")
    parsed = []
    names = {variable}
    for equation in equations:
        left, sep, right = equation.partition("=")
        if not sep:
            raise ValueError("an equation needs an equals sign")
        terms, constant = affine(f"({left.strip()})-({right.strip()})")
        parsed.append((terms, -constant))
        names.update(terms)
    if len(names) > 8:
        raise ValueError("too many variables")
    names = sorted(names)
    matrix = [[terms.get(name, Fraction(0)) for name in names] + [value] for terms, value in parsed]
    pivots = {}
    for col in range(len(names)):
        rank = len(pivots)
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][col]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        row = matrix[rank]
        divisor = row[col]
        row[:] = [value / divisor for value in row]
        for i, other in enumerate(matrix):
            if i != rank and other[col]:
                scale = other[col]
                other[:] = [a - scale * b for a, b in zip(other, row, strict=True)]
        pivots[col] = rank
    if any(not any(row[:-1]) and row[-1] for row in matrix):
        raise NoUniqueAnswer("logical_error", "the equations are inconsistent")
    target = names.index(variable)
    if target not in pivots or any(matrix[pivots[target]][j] for j in range(len(names)) if j not in pivots):
        raise NoUniqueAnswer("missing_information", "the requested variable is not uniquely determined")
    return str(matrix[pivots[target]][-1])


def solve_congruences(congruences, lower, upper, first=False):
    """Combine compatible modular constraints, including non-coprime moduli."""
    if not 1 <= len(congruences) <= 8 or max(abs(lower), abs(upper)) > 10**18:
        raise ValueError("invalid bounds or too many congruences")
    if lower > upper:
        raise NoUniqueAnswer("logical_error", "the interval is empty")
    residue, modulus = 0, 1
    for item in congruences:
        if len(item) != 2:
            raise ValueError("each constraint needs a residue and modulus")
        value, period = item
        if not period:
            raise NoUniqueAnswer("undefined_math", "modulo zero is undefined")
        if not 1 <= period <= 10**12 or abs(value) > 10**18:
            raise ValueError("use positive bounded moduli and residues")
        common = gcd(modulus, period)
        difference = value - residue
        if difference % common:
            raise NoUniqueAnswer("logical_error", "the residues are incompatible")
        reduced = period // common
        offset = (difference // common * pow(modulus // common, -1, reduced)) % reduced if reduced > 1 else 0
        residue += modulus * offset
        modulus *= reduced
        residue %= modulus
    candidate = lower + (residue - lower) % modulus
    if candidate > upper:
        raise NoUniqueAnswer("logical_error", "no integer satisfies the constraints in the interval")
    if not first and candidate + modulus <= upper:
        raise NoUniqueAnswer("missing_information", "multiple integers satisfy the constraints")
    return str(candidate)
