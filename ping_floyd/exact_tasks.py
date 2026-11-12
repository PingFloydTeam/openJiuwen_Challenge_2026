"""Bounded exact operations; no natural-language inference or model calls."""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from math import comb
import math
import re

from .domain import NoUniqueAnswer


def rational(value):
    if type(value) is float and math.isfinite(value):
        value = str(value)
    if type(value) is int and abs(value) <= 10**15:
        return Fraction(value)
    if isinstance(value, str) and len(value) <= 80 and re.fullmatch(r"[+-]?\d+(?:/\d+|\.\d+)?", value):
        result = Fraction(value)
        if max(abs(result.numerator), result.denominator) <= 10**18:
            return result
    raise ValueError("use a bounded integer or exact decimal/fraction string")


def interval_coverage(intervals, minimum=1):
    """Length covered by at least minimum intervals, with exact rational endpoints."""
    if not 1 <= minimum <= 1000 or len(intervals) > 1000:
        raise ValueError("supply at most 1000 intervals and a positive coverage count")
    events = defaultdict(int)
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("each interval needs a start and an end")
        start, end = map(rational, interval)
        if start > end:
            raise NoUniqueAnswer("logical_error", "interval ends before it starts")
        events[start] += 1
        events[end] -= 1
    total, active, previous = Fraction(0), 0, None
    for point in sorted(events):
        if previous is not None and active >= minimum:
            total += point - previous
        active += events[point]
        previous = point
    return str(total)


def elapsed_time(start, end, unit="minutes"):
    """Elapsed duration between ISO timestamps with explicit UTC offsets."""
    parsed = []
    for stamp in (start, end):
        if not re.fullmatch(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d{1,6})?)?(?:Z|[+-]\d\d:\d\d)", stamp):
            raise ValueError("timestamps must include a date, clock time and explicit UTC offset")
        parsed.append(datetime.fromisoformat(stamp).astimezone(timezone.utc))
    delta = parsed[1] - parsed[0]
    if delta.days < 0:
        raise NoUniqueAnswer("logical_error", "finish precedes start")
    scale = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}.get(unit)
    if scale is None:
        raise ValueError("unit must be seconds, minutes, hours or days")
    micros = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
    return str(Fraction(micros, scale * 1000000))


def urn_probability(counts, draw, distinct):
    """Probability of exactly k categories in a uniform draw without replacement."""
    if isinstance(counts, dict):
        counts = list(counts.values())
    if (not 1 <= len(counts) <= 32 or any(type(n) is not int or not 0 <= n <= 100000 for n in counts)
            or type(draw) is not int or not 0 <= draw <= 60 or not 0 <= distinct <= len(counts)):
        raise ValueError("urn limits: 32 categories, counts <=100000, draw <=60")
    total = sum(counts)
    if draw > total:
        raise NoUniqueAnswer("logical_error", "cannot draw more objects than exist without replacement")
    ways = {(0, 0): 1}
    for count in counts:
        updated = defaultdict(int)
        binom = [comb(count, n) for n in range(min(draw, count) + 1)]
        for (picked, kinds), weight in ways.items():
            for n in range(min(count, draw - picked) + 1):
                next_kinds = kinds + (n > 0)
                if next_kinds <= distinct:
                    updated[picked + n, next_kinds] += weight * binom[n]
        ways = updated
    return str(Fraction(ways.get((draw, distinct), 0), comb(total, draw)))


def integer_inverse(operations, target, modulus=1, residue=0):
    """Invert add/mul/floor_div on integers and require a unique congruent preimage."""
    if (type(target) is not int or abs(target) > 10**15 or not 1 <= modulus <= 10**9
            or abs(residue) > 10**9 or not 1 <= len(operations) <= 20):
        raise ValueError("integer inverse arguments exceed limits")
    lower = upper = target
    for operation in reversed(operations):
        if len(operation) != 2:
            raise ValueError("each operation is [name, integer operand]")
        name, operand = operation
        if type(operand) is not int or abs(operand) > 10**6:
            raise ValueError("integer operand exceeds limits")
        if name == "add":
            lower, upper = lower - operand, upper - operand
        elif name == "mul" and operand:
            if operand < 0:
                lower, upper, operand = -upper, -lower, -operand
            lower, upper = -((-lower) // operand), upper // operand
        elif name == "floor_div" and operand > 0:
            lower, upper = lower * operand, (upper + 1) * operand - 1
        else:
            raise ValueError("supported: add, nonzero mul, floor_div by a positive integer")
        if lower > upper:
            raise NoUniqueAnswer("logical_error", "no integer input produces the stated result")
    candidate = lower + (residue - lower) % modulus
    if candidate > upper:
        raise NoUniqueAnswer("logical_error", "no input satisfies the stated congruence")
    if candidate + modulus <= upper:
        raise NoUniqueAnswer("missing_information", "multiple integer inputs satisfy the conditions")
    return candidate
