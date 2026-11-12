"""Bounded relational plans over typed JSON and exact decimal CSV data.

A small declarative IR: filter -> latest-per-key -> join -> filter -> aggregate.
Order is explicit; missing data is checked only where an operation depends on it.
No Python, SQL, or dynamically executable strings are accepted.
"""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import operator

from .domain import NoUniqueAnswer, local_file
from .records import MISSING, identity, json_records, pointer, scalar
from .tools import delimited_rows

OPS = {'=': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le,
       '>': operator.gt, '>=': operator.ge}


def value(row, field):
    if not isinstance(field, str) or not field or len(field) > 500:
        raise ValueError('field must be a short name or JSON Pointer')
    return scalar(pointer(row, field) if field.startswith('/') else row.get(field, MISSING))


def number(v):
    if type(v) not in (int, Decimal, Fraction, str):
        raise ValueError('numeric fields must be finite exact numbers')
    if len(str(v)) > 120:
        raise ValueError('numeric field exceeds limit')
    try:
        result = Fraction(v)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        raise ValueError('invalid exact numeric field') from None
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 512:
        raise ValueError('numeric field exceeds 512 bits')
    return result


def load(root, source):
    if (not isinstance(source, dict) or set(source) - {'file', 'path', 'numbers'}
            or not isinstance(source.get('file'), str) or not isinstance(source.get('path', ''), str)):
        raise ValueError('source needs file and optional path/numbers')
    numeric = source.get('numbers', [])
    if not isinstance(numeric, list) or len(numeric) > 64 or any(not isinstance(n, str) for n in numeric):
        raise ValueError('numbers must be a list of numeric field names')
    path = local_file(root, source['file'], {'.csv', '.tsv', '.json', '.jsonl', '.ndjson'})
    if path.stat().st_size > 2_000_000:
        raise ValueError('source exceeds 2 MB')
    if path.suffix.lower() in {'.csv', '.tsv'}:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            _, iterator = delimited_rows(path, stream)
            rows = []
            for row in iterator:
                rows.append({k: (v if v != '' else None) for k, v in row.items()})
                if len(rows) > 20000:
                    raise ValueError('source exceeds 20000 rows')
    else:
        rows = json_records(root, source['file'], source.get('path', ''))
    # Numeric conversion is lazy, after selection, so unrelated unknowns don't
    # invalidate an otherwise completely determined result.
    return rows, set(numeric)


def column(row, field, numeric):
    v = value(row, field)
    return number(v) if field in numeric else v


def select(rows, filters, numeric):
    if not isinstance(filters, list) or len(filters) > 12:
        raise ValueError('too many predicates')
    for item in filters:
        if not isinstance(item, list) or len(item) != 3 or not isinstance(item[1], str) or item[1] not in OPS:
            raise ValueError('predicate is [field,comparison,value]')
    result = []
    for row in rows:
        unknown, mismatch = False, False
        for field, op, wanted in filters:
            try:
                got = column(row, field, numeric)
            except NoUniqueAnswer:
                unknown = True
                continue
            wanted = number(wanted) if field in numeric else wanted
            if op in {'=', '!='}:
                same = identity(got) == identity(wanted)
                ok = same if op == '=' else not same
            else:
                if type(got) != type(wanted) and not isinstance(got, (int, Decimal, Fraction)):
                    raise ValueError('ordered predicate has incompatible types')
                ok = OPS[op](got, wanted)
            mismatch |= not ok
        if mismatch:
            continue
        if unknown:
            raise NoUniqueAnswer('missing_information', 'selection depends on an unknown field')
        result.append(row)
    return result


def record_plan(root, source, steps, op, field='', factor='', group=''):
    if not isinstance(source, dict) or not isinstance(steps, list) or len(steps) > 6:
        raise ValueError('supply one source and at most six steps')
    if op not in {'unique', 'count', 'count_distinct', 'sum', 'sum_product', 'ratio_sums', 'argmax_sum', 'argmin_sum'}:
        raise ValueError('unsupported terminal operation')
    rows, numeric = load(root, source)
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('each step must be an object')
        if step.get('op') == 'argmax' and set(step) <= {'op', 'field', 'by'}:
            step = {'op': 'latest', 'by': step.get('by', []), 'order_by': step['field']}
        action = step.get('op')
        if action == 'where' and set(step) <= {'op', 'filters'}:
            rows = select(rows, step['filters'], numeric)
        elif action == 'latest' and set(step) <= {'op', 'by', 'order_by'}:
            keys, rank = step['by'], step['order_by']
            if not isinstance(keys, list) or len(keys) > 8:
                raise ValueError('latest by is a list of group fields, [] for one global group')
            groups = {}
            for row in rows:
                key = tuple(identity(column(row, k, numeric)) for k in keys)
                r = column(row, rank, numeric)
                old = groups.get(key)
                if old is None or r > old[0]:
                    groups[key] = (r, [row])
                elif r == old[0]:
                    old[1].append(row)
            rows = [row for _, tied in groups.values() for row in tied]
        elif action == 'join' and set(step) <= {'op', 'source', 'left', 'right', 'prefix', 'how'}:
            other, other_numeric = load(root, step['source'])
            prefix = step.get('prefix', 'joined')
            if not isinstance(prefix, str) or not prefix.isidentifier() or step.get('how', 'inner') not in {'inner', 'anti'}:
                raise ValueError('join needs a simple prefix and inner/anti mode')
            index = {}
            for row in other:
                index.setdefault(identity(column(row, step['right'], other_numeric)), []).append(row)
            joined = []
            for row in rows:
                matches = index.get(identity(column(row, step['left'], numeric)), [])
                if step.get('how', 'inner') == 'anti':
                    if not matches:
                        joined.append(row)
                else:
                    if prefix in row:
                        raise ValueError('join prefix would overwrite a source field')
                    for match in matches:
                        joined.append({**row, prefix: match})
                        if len(joined) > 20000:
                            raise ValueError('join exceeds 20000 rows')
            rows = joined
            if step.get('how', 'inner') != 'anti':
                numeric |= {'/'+prefix+(f if f.startswith('/') else '/'+f.replace('~','~0').replace('/','~1')) for f in other_numeric}
        else:
            raise ValueError('steps support where, latest and join; no unknown keys')
    if op == 'count':
        return len(rows)
    if not field:
        raise ValueError('terminal operation needs field')
    values = [column(row, field, numeric) for row in rows]
    if op == 'count_distinct':
        return len({identity(v) for v in values})
    if op == 'unique':
        distinct = {identity(v): v for v in values}
        if len(distinct) != 1:
            raise NoUniqueAnswer('missing_information', 'all surviving records do not agree on one value')
        answer = next(iter(distinct.values()))
        return str(answer).lower() if type(answer) is bool else str(answer)
    if op in {'argmax_sum', 'argmin_sum'}:
        if not group:
            raise ValueError('grouped extremum needs group')
        totals = {}
        labels = {}
        for row, v in zip(rows, values, strict=True):
            label = column(row, group, numeric)
            k = identity(label)
            totals[k] = totals.get(k, Fraction(0)) + number(v)
            labels[k] = label
        if not totals:
            raise NoUniqueAnswer('missing_information', 'no groups')
        best = (max if op == 'argmax_sum' else min)(totals.values())
        winners = [labels[k] for k,v in totals.items() if v == best]
        if len(winners) != 1:
            raise NoUniqueAnswer('missing_information', 'group totals tie without a priority')
        return str(winners[0])
    total = sum((number(v) for v in values), Fraction(0))
    if op == 'sum':
        return str(total)
    if not factor:
        raise ValueError('this aggregate also needs factor')
    factors = [number(column(row, factor, numeric)) for row in rows]
    if op == 'sum_product':
        return str(sum((number(v)*f for v,f in zip(values, factors, strict=True)), Fraction(0)))
    denominator = sum(factors, Fraction(0))
    if not denominator:
        raise NoUniqueAnswer('undefined_math', 'ratio denominator is zero')
    return str(total/denominator)
