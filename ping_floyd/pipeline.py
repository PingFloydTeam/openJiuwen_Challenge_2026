"""Bounded relational plans over typed JSON and exact decimal CSV data.

A small declarative IR: filter -> latest-per-key -> join -> filter -> aggregate.
Order is explicit; missing data is checked only where an operation depends on it.
No Python, SQL, or dynamically executable strings are accepted.
"""
from decimal import Decimal, InvalidOperation
from datetime import date, datetime, timezone
from fractions import Fraction
import operator

from .domain import NoUniqueAnswer, local_file
from .records import MISSING, identity, json_records, pointer, scalar
from .tools import delimited_rows

OPS = {'=': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le,
       '>': operator.gt, '>=': operator.ge}
TYPES = {'string', 'identifier', 'integer', 'decimal', 'number', 'date', 'datetime'}


def typed(v, kind, field):
    """Convert only explicitly declared types; never guess identifier semantics."""
    if kind in {'decimal', 'number', 'integer'}:
        try:
            result = number(v)
        except ValueError as error:
            raise ValueError(f'{field}: expected {kind}; dates need types {{"{field}":"date"}} or datetime') from error
        if kind == 'integer' and result.denominator != 1:
            raise ValueError(f'{field}: expected integer, got a fraction')
        return result
    if kind in {'string', 'identifier'}:
        if not isinstance(v, str):
            raise ValueError(f'{field}: expected string; conversion could merge distinct identifiers')
        return v
    if not isinstance(v, str):
        raise ValueError(f'{field}: expected ISO {kind} string')
    try:
        if kind == 'date':
            if len(v) != 10:
                raise ValueError('date needs YYYY-MM-DD')
            return date.fromisoformat(v).isoformat()
        parsed = datetime.fromisoformat(v.replace('Z', '+00:00'))
        if parsed.utcoffset() is None:
            raise ValueError('datetime needs an explicit UTC offset')
        return parsed.astimezone(timezone.utc).isoformat(timespec='microseconds')
    except (ValueError, OverflowError) as error:
        raise ValueError(f'{field}: invalid ISO {kind}: {error}') from None


def value(row, field):
    if not isinstance(field, str) or not field or len(field) > 500:
        raise ValueError('field must be a short name or JSON Pointer')
    selected = pointer(row, field) if field.startswith('/') else row.get(field, MISSING)
    # Accept an unambiguous nested shorthand. Literal CSV/JSON keys containing
    # slashes retain precedence, including explicit unknown values.
    if not field.startswith('/') and '/' in field and field not in row:
        nested = pointer(row, '/' + field)
        if nested is not MISSING:
            selected = nested
    return selected if isinstance(selected, Fraction) else scalar(selected)


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
    if (not isinstance(source, dict) or set(source) - {'file', 'path', 'numbers', 'types'}
            or not isinstance(source.get('file'), str) or not isinstance(source.get('path', ''), str)):
        raise ValueError('source needs file and optional path/numbers')
    numeric = source.get('numbers', [])
    if not isinstance(numeric, list) or len(numeric) > 64 or any(not isinstance(n, str) for n in numeric):
        raise ValueError('numbers must be a list of numeric field names')
    declared = source.get('types', {})
    if (not isinstance(declared, dict) or len(declared) > 64
            or any(not isinstance(k, str) or not isinstance(v, str) or v not in TYPES for k,v in declared.items())):
        raise ValueError('types maps fields to string/identifier/integer/decimal/number/date/datetime')
    if any(declared.get(k, 'number') not in {'number', 'decimal', 'integer'} for k in numeric):
        raise ValueError('numbers conflicts with types; dates and identifiers are not numeric')
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
    return rows, {**dict.fromkeys(numeric, 'number'), **declared}


def column(row, field, numeric):
    v = value(row, field)
    kind = numeric.get(field)
    if kind is None and '/' in field and not field.startswith('/') and field not in row:
        kind = numeric.get('/' + field)
    return typed(v, kind, field) if kind else v


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
            wanted = typed(wanted, numeric[field], field) if field in numeric else wanted
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


def grouped(rows, step, types):
    """Materialize relational group aggregates for further filtering/ranking."""
    keys, metrics = step.get('by'), step.get('aggregates')
    if (not isinstance(keys, list) or len(keys) > 8 or any(not isinstance(k,str) or not k for k in keys)
            or len(set(keys)) != len(keys) or any(k.startswith('/') for k in keys)):
        raise ValueError('group by needs distinct plain field names (max 8)')
    if not isinstance(metrics, dict) or not 1 <= len(metrics) <= 8:
        raise ValueError('group needs aggregates {alias:{op:sum|count|min|max,field:...}}')
    for alias, spec in metrics.items():
        if (not isinstance(alias, str) or not alias.isidentifier() or alias in keys
                or not isinstance(spec, dict) or set(spec) - {'op','field'}
                or spec.get('op') not in {'sum','count','min','max'}
                or (spec['op'] != 'count' and not isinstance(spec.get('field'),str))
                or (spec['op'] == 'count' and 'field' in spec)):
            raise ValueError('invalid group aggregate; count counts rows and takes no field')
    buckets = {}
    for row in rows:
        labels = [column(row,k,types) for k in keys]
        signature = tuple(identity(v) for v in labels)
        if signature not in buckets:
            buckets[signature] = (dict(zip(keys,labels)), [])
        buckets[signature][1].append(row)
    if not keys and not rows:
        buckets[()] = ({}, [])
    result = []
    for labels, members in buckets.values():
        row = dict(labels)
        for alias,spec in metrics.items():
            action = spec['op']
            if action == 'count':
                row[alias] = len(members)
            else:
                values = [number(column(r,spec['field'],types)) for r in members]
                if not values and action != 'sum':
                    raise NoUniqueAnswer('missing_information','min/max of empty group')
                row[alias] = sum(values,Fraction(0)) if action == 'sum' else (min if action == 'min' else max)(values)
        result.append(row)
    return result, {**{k:types[k] for k in keys if k in types}, **dict.fromkeys(metrics, 'number')}


def record_plan(root, source, steps, op, field='', factor='', group='', order_by=''):
    if not isinstance(source, dict) or not isinstance(steps, list) or len(steps) > 12:
        raise ValueError('supply one source and at most twelve steps')
    if op not in {'unique', 'count', 'count_distinct', 'sum', 'sum_product', 'ratio_sums', 'argmax_sum', 'argmin_sum', 'argmax', 'argmin'}:
        raise ValueError('unsupported terminal operation')
    if order_by and op not in {'argmax','argmin'}:
        raise ValueError('order_by applies only to terminal argmax/argmin')
    rows, numeric = load(root, source)
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('each step must be an object')
        action = step.get('op')
        if action == 'where' and set(step) <= {'op', 'filters'}:
            rows = select(rows, step['filters'], numeric)
        elif action == 'group' and set(step) <= {'op','by','aggregates'}:
            rows, numeric = grouped(rows, step, numeric)
        elif (action == 'latest' and set(step) <= {'op','by','order_by'} or
              action in {'argmax','argmin'} and set(step) <= {'op','by','field','order_by'}):
            keys, rank = step.get('by', []), step.get('order_by', step.get('field'))
            if not isinstance(rank,str) or not rank:
                raise ValueError('ranking stage needs order_by')
            if action in {'argmax','argmin'} and 'field' in step and step['field'] != rank:
                raise ValueError('ranking stage retains rows; field alias must equal order_by')
            if not isinstance(keys, list) or len(keys) > 8:
                raise ValueError('latest by is a list of group fields, [] for one global group')
            groups = {}
            for row in rows:
                key = tuple(identity(column(row, k, numeric)) for k in keys)
                r = column(row, rank, numeric)
                old = groups.get(key)
                if old is None or (r < old[0] if action == 'argmin' else r > old[0]):
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
                numeric.update({'/'+prefix+(f if f.startswith('/') else '/'+f.replace('~','~0').replace('/','~1')):kind
                                for f,kind in other_numeric.items()})
        else:
            raise ValueError('steps support where/latest/join/group; group needs by and aggregates, e.g. {"total":{"op":"sum","field":"amount"}}')
    if op == 'count':
        return len(rows)
    if not field:
        raise ValueError('terminal operation needs field')
    if op == 'sum_product':
        if not factor:
            raise ValueError('sum_product needs factor')
        total=Fraction(0)
        for row in rows:
            operands=[]
            for key in (field,factor):
                try:
                    operands.append(number(column(row,key,numeric)))
                except NoUniqueAnswer:
                    operands.append(None)
            if 0 in operands:
                continue
            if None in operands:
                raise NoUniqueAnswer('missing_information','nonzero product depends on an unknown value')
            total+=operands[0]*operands[1]
        return str(total)
    if op in {'argmax','argmin'}:
        if not order_by:
            raise ValueError('argmax/argmin needs order_by (rank) and field (answer)')
        ranked = [(column(r,order_by,numeric),r) for r in rows]
        if not ranked:
            raise NoUniqueAnswer('missing_information','no candidates')
        best = (max if op == 'argmax' else min)(v for v,_ in ranked)
        rows = [r for v,r in ranked if v == best]
        op = 'unique'
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
