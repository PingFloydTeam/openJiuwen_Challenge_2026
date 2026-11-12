"""Strict bounded queries on JSON records, JSON Lines, and nested scalar fields."""
from decimal import Decimal
from fractions import Fraction
import json
import re

from .domain import InvalidInput, NoUniqueAnswer, local_file, read_text

MISSING = object()


def pointer(value, path):
    if not path:
        return value
    if not path.startswith("/") or len(path) > 500:
        raise ValueError("use JSON Pointer paths starting with /; escape ~ as ~0 and / as ~1")
    for part in path[1:].split("/"):
        if re.search(r"~(?![01])", part):
            raise ValueError("invalid JSON Pointer escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value.get(part, MISSING)
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
        if value is MISSING:
            return value
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidInput("JSON has duplicate object keys; precedence is undefined")
        result[key] = value
    return result


def reject_constant(_):
    raise InvalidInput("JSON contains a nonfinite constant")


def decimal_value(text):
    value = Decimal(text)
    if len(value.as_tuple().digits) > 100 or abs(value.adjusted()) > 100:
        raise InvalidInput("JSON number exceeds bounded exact arithmetic limits")
    return value


def integer_value(text):
    if len(text) > 101:
        raise InvalidInput("JSON integer exceeds 100 digits")
    return int(text)


def json_document(root, file):
    source = local_file(root, file, {".json", ".jsonl", ".ndjson"})
    text = read_text(source, 2_000_000)
    options = {"object_pairs_hook": unique_object, "parse_constant": reject_constant,
               "parse_float": decimal_value, "parse_int": integer_value}
    try:
        if source.suffix.lower() in {".jsonl", ".ndjson"}:
            data = [json.loads(line, **options) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text, **options)
    except RecursionError:
        raise InvalidInput("JSON nesting exceeds limits") from None
    return data


def select_records(data, path=""):
    records = pointer(data, path)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or len(records) > 20000 or any(not isinstance(row, dict) for row in records):
        raise ValueError("path must select an object or an array of <=20000 objects")
    return records


def json_records(root, file, path=""):
    return select_records(json_document(root, file), path)


def scalar(value):
    if value is MISSING or value is None:
        raise NoUniqueAnswer("missing_information", "a potentially relevant record lacks a known value")
    if not isinstance(value, (str, int, Decimal, bool)):
        raise ValueError("field must select a scalar value")
    return value


def identity(value):
    if type(value) in (int, float, Decimal, Fraction):
        return "number", Fraction(str(value))
    return type(value).__name__, value


def field_paths(row, prefix="", depth=0):
    if depth > 8:
        return []
    paths = []
    for key, value in row.items():
        path = prefix + "/" + key.replace("~", "~0").replace("/", "~1")
        paths.extend(field_paths(value, path, depth+1) if isinstance(value, dict) else [path])
    return paths[:80]


def records_catalog(root, file):
    """Describe bounded candidate record paths; this is metadata, not answer evidence."""
    data = json_document(root, file)
    rows = select_records(data)
    catalog = {"": {"records": len(rows), "fields": sorted({p for row in rows[:3] for p in field_paths(row)})[:30]}}

    def walk(value, path, depth):
        if depth > 4 or len(catalog) >= 5:
            return
        if isinstance(value, list):
            if value and all(isinstance(row, dict) for row in value[:3]):
                catalog[path] = {"records": len(value),
                                 "fields": sorted({p for row in value[:3] for p in field_paths(row)})[:30]}
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + "/" + key.replace("~", "~0").replace("/", "~1"), depth+1)

    # A root object may contain arrays. Array elements are already records;
    # their nested fields are relative to each record, not alternate root paths.
    if isinstance(data, dict):
        walk(data, "", 0)
    return catalog


def records_query(root, file, path="", filters=None, op="describe", field="", order_by="", limit=3):
    rows = json_records(root, file, path)
    filters = filters or {}
    if len(filters) > 10 or not 1 <= limit <= 10:
        raise ValueError("up to ten filters and ten preview rows are supported")
    if op == "describe":
        return {"records": len(rows), "fields": sorted({p for row in rows[:10] for p in field_paths(row)})[:80],
                "sample": json.loads(json.dumps(rows[:limit], default=str))}
    if op not in {"unique", "count", "count_distinct", "sum", "argmin", "argmax"}:
        raise ValueError("use describe, unique, count, count_distinct, sum, argmin or argmax")
    if order_by and op not in {"argmin", "argmax"}:
        raise ValueError("order_by only applies to argmin/argmax; use record_plan latest then aggregate for grouped revisions")
    chosen = []
    for row in rows:
        missing, mismatch = False, False
        for key, expected in filters.items():
            value = pointer(row, key)
            missing |= value is MISSING
            if value is not MISSING and identity(value) != identity(expected):
                mismatch = True
        if mismatch:
            continue
        if missing:
            raise NoUniqueAnswer("missing_information", "filter depends on a missing JSON field")
        chosen.append(row)
    if op == "count":
        return len(chosen)
    if not field:
        raise ValueError("a JSON Pointer field is required")
    if op in {"argmin", "argmax"}:
        if not order_by:
            raise ValueError("argmin/argmax needs a JSON Pointer order_by")
        ranks = [scalar(pointer(row, order_by)) for row in chosen]
        if len({identity(rank)[0] for rank in ranks}) > 1:
            raise ValueError("order values have incompatible types")
        if ranks:
            best = (min if op == "argmin" else max)(ranks)
            chosen = [row for row, rank in zip(chosen, ranks, strict=True) if rank == best]
    values = [scalar(pointer(row, field)) for row in chosen]
    if op == "sum":
        if any(type(value) not in (int, Decimal) for value in values):
            raise ValueError("sum requires numeric JSON values")
        return str(sum((Fraction(value) for value in values), Fraction(0)))
    unique = {identity(value): value for value in values}
    if op == "count_distinct":
        return len(unique)
    if len(unique) != 1:
        raise NoUniqueAnswer("missing_information", "matching JSON records do not determine one value")
    result = next(iter(unique.values()))
    if type(result) is bool:
        return "true" if result else "false"
    return str(result) if isinstance(result, Decimal) else result
