"""Cheap, typed tools. Task data is never used as executable instructions."""
import ast
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import itertools
import operator
from pathlib import Path
import re
import sqlite3

from .domain import InvalidInput, NoUniqueAnswer, files, local_file, read_text

OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.Pow: operator.pow}
TABLES = {".csv", ".tsv", ".sqlite", ".sqlite3", ".db"}
TEXT = {".md", ".txt", ".json", ".py", ".csv", ".tsv", ".log"}


def calc(expr: str):
    if len(expr) > 384:
        raise ValueError("expression too long")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = Fraction(node.value)
        elif isinstance(node, ast.Constant) and type(node.value) is float:
            literal = ast.get_source_segment(expr, node)
            exponent = re.search(r"[eE]([+-]?[0-9_]+)$", literal)
            if exponent and abs(int(exponent[1].replace("_", ""))) > 1024:
                raise ValueError("decimal exponent too large")
            result = Fraction(literal)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            result = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATIONS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (right.denominator != 1 or abs(right) > 32):
                raise ValueError("power requires a small integer exponent")
            result = Fraction(OPERATIONS[type(node.op)](left, right))
        else:
            raise ValueError("allowed: numeric literals and + - * / // % **")
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
            raise ValueError("result too large")
        return result

    return str(evaluate(ast.parse(expr, mode="eval").body))


def db_schema(root: Path, file: str, table: str = ""):
    path = local_file(root, file, TABLES)
    if path.suffix.lower() in {".csv", ".tsv"}:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns, rows = delimited_rows(path, stream)
            return {"columns": columns, "sample": list(itertools.islice(rows, 2))}
    with connect(path) as conn:
        if not table:
            return table_names(conn)
        cursor = conn.execute("SELECT * FROM " + quote_identifier(resolve_table(conn, table)) + " LIMIT 2")
        return {"columns": [col[0] for col in cursor.description],
                "sample": [dict(row) for row in cursor]}


def table_names(conn):
    return [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_schema WHERE type IN ('table','view') AND sql NOT LIKE 'CREATE VIRTUAL%' LIMIT 40")]


def resolve_table(conn, table):
    record = conn.execute(
        "SELECT name, type, sql FROM sqlite_schema WHERE name=? COLLATE NOCASE AND type IN ('table','view')",
        (table,),
    ).fetchone()
    if record is None:
        raise NoUniqueAnswer("missing_information", "requested table is absent")
    name, kind, definition = record
    if not definition or (kind == "table" and definition.upper().startswith("CREATE VIRTUAL")):
        raise ValueError("virtual tables are unsupported")
    return name


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def connect(path):
    from contextlib import closing
    wal = path.with_name(path.name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise ValueError("SQLite WAL must be checkpointed before immutable access")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.execute("PRAGMA trusted_schema=OFF")
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2_000_000)
    conn.set_progress_handler(lambda: 1, 2_000_000)
    conn.row_factory = sqlite3.Row
    return closing(conn)


def delimited_rows(path, stream):
    reader = csv.reader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",", strict=True)
    columns = next(reader, [])
    if not columns or any(not field.strip() for field in columns) or len(set(columns)) != len(columns):
        raise InvalidInput("table needs unique, nonempty column names")

    def rows():
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                raise InvalidInput("row width differs from header")
            yield dict(zip(columns, row, strict=True))

    return columns, rows()


def decimal(value):
    literal = str(value)
    if len(literal) > 256:
        raise ValueError("numeric value too long")
    number = Decimal(literal)
    if not number.is_finite() or abs(number.adjusted()) > 100:
        raise ValueError("unsupported numeric value")
    return number


def exact_number(value):
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    scale = 0
    while denominator > 1 and denominator % 10 == 0:
        denominator //= 10
        scale += 1
    factor = 2 if denominator % 2 == 0 else 5
    while denominator % factor == 0:
        denominator //= factor
        scale += 1
    if denominator != 1:
        return str(value)
    digits = str(abs(value.numerator) * (10 ** scale // value.denominator)).zfill(scale + 1)
    sign = "-" if value.numerator < 0 else ""
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}"


def db_query(root: Path, file: str, op: str, column: str = "", filters: list | None = None,
             table: str = "", limit: int = 5, factor: str = "", group: str = "", tie_break: str = ""):
    """Filter with [column, operator, value] triples; numeric operands compare numerically."""
    path = local_file(root, file, TABLES)
    if path.stat().st_size > 32_000_000:
        raise ValueError("table exceeds size limit")
    grouped = op in {"argmax_sum", "argmin_sum"}
    if op not in {"rows", "count", "sum", "avg", "min", "max", "distinct",
                  "count_distinct", "unique", "sum_product", "ratio_sums"} and not grouped:
        raise ValueError("invalid operation")
    if not 1 <= limit <= 20:
        raise ValueError("invalid operation or limit")
    if tie_break and (not grouped or tie_break != "alphabetical_first"):
        raise ValueError("alphabetical_first is supported only for grouped extrema")

    comparisons = {"=": operator.eq, "!=": operator.ne, "<": operator.lt, ">": operator.gt,
                   "<=": operator.le, ">=": operator.ge}
    conditions = filters or []
    if len(conditions) > 10:
        raise ValueError("too many filters")

    def run(columns, rows, delimited=False):
        if any(len(rule) != 3 or rule[1] not in comparisons for rule in conditions):
            raise ValueError("invalid filter")
        required = ([column] if op not in {"rows", "count"} else []) + [rule[0] for rule in conditions]
        if op in {"sum_product", "ratio_sums"}:
            required.append(factor)
        if grouped:
            required.append(group)
        if any(field not in columns for field in required):
            raise NoUniqueAnswer("missing_information", "required column absent from table")

        def matches(row):
            missing_field = None
            for field, comparator, expected in conditions:
                value = row[field]
                if value is None or (delimited and value == "" and expected != ""):
                    if delimited:
                        missing_field = field
                        continue
                    return False  # SQL comparison with NULL is unknown and excluded by WHERE.
                try:
                    if isinstance(expected, (int, float, Decimal)):
                        value, expected = decimal(value), decimal(expected)
                    elif value is not None:
                        value = str(value)
                    if not comparisons[comparator](value, expected):
                        return False
                except (InvalidOperation, TypeError, ValueError) as error:
                    raise ValueError(f"cannot compare a value in {field}") from error
            if missing_field:
                raise NoUniqueAnswer("missing_information", f"matching depends on a blank {missing_field} cell")
            return True

        def number(row, field):
            value = row[field]
            if value in (None, ""):
                if delimited:
                    raise NoUniqueAnswer("missing_information", f"matching row has no {field} value")
                return None
            return Fraction(decimal(value))

        selected = (dict(row) for row in rows if matches(row))
        if op == "rows":
            return list(itertools.islice(selected, limit))
        if op == "count":
            return sum(1 for _ in selected)
        if grouped:
            totals = {}
            for row in selected:
                label = row[group]
                if label in (None, ""):
                    raise NoUniqueAnswer("missing_information", "matching row has no group label")
                value = number(row, column)
                if value is not None:
                    totals[label] = totals.get(label, Fraction(0)) + value
            if not totals:
                raise NoUniqueAnswer("undefined_math", "there are no groups to compare")
            best = (max if op == "argmax_sum" else min)(totals.values())
            winners = [label for label, total in totals.items() if total == best]
            if tie_break == "alphabetical_first" and len(winners) > 1:
                first = min(str(label).casefold() for label in winners)
                winners = [label for label in winners if str(label).casefold() == first]
            if len(winners) != 1:
                raise NoUniqueAnswer("missing_information", "multiple groups share the requested extreme")
            return winners[0]
        if op == "ratio_sums":
            numerator = denominator = Fraction(0)
            for row in selected:
                left, right = number(row, column), number(row, factor)
                if left is None or right is None:
                    raise NoUniqueAnswer("missing_information", "ratio requires complete paired values")
                numerator += left
                denominator += right
            if not denominator:
                raise NoUniqueAnswer("undefined_math", "sum of denominators is zero")
            return str(numerator / denominator)
        if op == "sum_product":
            total = Fraction(0)
            for row in selected:
                if row[column] in (None,'') and number(row,factor) == 0 or row[factor] in (None,'') and number(row,column) == 0:
                    continue
                left, right = number(row, column), number(row, factor)
                if left is not None and right is not None:
                    total += left * right
            return exact_number(total)
        if op in {"distinct", "count_distinct", "unique"}:
            values = {}
            for row in selected:
                value = row[column]
                if value in (None, ""):
                    if op == "unique" or (delimited and op == "count_distinct"):
                        raise NoUniqueAnswer("missing_information", "matching row has no requested value")
                    continue
                values[value] = None
                if op == "unique" and len(values) > 1:
                    raise NoUniqueAnswer("logical_error", "matching rows disagree on the requested value")
                if op == "distinct" and len(values) == limit:
                    break
            if op == "count_distinct":
                return len(values)
            if op == "unique":
                if not values:
                    raise NoUniqueAnswer("missing_information", "no matching row has the requested value")
                return next(iter(values))
            return list(values)
        values = (value for row in selected if (value := number(row, column)) is not None)
        if op in {"sum", "avg"}:
            total = Fraction(0)
            count = 0
            for value in values:
                total += value
                count += 1
            if op == "avg":
                return exact_number(total / count) if count else None
            return exact_number(total)
        extremum = (min if op == "min" else max)(values, default=None)
        return exact_number(extremum) if extremum is not None else None

    if path.suffix.lower() in {".csv", ".tsv"}:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns, rows = delimited_rows(path, stream)
            return run(columns, rows, delimited=True)
    with connect(path) as conn:
        cursor = conn.execute("SELECT * FROM " + quote_identifier(resolve_table(conn, table)))
        return run([item[0] for item in cursor.description], cursor)


FACT_LINE = re.compile(r"(?:[-*] +)?([A-Za-z][\w ]*):[ \t]*(.*)")


def markdown_records(root: Path, strict=False, max_total=8_000_000):
    if strict and root.is_symlink():
        raise ValueError("symlinked knowledge base is unsupported")
    total = 0
    for path in files(root):
        if path.suffix.lower() != ".md":
            if strict:
                raise ValueError("knowledge base has files outside the structured Markdown format")
            continue
        size = path.stat().st_size
        if size > 64_000:
            if strict:
                raise ValueError("knowledge base record exceeds size limit")
            continue
        total += size
        if total > max_total:
            raise ValueError("knowledge base exceeds scan limit")
        content = read_text(path, 64_000)
        heading = re.match(r"#[ \t]+([^\n]+)", content)
        if not heading:
            if strict:
                raise ValueError("knowledge base has unstructured records")
            continue
        fields, labels = {}, {}
        for line in content[heading.end():].splitlines():
            if not line.strip():
                continue
            fact = FACT_LINE.fullmatch(line.strip())
            if not fact:
                if strict:
                    raise ValueError("knowledge base has unstructured records")
                continue
            key, value = (part.strip() for part in fact.groups())
            key = labels.setdefault(key.casefold(), key)
            if key not in fields or fields[key] == "":
                fields[key] = value
            elif value and fields[key] != value:
                fields[key] = None
        yield heading[1].strip(), {"file": str(path.relative_to(root)), "facts": fields}


def kb_facts(root: Path, entity: str):
    return list(itertools.islice(
        (record for name, record in markdown_records(root) if name.casefold() == entity.strip().casefold()), 5))


def kb_lookup(root: Path, entity: str, fields: list[str], strict=False):
    """Follow a short chain of fields and return a scalar only when it is unique."""
    if not 1 <= len(fields) <= 8:
        raise ValueError("provide one to eight fields")
    index = {}
    for name, record in markdown_records(root, strict):
        index.setdefault(name, []).append({key.casefold(): value for key, value in record["facts"].items()})
    entities = {entity.strip()}
    for label in fields:
        field = label.strip().casefold()
        next_entities = set()
        for current in entities:
            names = [current] if current in index else [name for name in index if name.casefold() == current.casefold()]
            if len(names) > 1:
                raise NoUniqueAnswer("missing_information", "entity spelling is ambiguous")
            records = index.get(names[0], []) if names else []
            if any(field in record and record[field] is None for record in records):
                raise NoUniqueAnswer("logical_error", f"conflicting {label} for {current}")
            values = {record[field] for record in records if record.get(field)}
            if not values:
                if strict:
                    raise ValueError(f'unsupported field mapping: no exact {label} for {current}; inspect other fields')
                raise NoUniqueAnswer("missing_information", f"no {label} for {current}")
            next_entities.update(values)
        if len(next_entities) > 128:
            raise ValueError("too many alternative links")
        entities = next_entities
    if len(entities) != 1:
        raise NoUniqueAnswer("logical_error", "alternative paths give different final values")
    return entities.pop()


def related_records(root: Path, instruction: str):
    """Short context for explicitly named entities and their unambiguous links."""
    index = {}
    for name, record in itertools.islice(markdown_records(root), 250):
        index.setdefault(name.casefold(), []).append({"entity": name, **record})
    text = instruction.casefold()
    spans = []
    for name in index:
        if len(name) < 3:
            continue
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for match in itertools.islice(re.finditer(pattern, text), 3):
            spans.append((match.start(), match.end(), name))
    selected = []
    for start, end, name in sorted(spans, key=lambda span: (span[0] - span[1], span[0])):
        if not any(start < last and first < end for first, last, _ in selected):
            selected.append((start, end, name))
    roots = list(dict.fromkeys(name for _, _, name in sorted(selected)))
    if not 1 <= len(roots) <= 2:
        return []
    queue = [(name, 0) for name in roots]
    result, seen = [], set()
    for name, depth in queue:
        if name in seen or len(result) >= 6:
            continue
        seen.add(name)
        records = index.get(name, [])
        result.extend(records[:2])
        if depth >= 2 or len(records) != 1:
            continue
        for label, target in records[0]["facts"].items():
            if label.casefold() in {"mentor", "employer"} and target and target.casefold() in index:
                queue.append((target.casefold(), depth + 1))
    return result[:6]


def grep(root: Path, query: str):
    if not query.strip() or len(query) > 200:
        raise ValueError("enter a short literal")
    matches = []
    for path in files(root):
        if path.suffix.lower() not in TEXT or path.stat().st_size > 512_000:
            continue
        for number, line in enumerate(read_text(path).splitlines(), 1):
            if query.casefold() in line.casefold():
                matches.append([str(path.relative_to(root)), number, line[:300]])
                if len(matches) == 10:
                    return matches
    return matches


def excerpt(root: Path, file: str, start: int = 1, lines: int = 50):
    if start < 1 or not 1 <= lines <= 100:
        raise ValueError("invalid line range")
    text = read_text(local_file(root, file, TEXT))
    return "\n".join(f"{n}: {line}" for n, line in enumerate(
        text.splitlines()[start - 1:start - 1 + lines], start))[:4500]
