"""Bounded SQL reads over SQLite or several delimited workspace files."""
from contextlib import closing
import re
import sqlite3

from .domain import NoUniqueAnswer, local_file
from .tools import connect, delimited_rows, quote_identifier

SQLITE_EXTENSIONS = {".sqlite", ".sqlite3", ".db"}
DELIMITED = {".csv", ".tsv"}
FUNCTIONS = frozenset({
    "abs", "avg", "cast", "coalesce", "count", "dense_rank", "glob", "group_concat",
    "ifnull", "instr", "lag", "lead", "length", "like", "lower", "max", "min", "nullif",
    "rank", "replace", "round", "row_number", "substr", "sum", "total", "trim", "upper",
})


def authorize(action, arg1, arg2, database, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    # SQLite omits the database name for implicit reads such as COUNT(*).
    if action == sqlite3.SQLITE_READ and (database == "main" or (database is None and arg2 == "")):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").casefold() in FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def database_file(root, file):
    path = local_file(root, file, SQLITE_EXTENSIONS)
    if path.stat().st_size > 32_000_000:
        raise ValueError("database exceeds size limit")
    return path


def sql_catalog(root, file):
    """Return column names without sampling rows, for small SQLite schemas."""
    with connect(database_file(root, file)) as conn:
        tables = conn.execute("SELECT name FROM sqlite_schema WHERE type='table' "
                              "AND name NOT GLOB 'sqlite_*' AND sql NOT LIKE 'CREATE VIRTUAL%' "
                              "ORDER BY name LIMIT 7").fetchall()
        if len(tables) > 6:
            return {}
        catalog = {}
        for (table,) in tables:
            columns = conn.execute("SELECT name FROM pragma_table_info(?, 'main') LIMIT 13", (table,)).fetchall()
            if len(columns) > 12:
                return {}
            catalog[table] = [name for (name,) in columns]
        return catalog


def import_tables(conn, root, tables, integers, require_complete=False):
    if not 1 <= len(tables) <= 4 or set(integers) - tables.keys():
        raise ValueError("supply one to four tables and integer fields for those tables")
    size_left, rows_left = 8_000_000, 20_000
    if len({name.casefold() for name in tables}) != len(tables):
        raise ValueError("table aliases must be distinct")
    for alias, file in tables.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", alias) or alias.lower().startswith("sqlite_"):
            raise ValueError("invalid table alias")
        path = local_file(root, file, DELIMITED)
        size_left -= path.stat().st_size
        if size_left < 0:
            raise ValueError("input files exceed the combined size limit")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns, rows = delimited_rows(path, stream)
            numeric = set(integers.get(alias, ()))
            if (len(columns) > 64 or len({col.casefold() for col in columns}) != len(columns)
                    or any(len(col) > 80 or "\0" in col for col in columns) or numeric - set(columns)):
                raise ValueError("unsupported columns or unknown integer field")
            fields = ",".join(quote_identifier(col) + (" INTEGER" if col in numeric else " TEXT") for col in columns)
            table = quote_identifier(alias)
            conn.execute(f"CREATE TABLE {table} ({fields})")
            insert = f"INSERT INTO {table} VALUES ({','.join('?' for col in columns)})"
            for row in rows:
                rows_left -= 1
                if rows_left < 0:
                    raise ValueError("input tables exceed the combined row limit")
                values = []
                for col in columns:
                    value = row[col]
                    if value == "" and require_complete:
                        raise NoUniqueAnswer("missing_information", "the task declares blanks unknown; SQL must not silently discard them")
                    if col in numeric and value:
                        if not re.fullmatch(r"[+-]?[0-9]{1,19}", value) or not -(2**63) <= int(value) < 2**63:
                            raise ValueError(f"invalid integer in {alias}.{col}")
                        value = int(value)
                    values.append(value if value != "" else None)
                conn.execute(insert, values)


def sql_query(root, file, statement, mode="scalar", tables=None, integers=None, require_complete=False):
    if (len(statement) > 3500 or not re.match(r"\s*(?:SELECT|WITH)\b", statement, re.I)
            or mode not in {"scalar", "rows"}):
        raise ValueError("supply one short SELECT or WITH statement and scalar or rows mode")
    if bool(file) == bool(tables) or (file and integers):
        raise ValueError("supply a SQLite file or delimited tables with optional integer columns")
    connection = connect(database_file(root, file)) if file else closing(sqlite3.connect(":memory:"))
    with connection as conn:
        if tables:
            conn.execute("PRAGMA trusted_schema=OFF")
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 256_000)
            import_tables(conn, root, tables, integers or {}, require_complete)
            conn.set_progress_handler(lambda: 1, 2_000_000)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA cache_size=-4096")
        conn.execute("PRAGMA temp_store=FILE")
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 256_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 64)
        conn.set_authorizer(authorize)
        cursor = conn.execute(statement)
        if cursor.description is None:
            raise ValueError("the statement must return rows")
        if mode == "scalar":
            if len(cursor.description) != 1:
                raise ValueError("scalar mode requires one column")
            row = cursor.fetchone()
            if row is None or row[0] is None:
                raise NoUniqueAnswer("missing_information", "the query returned no scalar value")
            if cursor.fetchone() is not None:
                raise NoUniqueAnswer("missing_information", "the query returned multiple candidate rows")
            if type(row[0]) not in (str, int, float) or len(str(row[0])) > 1000:
                raise ValueError("query result is not a short scalar")
            if isinstance(row[0], str) and re.search(r"\bLIMIT\b", statement, re.I):
                raise NoUniqueAnswer("missing_information", "row-limited SQL cannot prove a unique text answer; return all tied candidates")
            return row[0]
        results, size = [], 0
        for row in cursor:
            values = list(row)
            if any(type(value) not in (str, int, float, type(None)) for value in values):
                raise ValueError("query returned binary data")
            size += sum(len(str(value)) for value in values)
            if len(results) >= 25 or size > 6000:
                raise ValueError("too many result rows or bytes; narrow the query")
            results.append(values)
        return {"columns": [col[0] for col in cursor.description], "rows": results}
