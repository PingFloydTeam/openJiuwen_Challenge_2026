"""Deterministic operations that combine facts from several sources."""
from collections import defaultdict
from datetime import date
from io import StringIO
import re

from .domain import InvalidInput, NoUniqueAnswer, files, local_directory, local_file, read_text
from .tools import FACT_LINE, db_query, delimited_rows, kb_lookup, markdown_records

DELIMITED = {".csv", ".tsv"}
TABLE = r"[\w./-]+\.(?:csv|tsv)"
TEXT = r"[\w./-]+\.txt"
NAME = r"[\w -]+"


def inventory_balance(total, first_change, second_change, ratio, target=0, closing=None):
    """Use integer stocks and all supplied totals to solve a two-inventory ratio."""
    if not (0 <= total <= 10**15 and 1 <= ratio <= 10**6 and target in (0, 1)
            and max(abs(first_change), abs(second_change)) <= 10**15):
        raise ValueError("inventory inputs exceed limits")
    if closing is not None and closing != total + first_change + second_change:
        raise NoUniqueAnswer("logical_error", "the stated closing total contradicts the movements")
    first, remainder = divmod(ratio * (total + second_change) - first_change, ratio + 1)
    second = total - first
    if remainder or min(first, second, first + first_change, second + second_change) < 0:
        raise NoUniqueAnswer("logical_error", "there is no valid integer inventory")
    return (first, second)[target]


def table_rows(root, name, required):
    path = local_file(root, name, DELIMITED)
    columns, rows = delimited_rows(path, StringIO(read_text(path, 2_000_000), newline=""))
    if not required <= set(columns):
        raise NoUniqueAnswer("missing_information", "a required table column is absent")
    for index, row in enumerate(rows):
        if index >= 20_000:
            raise ValueError("table row limit exceeded")
        yield row


def net_ledger(root, registry, journal, region, exclude_states=()):
    if exclude_states is not None and (len(exclude_states) > 10 or any(not s.strip() for s in exclude_states)):
        raise ValueError("provide up to ten nonempty states to exclude")
    excluded = {state.casefold() for state in exclude_states or ()}
    regions = defaultdict(set)
    for row in table_rows(root, registry, {"account", "region"}):
        if not row["account"]:
            raise NoUniqueAnswer("missing_information", "a registry record has no account ID")
        regions[row["account"]].add(row["region"])

    latest = {}
    columns = {"order", "account", "revision", "event", "cents"}
    if excluded:
        columns.add("state")
    for row in table_rows(root, journal, columns):
        if "state" in row and exclude_states is None:
            raise ValueError("state column present: specify exclude_states (empty to include all states)")
        if excluded:
            if not row["state"].strip():
                raise NoUniqueAnswer("missing_information", "a journal row has no state for the required filter")
            if row["state"].casefold() in excluded:
                continue
        order, account, revision = row["order"], row["account"], row["revision"]
        if not order or not re.fullmatch(r"[0-9]{1,12}", revision):
            raise NoUniqueAnswer("missing_information", "an order or revision cannot be identified")
        rank = int(revision)
        previous_rank, candidates = latest.get(order, (-1, set()))
        if rank >= previous_rank:
            if rank > previous_rank:
                candidates = set()
            cents = row["cents"]
            if re.fullmatch(r"[0-9]{1,15}", cents):
                cents = str(int(cents))
            candidates.add((account, row["event"].casefold(), cents))
            latest[order] = (rank, candidates)

    totals = defaultdict(int)
    for _, candidates in latest.values():
        if len(candidates) != 1:
            raise NoUniqueAnswer("logical_error", "the latest revision has conflicting entries")
        account, event, cents = next(iter(candidates))
        if not account or not re.fullmatch(r"[0-9]{1,15}", cents) or event not in {"sale", "refund"}:
            raise NoUniqueAnswer("missing_information", "a latest revision has an unknown account, amount or event")
        totals[account] += int(cents) if event == "sale" else -int(cents)

    answer = 0
    for account, amount in totals.items():
        candidates = regions.get(account, set())
        if not amount:
            continue
        if not candidates or "" in candidates or (region in candidates and len(candidates) > 1):
            raise NoUniqueAnswer("missing_information", "a nonzero account has an unresolved region")
        if region in candidates:
            answer += amount
    return answer


def latest_notice(root, directory, project, policy, field):
    policy_text = read_text(local_file(root, policy, {".txt"}), 16_000).strip()
    accepted = re.fullmatch(
        r"Only ([\w-]+) notices count\. Use the most recent issue date among ([\w-]+) notices\.", policy_text, re.I)
    if not accepted or accepted[1].casefold() != accepted[2].casefold():
        raise ValueError("unsupported notice policy")
    paths = (path for path in files(local_directory(root, directory)) if path.suffix.lower() == ".txt")
    return select_latest((read_notice(path) for path in paths), {"project": project, "status": accepted[1]}, "issued", field)


def read_notice(path):
    record = {}
    for line in read_text(path, 16_000).splitlines():
        if not line.strip():
            continue
        fact = FACT_LINE.fullmatch(line.strip())
        if not fact:
            raise InvalidInput("notice must contain one field per line")
        key, value = fact[1].casefold(), fact[2].strip()
        if key in record and record[key] != value:
            raise NoUniqueAnswer("logical_error", "conflicting fields in a notice")
        record[key] = value
    return record


def select_latest(records, filters, date_field, field, cutoff=None, exact_filters=False):
    cutoff = date.fromisoformat(cutoff) if cutoff else None
    newest = None
    values = set()
    for index, record in enumerate(records):
        if index >= 100:
            raise ValueError("notice count exceeds limit")
        if any(record.get(key) and (record[key] != value if exact_filters else record[key].casefold() != value.casefold())
               for key, value in filters.items()):
            continue
        if not all(record.get(key) for key in (*filters, date_field)):
            missing = [key for key in (*filters, date_field) if not record.get(key)]
            raise NoUniqueAnswer("missing_information", f"notice lacks fields {missing}; available fields: {sorted(record)}")
        issued = date.fromisoformat(record[date_field])
        if cutoff and issued > cutoff:
            continue
        if newest is None or issued > newest:
            newest, values = issued, set()
        if issued == newest:
            values.add(record.get(field.casefold()))
    if len(values) != 1 or not next(iter(values), None):
        raise NoUniqueAnswer("missing_information", "latest eligible notices do not determine a unique value")
    return values.pop()


def dated_fact(root, filters, date_field, field, cutoff, directory="", file=""):
    """Pick a fact using an explicit date column and inclusive cutoff in text notices."""
    if not 1 <= len(filters) <= 8 or not date_field.strip() or not field.strip():
        raise ValueError("supply filters, date field, answer field and cutoff")
    if bool(directory) == bool(file):
        raise ValueError("supply either a TXT notice directory or a CSV/TSV file")
    if file:
        def records():
            for row in table_rows(root, file, set()):
                if len({key.casefold() for key in row}) != len(row):
                    raise InvalidInput("ambiguous case-insensitive notice columns")
                yield {key.casefold(): value for key, value in row.items()}
    else:
        def records():
            paths = (path for path in files(local_directory(root, directory)) if path.suffix.lower() == ".txt")
            yield from (read_notice(path) for path in paths)
    return select_latest(records(),
                         {key.casefold(): value for key, value in filters.items()},
                         date_field.casefold(), field, cutoff, exact_filters=True)


def linked_fact(root, file, filters, path_column, field):
    """Follow all exact matching index rows and require one final fact value."""
    if not 1 <= len(filters) <= 8 or not path_column or not field:
        raise ValueError("supply exact filters, a path column and an answer field")
    paths = set()
    for row in table_rows(root, file, {*filters, path_column}):
        if any(row[key] != value for key, value in filters.items()):
            continue
        if not row[path_column]:
            raise NoUniqueAnswer("missing_information", "matching index row has no path")
        paths.add(row[path_column])
        if len(paths) > 40:
            raise ValueError("too many matching linked documents")
    values = set()
    for path in sorted(paths):
        record = read_notice(local_file(root, path, {".txt"}))
        value = record.get(field.casefold())
        if not value:
            raise NoUniqueAnswer("missing_information", "a linked document lacks the requested field")
        values.add(value)
    if len(values) != 1:
        raise NoUniqueAnswer("missing_information", "index links do not determine a unique final fact")
    return values.pop()


def latest_kb_notice(root, directory, project, status, scope, field):
    records = ({key.casefold(): value for key, value in entry["facts"].items()}
               for _, entry in markdown_records(local_directory(root, directory), strict=True, max_total=2_000_000))
    return select_latest(records, {"project": project, "status": status, "scope": scope}, "date", field)


def ledger_request(root, question):
    filtered = re.fullmatch(
        rf"(?:Use|Using) ({TABLE}) and ({TABLE})\. (?:Discard|Exclude) ([\w-]+) rows before selecting the greatest revision per order\. "
        r"Join account IDs as exact strings with leading zeroes\. For ([\w -]+), add sale cents and subtract refund cents\. "
        rf"An unregistered account can belong to ({NAME}) or ({NAME})\. Give net cents for \4[.?]", question, re.I)
    if filtered and filtered[4].casefold() in {filtered[5].casefold(), filtered[6].casefold()}:
        return net_ledger(root, filtered[2], filtered[1], filtered[4], exclude_states=(filtered[3],))
    patterns = (
        rf"Compute the net revenue in cents for the ({NAME}) region from ({TABLE}) and ({TABLE})\. "
        r"Join on account ID\. Keep only the highest revision of each order ID and subtract refunds\. "
        rf"Every journal account has one region, and the region of an account missing from the registry may be either {NAME} or {NAME}\.",
        rf"What is net revenue in cents for region ({NAME}) using registry ({TABLE}) and journal ({TABLE})\? "
        r"Link rows by account ID, keep the latest revision per order, and count refunds as negative\. "
        r"An unlisted account can belong to any region\.",
    )
    for pattern in patterns:
        if match := re.fullmatch(pattern, question, re.I):
            return net_ledger(root, match[2], match[3], match[1])
    return None


def search_request(root, question):
    precedence = re.fullmatch(
        rf"For project ({NAME}), what is the ({NAME}) in the (?:newest|latest) ([\w-]+) ([\w-]+) notice in ([\w/-]+)/\? "
        r"Date is the issue date\. A [\w-]+ notice or a [\w-]+ cannot replace a \3 \4 notice\.", question, re.I)
    if precedence:
        project, field, scope, status, directory = precedence.groups()
        return latest_kb_notice(root, directory, project, status, scope, field)
    badge = re.fullmatch(
        rf"The badge roster in ({TABLE}) maps badge IDs to names\. Find the person with badge ([\w-]+), then find "
        rf"their employer in ([\w/-]+)/\. Using ({TABLE}), return that employer's active permit code for the ({NAME}) district\.",
        question, re.I)
    if badge:
        roster, identifier, directory, licenses, district = badge.groups()
        name = db_query(root, roster, "unique", "name", [["id", "=", identifier]])
        employer = kb_lookup(local_directory(root, directory), name, ["Employer"], strict=True)
        return db_query(root, licenses, "unique", "code", [
            ["employer", "=", employer], ["district", "=", district], ["status", "=", "active"]])
    notice = re.fullmatch(
        rf"Find the ({NAME}) for project ({NAME})\. Apply the document selection policy in ({TEXT}) to the "
        rf"(?:\w+ )?notices under ([\w/-]+)/\. Return the code for the selected notice\.", question, re.I)
    if notice:
        field, project, policy, directory = notice.groups()
        if not field.casefold().endswith("code"):
            return None
        return latest_notice(root, directory, project, policy, field)
    return None
