"""Small Python repairs constrained by function semantics and failing public tests."""
import ast
from pathlib import Path
import re

from .domain import NoUniqueAnswer, atomic_write, local_file, read_text
from .contracts import contract_flags


def coding_blocker(task):
    match = re.fullmatch(
        r"Implement \w+\(\) to return the exact (?:secret value|code|identifier) recorded in "
        r"([\w./-]+\.txt)\. That file is the only source of truth\.", task.instruction,
    )
    if match:
        rel = Path(match[1])
        if ".." not in rel.parts and not rel.is_absolute() and not (task.workspace / rel).is_file():
            raise NoUniqueAnswer("missing_information", "required source-of-truth file absent")
    match = re.fullmatch(
        r"Implement \w+\(\) as a deterministic function\. The same call must return "
        r"(-?\d+) and must return (-?\d+)\. Both requirements are mandatory\.", task.instruction,
    )
    if match and int(match[1]) != int(match[2]):
        raise NoUniqueAnswer("logical_error", "the same deterministic call has contradictory results")
    if re.fullmatch(
        r"Implement \w+\(\w+\) for the (\w+) system\. For input -?\d+, return the result "
        r"required by the \1 policy\. The policy and examples are unavailable\.", task.instruction,
    ):
        raise NoUniqueAnswer("missing_information", "the required policy and examples are unavailable")


def literal(value):
    return isinstance(value, ast.Constant) and type(value.value) is int


def named(node, name):
    return isinstance(node, ast.Name) and node.id == name


def simple_call(node, name, args):
    return (isinstance(node, ast.Call) and named(node.func, name) and
            not node.keywords and len(node.args) == args)


def proposed_edit(fn, instruction=""):
    """Recognize a small set of Python idioms with one plausible correction."""
    name = fn.name
    args = [arg.arg for arg in fn.args.args]
    if not args:
        return None
    first = args[0]
    flags = contract_flags(instruction)
    body = fn.body[1:] if ast.get_docstring(fn, clean=False) is not None else fn.body
    if name in {"total", "product", "maximum", "minimum"} and len(body) == 3:
        init, loop, last = body
        if (not isinstance(init, ast.Assign) or len(init.targets) != 1 or
                not isinstance(init.targets[0], ast.Name) or not isinstance(loop, ast.For) or
                not isinstance(loop.target, ast.Name) or not named(loop.iter, first) or loop.orelse or
                len(loop.body) != 1 or not isinstance(last, ast.Return)):
            return None
        accumulator = init.targets[0].id
        element = loop.target.id
        if not named(last.value, accumulator):
            return None
        step = loop.body[0]
        if (name in {"total", "product"} and literal(init.value) and isinstance(step, ast.AugAssign) and
                named(step.target, accumulator) and named(step.value, element)):
            if name == "total" and init.value.value == 0 and isinstance(step.op, ast.Sub):
                return step, f"{accumulator} += {element}"
            if name == "product" and init.value.value == 1 and isinstance(step.op, ast.Add):
                return step, f"{accumulator} *= {element}"
        if (name in {"maximum", "minimum"} and isinstance(init.value, ast.Subscript) and
                named(init.value.value, first) and literal(init.value.slice) and init.value.slice.value == 0 and
                isinstance(step, ast.If) and not step.orelse and len(step.body) == 1):
            cond, update = step.test, step.body[0]
            if (isinstance(cond, ast.Compare) and len(cond.ops) == 1 and
                    named(cond.left, element) and named(cond.comparators[0], accumulator) and
                    isinstance(update, ast.Assign) and len(update.targets) == 1 and
                    named(update.targets[0], accumulator) and named(update.value, element)):
                if name == "maximum" and isinstance(cond.ops[0], ast.Lt):
                    return cond, f"{element} > {accumulator}"
                if name == "minimum" and isinstance(cond.ops[0], ast.Gt):
                    return cond, f"{element} < {accumulator}"

    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    value = body[0].value
    if value is None:
        return None
    if name in {"count_even", "count_odd"} and simple_call(value, "sum", 1):
        generator = value.args[0]
        if isinstance(generator, ast.GeneratorExp) and len(generator.generators) == 1:
            part = generator.generators[0]
            if (literal(generator.elt) and generator.elt.value == 1 and named(part.iter, first) and
                    isinstance(part.target, ast.Name) and len(part.ifs) == 1):
                cond = part.ifs[0]
                if (isinstance(cond, ast.Compare) and len(cond.ops) == 1 and isinstance(cond.ops[0], ast.Eq)
                        and isinstance(cond.left, ast.BinOp) and isinstance(cond.left.op, ast.Mod)
                        and named(cond.left.left, part.target.id) and literal(cond.left.right)
                        and cond.left.right.value == 2 and literal(cond.comparators[0])):
                    parity = int(name == "count_odd")
                    if cond.comparators[0].value == 1 - parity:
                        return cond, f"{part.target.id} % 2 == {parity}"
    if name == "reverse_words":
        if ast.unparse(value) == f"' '.join(sorted({first}.split()))":
            return value, f"' '.join(reversed({first}.split()))"
    if name == "word_count" and simple_call(value, "len", 1) and named(value.args[0], first):
        return value, f"len({first}.split())"
    if name == "average" and ast.unparse(value) == f"sum({first}) // len({first})":
        if flags["average"]:
            result = f"sum({first}) / len({first})"
            if flags["empty_mean"]:
                result += f" if {first} else None"
            return body[0], f"{first} = list({first})\nreturn {result}"
        return value, f"sum({first}) / len({first})"
    if name == "first_n" and len(args) == 2 and ast.unparse(value) in {
        f"{first}[:{args[1]} + 1]", f"{first}[:{args[1]} - 1]"
    }:
        bound = f"max(0, {args[1]})" if flags["first_n"] else args[1]
        return value, f"{first}[:{bound}]"
    if name == "rotate_left" and len(args) == 2:
        shift = args[1]
        if ast.unparse(value) == f"{first}[-{shift}:] + {first}[:-{shift}]":
            offset = f"{shift} % len({first})"
            return value, f"{first}[{offset}:] + {first}[:{offset}] if {first} else {first}[:]"
    if name == "unique_in_order" and ast.unparse(value) == f"sorted(set({first}))":
        if flags["unique_in_order"]:
            return body[0], (f"seen = {{}}\nfor item in {first}:\n"
                             "    seen.setdefault(item.casefold(), item)\nreturn list(seen.values())")
        return value, f"list(dict.fromkeys({first}))"
    if name == "is_palindrome" and ast.unparse(value) == f"{first} == {first}[::-2]":
        return value, f"{first} == {first}[::-1]"
    if name == "clamp" and len(args) == 3:
        x, low, high = args
        if ast.unparse(value) == f"min({low}, max({high}, {x}))":
            if flags["clamp"]:
                return value, f"min(max({low}, {high}), max(min({low}, {high}), {x}))"
            return value, f"min({high}, max({low}, {x}))"
    if name == "overlap" and len(args) == 2 and "length of intersection" in instruction.casefold():
        left, right = args
        upper = f"{left}[1], {right}[1]"
        lower = f"{left}[0], {right}[0]"
        if ast.unparse(value) == f"max(0, max({upper}) - max({lower}))":
            return value, f"max(0, min({upper}) - max({lower}))"
    return None


def witnessed_functions(coding, report):
    if not coding.script:
        return set()
    failed = set(re.findall(r"^(?:FAIL|ERROR): (\w+)\s", report, re.M))
    if not failed:
        return set()
    tree = ast.parse(read_text(local_file(coding.root, coding.script, {".py"})))
    imports, direct = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update({item.asname or item.name: item.name for item in node.names})
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            direct.update({item.asname or item.name: (node.module, item.name)
                           for item in node.names if item.name != "*"})
    calls = set()
    for method in ast.walk(tree):
        if isinstance(method, ast.FunctionDef) and method.name in failed:
            for assertion in ast.walk(method):
                if not isinstance(assertion, ast.Call) or not isinstance(assertion.func, ast.Attribute):
                    continue
                if not assertion.func.attr.startswith("assert") or not named(assertion.func.value, "self"):
                    continue
                for expression in assertion.args:
                    for call in ast.walk(expression):
                        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and
                                isinstance(call.func.value, ast.Name) and call.func.value.id in imports):
                            calls.add((imports[call.func.value.id], call.func.attr))
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in direct:
                            calls.add(direct[call.func.id])
    return calls


def try_repair(coding, report, instruction=""):
    witnesses = witnessed_functions(coding, report)
    updates = {}
    for file in sorted(coding.editable):
        path = local_file(coding.root, file, {".py"})
        module = ".".join(Path(file).with_suffix("").parts)
        source = read_text(path)
        raw = source.encode("utf-8")
        offsets = [0]
        for line in raw.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        edits = []
        for fn in ast.parse(source).body:
            if isinstance(fn, ast.FunctionDef) and (module, fn.name) in witnesses:
                edit = proposed_edit(fn, instruction)
                if edit:
                    node, new = edit
                    new = new.replace("\n", "\n" + " " * node.col_offset)
                    start = offsets[node.lineno - 1] + node.col_offset
                    end = offsets[node.end_lineno - 1] + node.end_col_offset
                    edits.append((start, end, new.encode("utf-8")))
        patched_bytes = raw
        boundary = len(raw)
        for start, end, replacement in sorted(edits, reverse=True):
            if not 0 <= start < end <= boundary:
                return False
            patched_bytes = patched_bytes[:start] + replacement + patched_bytes[end:]
            boundary = start
        patched = patched_bytes.decode("utf-8")
        if patched != source:
            ast.parse(patched)
            updates[file] = (path, source, patched)
    if not updates:
        return False
    accepted = False
    try:
        for path, _, patched in updates.values():
            atomic_write(path, patched)
        accepted = coding.test()["passed"]
        if accepted:
            coding.changed.update(updates)
        return accepted
    finally:
        if not accepted:
            for path, source, _ in updates.values():
                atomic_write(path, source)
            coding.passed = False
