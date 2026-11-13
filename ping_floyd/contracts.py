"""Additional checks derived only from explicit function requirements."""
import ast
from pathlib import Path
import re

from .domain import NoUniqueAnswer, files, local_file, read_text
from .tools import calc
from .properties import quantified_checks


def stateful_contract(instruction):
    return bool(re.search(r'\bstateful\b|\bsuccessive calls\b|\bconsecutive calls\b|\b(?:persistent|mutable|global) (?:state|counter)\b',instruction,re.I))


def literal_assertions(instruction):
    """Parse explicit backticked assertions; never execute task text."""
    assertions = []
    for text in re.findall(r"`(assert [^`\n]{1,1500})`", instruction)[:24]:
        try:
            node = ast.parse(text).body[0]
            cmp = node.test
            if (not isinstance(node, ast.Assert) or not isinstance(cmp, ast.Compare)
                    or len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq)
                    or not isinstance(cmp.left, ast.Call) or not isinstance(cmp.left.func, ast.Name)):
                continue
            call = cmp.left
            args = [ast.literal_eval(a) for a in call.args]
            kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            if None in kwargs:
                continue
            expected = ast.literal_eval(cmp.comparators[0])
            assertions.append((call.func.id, args, kwargs, expected))
        except (ValueError, SyntaxError, AttributeError):
            continue
    return assertions


def universal_contract_conflicts(instruction):
    """Find concrete contradictory obligations in explicit integer equations."""
    if stateful_contract(instruction):
        return  # Equal arguments in different states need not have equal results.
    examples = {}
    for name, args, kwargs, expected in literal_assertions(instruction):
        key = name, repr(args), repr(sorted(kwargs.items()))
        if key in examples and examples[key] != expected:
            raise NoUniqueAnswer("logical_error", "explicit assertions require different results for the same call")
        examples[key] = expected
    types = re.search(r"must return a built-in (\w+) and must return a built-in (\w+) for the same call", instruction, re.I)
    if types and "exact type" in instruction.casefold():
        supported = {"list", "tuple", "dict", "set", "str", "int", "float", "bool", "bytes"}
        first, second = (name.casefold() for name in types.groups())
        if first in supported and second in supported and first != second:
            raise NoUniqueAnswer("logical_error", "one result cannot have two different exact built-in types")
    pattern = (r"For (?:every|each) integer ([A-Za-z]\w{0,15}),\s*([A-Za-z_]\w{0,39})"
               r"\(([^;,.\n]{1,80})\) must equal ([^;.\n]{1,80}?)(?: exactly)?\.")
    obligations = {}
    for match in list(re.finditer(pattern, instruction, re.I))[:8]:
        variable, function, argument, expected = match.groups()
        try:
            expressions = [ast.parse(part, mode="eval") for part in (argument, expected)]
        except SyntaxError:
            continue
        if any(isinstance(node, ast.Name) and node.id != variable
               for tree in expressions for node in ast.walk(tree)):
            continue
        values = []
        for value in (-3, -2, -1, 0, 1, 2, 3):
            try:
                args, result = [calc(re.sub(r"\b" + re.escape(variable) + r"\b", f"({value})", part))
                                for part in (argument, expected)]
            except ZeroDivisionError:
                raise NoUniqueAnswer("undefined_math", f"the universal contract is undefined at {variable}={value}") from None
            except (ValueError, SyntaxError):
                values = []
                break
            values.append((str(args), str(result), value))
        for args, result, value in values:
            key = function, args
            if key in obligations and obligations[key][0] != result:
                prior, witness = obligations[key]
                raise NoUniqueAnswer("logical_error", f"{function}({args}) must equal both {prior} and {result}; witnesses {witness} and {value}")
            obligations[key] = result, value


def required_inputs(workspace, instruction):
    """Reject absent data explicitly named as the source for the implementation."""
    pattern = (r"\b(?:from|according to|under|using|recorded in|defined in|specified in)\s+"
               r"(?:the\s+)?(?:(?:table|policy|rules|data|file|specification|mapping)\s+(?:in|at)\s+)?"
               r"[`\"']?([\w./-]+\.(?:csv|tsv|json|txt))\b")
    for match in re.finditer(pattern, instruction, re.I):
        filename=match[1]
        if re.search(r'\b(?:caller|test|tests|calling code)\b[^.]{0,120}\b(?:creates?|writes?|provides?)\b[^.]{0,80}(?:'+re.escape(filename)+r'|this file|the file)',instruction,re.I):
            continue
        try:
            local_file(workspace, match[1])
        except ValueError:
            raise NoUniqueAnswer("missing_information", f"required input {match[1]} is unavailable") from None


def contract_flags(instruction):
    text = instruction.casefold()
    return {
        "clamp": "clamp(" in text and bool(re.search(r"unordered endpoints|either order|reversed (?:bounds|endpoints)", text)),
        "first_n": "first_n(" in text and bool(re.search(r"zero or negative|nonpositive|non-positive", text)),
        "average": "average(" in text and bool(re.search(r"one-shot|iterator|iterable", text)),
        "empty_mean": "average(" in text and bool(re.search(r"none (?:when|if) (?:it is |the iterable is )?empty", text)),
        "unique_in_order": "unique_in_order(" in text and "casefold" in text,
    }


def input_invariants(root, editable, instruction):
    """Reuse literal public examples for explicit nonmutation/iterable contracts."""
    preserve = bool(re.search(r"(?:do|must|should) not (?:mutate|modify) (?:the )?inputs?", instruction, re.I))
    iterable = bool(re.search(r"finite iterable|one-shot iterator", instruction, re.I))
    if not (preserve or iterable):
        return []
    requested = set(re.findall(r"\b([A-Za-z_]\w*)\(", instruction))
    modules = {".".join(Path(file).with_suffix("").parts) for file in editable}
    bodies = []
    for path in files(root):
        if path.suffix != ".py" or not path.name.startswith("test"):
            continue
        tree = ast.parse(read_text(path, 200000))
        direct, imports = {}, {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in modules:
                direct.update({alias.asname or alias.name: (node.module, alias.name) for alias in node.names})
            elif isinstance(node, ast.Import):
                imports.update({alias.asname or alias.name: alias.name for alias in node.names if alias.name in modules})
        for assertion in ast.walk(tree):
            if (not isinstance(assertion, ast.Call) or not isinstance(assertion.func, ast.Attribute)
                    or assertion.func.attr not in {"assertEqual", "assertListEqual", "assertTupleEqual"}
                    or not assertion.args or not isinstance(assertion.args[0], ast.Call)):
                continue
            call = assertion.args[0]
            target = direct.get(call.func.id) if isinstance(call.func, ast.Name) else None
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                module = imports.get(call.func.value.id)
                target = (module, call.func.attr) if module else None
            if not target or target[1] not in requested:
                continue
            try:
                args = []
                for arg in call.args:
                    args.extend(ast.literal_eval(arg.value)) if isinstance(arg, ast.Starred) else args.append(ast.literal_eval(arg))
                kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in call.keywords}
            except (ValueError, TypeError, SyntaxError):
                continue
            if None in kwargs or len(repr((args, kwargs))) > 1500:
                continue
            body = [f"from {target[0]} import {target[1]} as fn", "from copy import deepcopy",
                    f"args, kwargs = {args!r}, {kwargs!r}", "before = deepcopy((args, kwargs))",
                    "result = fn(*args, **kwargs)"]
            if preserve:
                body.append("self.assertEqual((args, kwargs), before)")
                if args and isinstance(args[0], list) and "any order" in instruction.casefold():
                    body.extend(["args, kwargs = deepcopy(before)", "args[0].reverse()", "before_reversed = deepcopy((args, kwargs))",
                                 "fn(*args, **kwargs)", "self.assertEqual((args, kwargs), before_reversed)"])
            if iterable and args and isinstance(args[0], (list, tuple)):
                body.extend(["args, kwargs = deepcopy(before)", "args[0] = iter(args[0])",
                             "self.assertEqual(fn(*args, **kwargs), result)"])
            bodies.append(body)
            if len(bodies) >= 24:
                return bodies
    return bodies


def contract_tests(root, editable, instruction):
    flags = contract_flags(instruction)
    checks = {
        "clamp": [
            "for low, high in [(5, -2), (-2, 5), (3, 3)]:",
            "    for x in range(-5, 9):",
            "        expected = sorted([min(low, high), x, max(low, high)])[1]",
            "        self.assertEqual(fn(x, low, high), expected)",
        ],
        "first_n": [
            "values = [9, -2, 7]",
            "for count in range(-4, 6):",
            "    expected = [value for index, value in enumerate(values) if index < count]",
            "    self.assertEqual(fn(values, count), expected)",
            "self.assertEqual(values, [9, -2, 7])",
        ],
        "average": [
            "self.assertEqual(fn(iter([2, 3, 8])), 13 / 3)",
            "self.assertEqual(fn(iter([0])), 0)",
        ],
        "unique_in_order": [
            "self.assertEqual(fn(['Z', 'z', 'Maße', 'MASSE', 'A']), ['Z', 'Maße', 'A'])",
            "self.assertEqual(fn([]), [])",
        ],
    }
    if flags["empty_mean"]:
        checks["average"].append("self.assertIsNone(fn(iter([])))")
    tests = []
    symbols = {}
    for file in sorted(editable):
        module = '.'.join(Path(file).with_suffix('').parts)
        if not all(part.isidentifier() for part in module.split('.')):
            continue
        for node in ast.parse(read_text(root / file)).body:
            if isinstance(node, ast.FunctionDef):
                symbols.setdefault(node.name, []).append(module)
    for name, args, kwargs, expected in ([] if stateful_contract(instruction) else literal_assertions(instruction)):
        if len(symbols.get(name, [])) == 1:
            body = [f'from {symbols[name][0]} import {name} as fn',
                    f'self.assertEqual(fn(*{args!r}, **{kwargs!r}), {expected!r})']
            tests.append(f"    def test_contract_{len(tests)}(self):\n" + ''.join('        '+line+'\n' for line in body))
    for body in ([] if stateful_contract(instruction) else quantified_checks(instruction, symbols)):
        tests.append(f"    def test_property_{len(tests)}(self):\n" + ''.join('        '+line+'\n' for line in body))
    for body in input_invariants(root, editable, instruction):
        tests.append(f"    def test_contract_{len(tests)}(self):\n" + "".join("        " + line + "\n" for line in body))
    for file in sorted(editable):
        module = ".".join(Path(file).with_suffix("").parts)
        if not all(part.isidentifier() for part in module.split(".")):
            continue
        source = ast.parse((root / file).read_text())
        for fn in source.body:
            if isinstance(fn, ast.FunctionDef) and flags.get(fn.name):
                body = [f"from {module} import {fn.name} as fn", *checks[fn.name]]
                tests.append(f"    def test_contract_{len(tests)}(self):\n"
                             + "".join("        " + line + "\n" for line in body))
    if not tests:
        return None
    return ("import unittest\n\nclass RequiredBehavior(unittest.TestCase):\n" + "".join(tests)
            + "\nif __name__ == '__main__':\n    unittest.main()\n")
