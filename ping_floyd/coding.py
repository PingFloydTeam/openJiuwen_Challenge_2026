"""Stage patches and validate them before touching the contest sources."""
import ast
import inspect
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import hashlib

from .domain import atomic_write, files, local_file, read_text
from .contracts import contract_tests

WORKER = Path(__file__).with_name("test_worker.py").resolve()


def is_test(path):
    return (any(p in {"test", "tests"} for p in path.parts[:-1]) or
            path.name.startswith("test") or path.name.endswith("_test.py"))


def test_run(root: Path, script=""):
    if script:
        local_file(root, script, {".py"})
    protected={p:hashlib.sha256(p.read_bytes()).digest() for p in root.rglob('*.py') if p.is_file()}
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(WORKER), script],
                cwd=root, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                env={"LANG": "C.UTF-8"}, timeout=8, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "isolation_error": False, "output": "test timeout"}
        size = output.tell()
        output.seek(max(0, size - 2500))
        tail = output.read().decode("utf-8", errors="replace")
    ran = any(int(count) > 0 for count in re.findall(r"Ran (\d+) tests?", tail))
    if any(not p.is_file() or hashlib.sha256(p.read_bytes()).digest()!=digest for p,digest in protected.items()):
        return {'passed':False,'isolation_error':False,'output':'test execution modified protected Python files'}
    return {"passed": result.returncode == 0 and ran,
            "isolation_error": result.returncode == 125, "output": tail.strip()}


def apply_sources(updates):
    written = []
    try:
        for path, original, source in updates:
            atomic_write(path, source)
            written.append((path, original))
    except OSError:
        for path, original in reversed(written):
            atomic_write(path, original)
        raise


class Coding:
    def __init__(self, workspace: Path, stage: Path, instruction: str):
        self.root = stage
        self.workspace = workspace
        self.instruction = instruction
        self.changed: set[str] = set()
        self.passed = False
        self.editable: set[str] = set()
        self.probes = []
        total = 0
        for path in files(workspace):
            total += path.stat().st_size
            if total > 32_000_000:
                raise ValueError("coding workspace exceeds staging limit")
            relative = path.relative_to(workspace)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if relative.suffix == ".py" and not is_test(relative):
                self.editable.add(relative.as_posix())
        match = re.search(r"`python3? ([-\w./]+\.py)`", instruction)
        candidate = match[1] if match else ""
        self.script = candidate if candidate and is_test(Path(candidate)) else ""

    def inspect_code(self, file, function=""):
        source = read_text(local_file(self.root, file, {".py"}), 200000)
        symbols = []
        def walk(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = prefix + node.name
                    symbols.append((name, node))
                    if len(symbols) > 100:
                        raise ValueError("source has too many symbols; use read for a smaller excerpt")
                    walk(node.body, name + ".")
        walk(ast.parse(source).body)
        if function:
            matches = [node for name, node in symbols if name == function]
            if len(matches) != 1:
                raise ValueError("function or class must have a unique qualified name")
            node = matches[0]
            text = "".join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
            if len(text) > 6000:
                raise ValueError("symbol exceeds excerpt size; use read with line numbers")
            return {"name": function, "start": node.lineno, "source": text}
        return [{"name": name, "start": node.lineno, "end": node.end_lineno,
                 "args": ast.unparse(node.args) if hasattr(node, "args") else "class"}
                for name, node in symbols[:30]]

    def patch(self, file: str, edits: list, checks=None):
        return self.patch_files({file: edits}, checks)

    def patch_files(self, patches: dict, checks=None):
        probes = self.validate_checks(checks)
        if not 1 <= len(patches) <= 4:
            raise ValueError("supply patches for one to four existing source files")
        prepared = {}
        for file, edits in patches.items():
            if file not in self.editable or not 1 <= len(edits) <= 8:
                raise ValueError("patch an existing source .py file only")
            path = local_file(self.root, file, {".py"})
            original = source = read_text(path)
            for old, new in edits:
                if not isinstance(old, str) or not isinstance(new, str) or not old or source.count(old) != 1:
                    raise ValueError("each old fragment must occur exactly once")
                source = source.replace(old, new, 1)
            compile(source, file, "exec", dont_inherit=True)
            prepared[file] = path, original, source
        apply_sources(prepared.values())
        self.probes.extend(probes)
        self.changed.update(prepared)
        self.passed = False
        return self.test()

    def replace_body(self, file: str, function: str, body: str, checks=None):
        probes = self.validate_checks(checks)
        if file not in self.editable or not function.isidentifier() or not 1 <= len(body) <= 4096:
            raise ValueError("supply an existing source file, function and nonempty body")
        path = local_file(self.root, file, {".py"})
        source = read_text(path)
        candidates = [item for item in ast.parse(source).body
                      if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function]
        if len(candidates) != 1:
            raise ValueError("the requested top-level function is absent or repeated")
        fn = candidates[0]
        statements = fn.body[1:] if ast.get_docstring(fn, clean=False) is not None else fn.body
        if not statements or statements[0].lineno == fn.lineno:
            raise ValueError("body replacement needs a multiline function with a statement")
        lines = [line.decode("utf-8") for line in source.encode("utf-8").splitlines(keepends=True)]
        start = statements[0].lineno - 1
        prefix = re.match(r"[ \t]+", lines[start])
        if not prefix:
            raise ValueError("could not locate function indentation")
        if start and ast.get_docstring(fn, clean=False) is not None and fn.body[0].end_lineno > start:
            raise ValueError("body must start on its own line after the docstring")
        normalized = textwrap.dedent(body).strip("\r\n")
        try:
            compile('def _probe():\n' + textwrap.indent(normalized, '    '), file, 'exec')
        except IndentationError:
            # Handle a stripped first line followed by a uniformly indented
            # function body. Accept only if Python itself validates the result.
            normalized = inspect.cleandoc(body)
            compile('def _probe():\n' + textwrap.indent(normalized, '    '), file, 'exec')
        indented = textwrap.indent(normalized, prefix[0])
        if not indented.strip():
            raise ValueError("supply a nonempty function body")
        patched = "".join(lines[:start]) + indented + "\n" + "".join(lines[fn.end_lineno:])
        compile(patched, file, "exec", dont_inherit=True)
        self.probes.extend(probes)
        return self._update(file, path, patched)

    def validate_checks(self, checks):
        """Model-supplied examples are supplementary, never replacements for tests."""
        if checks is None:
            return []
        if not isinstance(checks, list) or len(self.probes) + len(checks) > 24 or len(repr(checks)) > 8000:
            raise ValueError("supply at most 24 small supplementary checks")
        bodies = []
        for check in checks:
            if not isinstance(check, dict) or set(check) - {'file', 'function', 'args', 'kwargs', 'expected', 'raises'}:
                raise ValueError("check needs file,function,args,expected OR raises")
            file, fn = check.get('file'), check.get('function', '')
            if file not in self.editable or not fn.isidentifier():
                raise ValueError("check must call a source function")
            module = '.'.join(Path(file).with_suffix('').parts)
            if not all(p.isidentifier() for p in module.split('.')):
                raise ValueError("check source must be importable")
            args, kwargs = check.get('args', []), check.get('kwargs', {})
            if not isinstance(args, list) or not isinstance(kwargs, dict) or any(not k.isidentifier() for k in kwargs):
                raise ValueError("check args/kwargs have invalid types")
            if ('expected' in check) == ('raises' in check):
                raise ValueError("check needs exactly one expected result or exception")
            call = f'fn(*{args!r}, **{kwargs!r})'
            body = [f'from {module} import {fn} as fn']
            if 'raises' in check:
                exc = check['raises']
                if exc not in {'ValueError', 'TypeError', 'KeyError', 'IndexError', 'ZeroDivisionError'}:
                    raise ValueError('unsupported expected exception')
                body.extend([f'with self.assertRaises({exc}):', f'    {call}'])
            else:
                body.append(f'self.assertEqual({call}, {check["expected"]!r})')
            # Only literal values plus fixed test scaffolding may be emitted.
            ast.literal_eval(repr((args, kwargs, check.get('expected'))))
            bodies.append(body)
        return bodies

    def _update(self, file, path, source):
        compile(source, file, "exec", dont_inherit=True)
        atomic_write(path, source)
        self.changed.add(file)
        self.passed = False
        return self.test()

    def test(self):
        result = test_run(self.root, self.script)
        if result["passed"]:
            source = contract_tests(self.root, self.editable, self.instruction)
            if source:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                                  prefix="pf_contract_", suffix=".py", delete=False) as stream:
                    path = Path(stream.name)
                    stream.write(source)
                try:
                    result = test_run(self.root, path.name)
                finally:
                    path.unlink()
        if result['passed'] and self.probes:
            source = 'import unittest\nclass Supplementary(unittest.TestCase):\n'
            for i, body in enumerate(self.probes):
                source += f'    def test_probe_{i}(self):\n' + ''.join('        '+line+'\n' for line in body)
            source += "if __name__ == '__main__': unittest.main()\n"
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                                              prefix='pf_probes_', suffix='.py', delete=False) as stream:
                path = Path(stream.name)
                stream.write(source)
            try:
                result = test_run(self.root, path.name)
            finally:
                path.unlink()
        self.passed = result["passed"]
        return result

    def commit(self):
        if not self.passed:
            raise ValueError("coding solution requires passing tests")
        updates = []
        for name in sorted(self.changed):
            source = local_file(self.root, name, {".py"})
            target = local_file(self.workspace, name, {".py"})
            updates.append((target, read_text(target), read_text(source)))
        apply_sources(updates)
