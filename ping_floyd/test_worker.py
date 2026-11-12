"""Run source-only unittest tasks in a Linux subprocess with a syscall allowlist."""
import ctypes as C
import errno
import importlib
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import linecache
import os
from pathlib import Path
import resource
import sys
from types import ModuleType
import unittest


class WorkspaceModules(MetaPathFinder, Loader):
    def __init__(self, paths):
        self.modules = {}
        self.scripts = {}
        for path in paths:
            parts = list(path.with_suffix("").parts)
            package = parts[-1] == "__init__"
            if package:
                parts.pop()
            source = path.read_bytes()
            filename = path.as_posix()
            try:
                code = compile(source, filename, "exec")
            except SyntaxError:
                # A source syntax error should surface when the test imports it.
                code = source
            linecache.cache[filename] = (
                len(source), None, source.decode("utf-8", errors="replace").splitlines(keepends=True), filename
            )
            self.scripts[filename] = code
            if parts and all(part.isidentifier() for part in parts):
                self.modules[".".join(parts)] = (filename, code, package)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            filename, _, package = self.modules[fullname]
            return spec_from_loader(fullname, self, origin=filename, is_package=package)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__, code, _ = self.modules[module.__name__]
        exec(code if not isinstance(code, bytes) else compile(code, module.__file__, "exec"), module.__dict__)


def load_workspace():
    paths = []
    total = 0
    for folder, dirs, names in os.walk("."):
        dirs[:] = sorted(name for name in dirs
                         if not name.startswith(".") and not (Path(folder) / name).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            if not name.startswith(".") and path.suffix == ".py" and not path.is_symlink() and path.is_file():
                total += path.stat().st_size
                if total > 8_000_000 or len(paths) >= 1000:
                    raise ValueError("source limit exceeded")
                paths.append(path)
    return WorkspaceModules(paths)


def isolate():
    if sys.platform != "linux":
        raise RuntimeError("Python tests require the Linux Docker image")
    libc = C.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):  # NO_NEW_PRIVS
        raise OSError(C.get_errno(), "prctl failed")
    seccomp = C.CDLL("libseccomp.so.2")
    seccomp.seccomp_init.argtypes = [C.c_uint32]
    seccomp.seccomp_init.restype = C.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint]
    seccomp.seccomp_load.argtypes = [C.c_void_p]
    seccomp.seccomp_release.argtypes = [C.c_void_p]
    context = seccomp.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("seccomp initialization failed")
    try:
        allowed = (
            "read write readv writev close fstat lseek "
            "mmap mprotect munmap mremap madvise brk "
            "rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack "
            "getpid gettid getuid geteuid getgid getegid getrandom "
            "clock_gettime clock_nanosleep nanosleep sched_yield futex "
            "getrusage times exit exit_group getcwd"
        )
        for name in allowed.split():
            syscall = seccomp.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and seccomp.seccomp_rule_add(context, 0x7FFF0000, syscall, 0):
                raise RuntimeError(f"seccomp rule failed: {name}")
        if seccomp.seccomp_load(context):
            raise RuntimeError("seccomp load failed")
    finally:
        seccomp.seccomp_release(context)


def main():
    script = sys.argv[1]
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (384 << 20, 384 << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 10, 256 << 10))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        modules = load_workspace()
        # Preload common stdlib dependencies before denying filesystem access.
        for name in ("bisect", "calendar", "collections", "copy", "csv", "datetime", "decimal",
                     "fractions", "functools", "heapq", "itertools", "json", "math", "operator",
                     "random", "re", "statistics", "string", "typing", "unittest.mock"):
            importlib.import_module(name)
        isolate()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"test isolation unavailable: {error}", file=sys.stderr)
        return 125

    sys.meta_path.insert(0, modules)
    if script:
        script = Path(script).as_posix()
        code = modules.scripts[script]
        entry = ModuleType("__main__")
        entry.__file__ = script
        sys.argv = [script]
        sys.modules["__main__"] = entry
        exec(code if not isinstance(code, bytes) else compile(code, script, "exec"), entry.__dict__)
        return 0
    suite = unittest.TestSuite()
    for name in modules.modules:
        if name.split(".")[-1].startswith("test"):
            module = importlib.import_module(name)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner().run(suite)
    return 0 if result.testsRun and result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
