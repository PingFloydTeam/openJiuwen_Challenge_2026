"""Experiment-only test worker; launched inside an external macOS sandbox."""
import importlib
import importlib.util
import resource
import sys
from pathlib import Path
from types import ModuleType
import unittest


def main():
    worker, script = sys.argv[1:]
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 10, 256 << 10))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    spec = importlib.util.spec_from_file_location('isolated_worker', worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules = module.load_workspace()
    sys.meta_path.insert(0, modules)
    if script:
        code = modules.scripts[script]
        entry = ModuleType('__main__')
        entry.__file__ = script
        sys.argv = [script]
        sys.modules['__main__'] = entry
        exec(code if not isinstance(code, bytes) else compile(code, script, 'exec'), entry.__dict__)
        return 0
    suite = unittest.TestSuite()
    for name in modules.modules:
        if name.split('.')[-1].startswith('test'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
    result = unittest.TextTestRunner().run(suite)
    return 0 if result.testsRun and result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
