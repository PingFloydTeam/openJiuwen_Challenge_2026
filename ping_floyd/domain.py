"""Contest input and output boundaries."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import os
import stat

TYPES = {"math", "search", "data_analysis", "coding"}


class NoUniqueAnswer(ValueError):
    def __init__(self, category: str, reason: str):
        super().__init__(reason)
        self.category = category


class InvalidInput(ValueError):
    """A named input does not meet the format required for reliable computation."""


@dataclass(frozen=True)
class Task:
    root: Path
    kind: str
    instruction: str
    reward: int
    check: str
    task_id: str = ""

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @classmethod
    def load(cls, manifest: Path):
        data = json.loads(read_text(manifest, 128_000))
        if data["task_type"] not in TYPES:
            raise ValueError("unknown task type")
        if not isinstance(data["instruction"], str) or not isinstance(data.get("check", ""), str):
            raise ValueError("instruction and check must be strings")
        task_id = data.get("task_id")
        return cls(manifest.parent, data["task_type"], data["instruction"],
                   int(data["reward"]), data.get("check", ""),
                   task_id if isinstance(task_id, str) and task_id else manifest.parent.name)


def atomic_write(path: Path, value: str):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(value)
            f.flush()
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            os.fchmod(f.fileno(), mode)
            os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


class AnswerLog:
    """Plain text snapshot of published answers, rebuilt from task files on resume."""

    def __init__(self, path, tasks, trace):
        self.path = (path / "answers.txt" if path.is_dir() else path).resolve()
        if any(self.path.is_relative_to(task.root) for task in tasks):
            raise ValueError("OUT must be outside individual task directories")
        self.trace = trace
        self.answers = {}
        self.dirty = True
        for task in tasks:
            answer_path = task.root / "answer.txt"
            if answer_path.is_file() and not answer_path.is_symlink():
                answer = read_text(answer_path).removesuffix("\n")
                self.answers[task.root.name] = (task.task_id, answer)
        self.flush()

    def record(self, task, answer):
        self.answers[task.root.name] = (task.task_id, answer)
        self.dirty = True
        self.flush()

    def flush(self):
        if not self.dirty:
            return

        def escape(value):
            return value.translate(str.maketrans({"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"}))

        snapshot = "".join(f"{escape(task_id)}\t{escape(answer)}\n"
                           for task_id, answer in sorted(self.answers.values()))
        try:
            atomic_write(self.path, snapshot)
        except OSError as error:
            self.trace({"event": "out_write_error", "error": type(error).__name__})
        else:
            self.dirty = False


def files(root: Path):
    if not root.is_dir() or root.is_symlink():
        return
    for folder, dirs, names in os.walk(root):
        directory = Path(folder)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__"
                         and not (directory / d).is_symlink())
        for name in sorted(names):
            path = directory / name
            if not name.startswith(".") and not path.is_symlink() and path.is_file():
                yield path


def local_file(root: Path, name: str, extensions: set[str] | None = None) -> Path:
    rel = Path(name)
    if not name or rel.is_absolute() or ".." in rel.parts or any(p.startswith(".") for p in rel.parts):
        raise ValueError("use a relative workspace path")
    path = root / rel
    if not path.is_file() or root.is_symlink() or path.resolve() != root.resolve() / rel:
        raise ValueError("workspace file unavailable")
    if extensions and path.suffix.lower() not in extensions:
        raise ValueError("unsupported file format")
    return path


def local_directory(root: Path, name: str) -> Path:
    relative = Path(name)
    path = root / relative
    if (root.is_symlink() or relative.is_absolute() or any(p.startswith(".") for p in relative.parts)
            or not path.is_dir() or path.resolve() != root.resolve() / relative):
        raise ValueError("provide a directory inside the workspace")
    return path


def read_text(path: Path, max_bytes=1_000_000):
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("file exceeds read limit")
    return raw.decode("utf-8-sig")
