"""Write a deterministic source release with frozen validation inputs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent.parent
VERSION = "2026.11.09-rc1"
ROOT_FILES = (".dockerignore", ".env.example", ".gitignore", "Dockerfile", "README.md", "requirements.in",
              "requirements.txt", "run_agent.py")


def pack(output, validation, version=VERSION, suites=("development", "qualification", "transfer")):
    files = {name: (ROOT / name).read_bytes() for name in ROOT_FILES}
    for directory in ("ping_floyd", "bench"):
        for path in (ROOT / directory).glob("*.py"):
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("results.json", "calibration.json", "report.md", "tests.log"):
        path = validation / name
        if path.is_file():
            files[f"validation/{name}"] = path.read_bytes()
        elif name == "results.json":
            raise ValueError("missing validation/results.json")
    for name in suites:
        if not name or Path(name).name != name or name.startswith("."):
            raise ValueError("suite names must be directory basenames")
        suite = validation / name
        if not (suite / "oracle.json").is_file() or not (suite / "tasks").is_dir():
            raise ValueError(f"missing release validation suite: {name}")
        for path in suite.rglob("*"):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                if path.name == "answer.txt":
                    raise ValueError("release requires pristine validation inputs")
                files[f"validation/{name}/{path.relative_to(suite).as_posix()}"] = path.read_bytes()
    manifest = {"version": version, "sha256": {name: sha256(data).hexdigest() for name, data in sorted(files.items())}}
    files["release-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with ZipFile(output, "x") as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, (2026, 11, 9, 0, 0, 0))
            archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=6)
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--suite", action="append")
    args = parser.parse_args()
    pack(args.output, args.validation, args.version, args.suite or ("development", "qualification", "transfer"))


if __name__ == "__main__":
    main()
