"""Build the deterministic offline cfdbender offline kit from this checkout.

Run ``python tools/build_kit.py --out OUTPUT [--wheels CACHE]`` with build,
setuptools>=77, pip and gmsh available in the build interpreter. The cfdbender wheel is
built with ``python -m build --wheel --no-isolation``; gmsh is a build dependency only.
The wheel of every runtime dependency, for each kit target (Windows and Linux, x86-64,
CPython 3.12), is fetched with ``pip download`` into the build folder and packed under
``wheels/``, so the kit installs with ``pip install --no-index --find-links wheels``.
Without ``--wheels`` the wheels are downloaded from the package index; with it they are
taken from that local folder only, and no network is used.
The output directory must exist; an existing deliverable is never overwritten.

Adapted from pyflightstream build_kit_0540.py: explicit members, fresh staging,
fixed ZipInfo timestamps, sorted members, wheel metadata checks and hashes.
The unzip/paste workflow follows the ghc-kit FIRST-PROMPT convention.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "tools/kit"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
NAME = f"cfdbender-kit-{VERSION}.zip"
WHEEL = f"cfdbender-{VERSION}-py3-none-any.whl"
STAMP = (2026, 11, 10, 0, 0, 0)
#: The kit targets: (name, pip download platform tags). The Python is CPython 3.12.
TARGETS = (
    ("windows", ("win_amd64",)),
    ("linux", ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64")),
)
PYTHON_VERSION = "3.12"
EPOCH = "1794268800"
TEMPLATE_FILES = (
    "README-FIRST.md",
    "FIRST-PROMPT.md",
    "SMOKE-CASE.md",
    "smoke/inputs/matrices/smoke.fs",
    "smoke/inputs/setups/s001.toml",
    "smoke/inputs/references/r001.toml",
    "smoke/inputs/pproc/p001.toml",
    "smoke/inputs/hpc/h001.toml",
)


def _name(requirement: str) -> str:
    """Return the distribution name of a requirement string."""
    return re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0].strip().lower()


def run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    """Run one build command and name its failure before publishing any output.

    Parameters
    ----------
    argv : list of str
        Program and arguments without shell interpretation.
    cwd : Path
        Private build directory.
    env : dict of str to str
        Build-local environment.
    """
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError(
            f"command failed ({result.returncode}): {argv[0]}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def regular(path: Path) -> bytes:
    """Read a named regular input, refusing symlinks and Windows junctions.

    Parameters
    ----------
    path : Path
        Input below the supplied repository root.
    """
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError(f"linked kit input refused: {path}")
        if part == ROOT:
            break
    return path.read_bytes()


def check_public(name: str, data: bytes) -> None:
    """Refuse private filesystem paths and Git metadata in packed payloads.

    Parameters
    ----------
    name : str
        Archive member name.
    data : bytes
        Exact payload to inspect.
    """
    if ".git" in Path(name).parts:
        raise ValueError(f"private path or Git metadata refused in {name}")
    if name.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(data)) as wheel:
            for member in wheel.namelist():
                check_public(member, wheel.read(member))
    elif re.search(rb"\b[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+[\\/]", data):
        raise ValueError(f"private path refused in {name}")


def dependency_wheels(
    requirements: list[str], stage: Path, env: dict[str, str], cache: Path | None
) -> dict[str, bytes]:
    """Return ``wheels/<file>`` members for every runtime dependency of every kit target.

    Parameters
    ----------
    requirements : list of str
        The project's runtime dependencies.
    stage : Path
        Private build directory; the wheels are downloaded below it.
    env : dict of str to str
        Build-local environment.
    cache : Path or None
        A local folder of wheels to take them from with no network, or None for the index.
    """
    members: dict[str, bytes] = {}
    for target, platforms in TARGETS:
        folder = stage / "dependencies" / target
        folder.mkdir(parents=True)
        argv = [sys.executable, "-m", "pip", "download", "--dest", str(folder)]
        argv += ["--only-binary=:all:", "--python-version", PYTHON_VERSION]
        argv += ["--implementation", "cp", "--disable-pip-version-check"]
        for platform in platforms:
            argv += ["--platform", platform]
        if cache is not None:
            argv += ["--no-index", "--find-links", str(cache)]
        try:
            run([*argv, *requirements], cwd=stage, env=env)
        except ValueError as error:
            raise ValueError(f"dependency wheels for {target} not fetched: {error}") from error
        for wheel in sorted(folder.glob("*.whl")):
            name = f"wheels/{wheel.name}"
            data = wheel.read_bytes()
            if members.get(name, data) != data:
                raise ValueError(f"two different files named {wheel.name}")
            members[name] = data
    return members


def build(out: Path, cache: Path | None = None) -> Path:
    """Build and verify the one deliverable, publishing only after all checks.

    Parameters
    ----------
    out : Path
        Existing output directory, optionally a deliberately supplied junction.
    cache : Path, optional
        Local folder of dependency wheels; the package index is used when omitted.

    Returns
    -------
    Path
        Completed ZIP. Wheel and ZIP hashes are printed over exact file bytes.
    """
    target = out / NAME
    if not out.is_dir() or target.exists():
        raise ValueError("output directory must exist and the kit name must be unused")
    # Fail before any expensive build when the required mesh API is absent.
    if importlib.util.find_spec("gmsh") is None:
        raise ValueError("gmsh Python API is required in the build interpreter; install gmsh first")
    with tempfile.TemporaryDirectory(prefix="cfdbender-kit-") as temporary:
        stage = Path(temporary)
        source = stage / "source"
        source.mkdir()
        # Copy only packaging inputs. Never traverse the repository or its .git tree.
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            (source / name).write_bytes(regular(ROOT / name))
        source_root = ROOT / "src"
        if source_root.is_symlink() or source_root.is_junction():
            raise ValueError("linked source tree refused")
        for directory, dirs, files in os.walk(source_root, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink() or path.is_junction():
                    raise ValueError(f"linked source input refused: {path}")
            dirs[:] = sorted(name for name in dirs if name != "__pycache__")
            for name in sorted(files):
                path = Path(directory) / name
                schema = path.parent == source_root / "cfdbender/geometry/schemas"
                if not name.endswith(".py") and not (schema and name.endswith(".json")):
                    continue
                destination = source / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(regular(path))
        project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
        if project["version"] != VERSION:
            raise ValueError("kit requires the current cfdbender version")
        requirements = list(project["dependencies"])
        env = dict(os.environ, SOURCE_DATE_EPOCH=EPOCH, PYTHONDONTWRITEBYTECODE="1")
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(stage / "wheels"),
                str(source),
            ],
            cwd=stage,
            env=env,
        )
        wheel_path = stage / "wheels" / WHEEL
        wheel_data = wheel_path.read_bytes()
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata = BytesParser().parsebytes(
                wheel.read(f"cfdbender-{VERSION}.dist-info/METADATA")
            )
        runtime = [r for r in metadata.get_all("Requires-Dist", []) if "extra ==" not in r]
        if metadata["Version"] != VERSION or sorted(map(_name, runtime)) != sorted(
            map(_name, requirements)
        ):
            raise ValueError("wheel version or runtime dependencies differ from the release")
        workspace = stage / "smoke"
        # Dogfood the CLI from the actual wheel, rather than hand-writing init.toml.
        run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from cfdbender.workspace.cli import main; raise SystemExit(main(sys.argv[2:]))",
                str(wheel_path),
                "init",
                "--solver",
                "su2",
                "--workspace",
                str(workspace),
            ],
            cwd=stage,
            env=env,
        )
        spec = importlib.util.spec_from_file_location("kit_mesh", TEMPLATES / "tiny_case.py")
        if spec is None or spec.loader is None:
            raise ValueError("tiny-case generator is unavailable")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        mesh = workspace / "inputs/mesh/tiny.su2"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        cells = generator.generate(mesh)
        if not 100 < cells <= 5000:
            raise ValueError(f"smoke mesh must stay small; got {cells} cells")
        members = {
            name: regular(TEMPLATES / name).replace(b"\r\n", b"\n") for name in TEMPLATE_FILES
        }
        members.update(
            {
                WHEEL: wheel_data,
                "smoke/init.toml": (workspace / "init.toml").read_bytes(),
                "smoke/inputs/mesh/tiny.su2": mesh.read_bytes(),
            }
        )
        for name, data in members.items():
            check_public(name, data)
        # Upstream dependency wheels are packed as published; only this project's files are
        # checked for private paths.
        bundled = dependency_wheels(requirements, stage, env, cache)
        members.update(bundled)
        candidate = stage / NAME
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo(name, STAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)
        with zipfile.ZipFile(candidate) as archive:
            if archive.testzip() or archive.namelist() != sorted(members):
                raise ValueError("kit archive verification failed")
        # Exclusive creation prevents a competing build from silently replacing delivery.
        with target.open("xb") as stream, candidate.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream)
        print(f"kit: {target}")
        print(f"zip sha256: {hashlib.sha256(candidate.read_bytes()).hexdigest()}")
        print(f"wheel sha256: {hashlib.sha256(wheel_data).hexdigest()}")
        print(f"members: {len(members)}; mesh cells: {cells}; dependency wheels: {len(bundled)}")
    return target


def main() -> int:
    """Build the kit in a named output directory, with a clear failure message."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="existing output directory")
    parser.add_argument(
        "--wheels", type=Path, help="local folder of dependency wheels (no network)"
    )
    args = parser.parse_args()
    try:
        build(args.out.resolve(), args.wheels.resolve() if args.wheels else None)
    except (OSError, ValueError, ImportError, subprocess.TimeoutExpired) as error:
        print(f"kit not built: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
