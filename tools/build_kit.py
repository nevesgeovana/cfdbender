"""Build the deterministic offline cfdbender offline kit from this checkout.

Run ``python tools/build_kit.py --out OUTPUT`` with build, setuptools>=77 and
gmsh available in the build interpreter. No network is used: the wheel is built
with ``python -m build --wheel --no-isolation``. gmsh is a build dependency only.
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


def build(out: Path) -> Path:
    """Build and verify the one deliverable, publishing only after all checks.

    Parameters
    ----------
    out : Path
        Existing output directory, optionally a deliberately supplied junction.

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
                if not name.endswith(".py"):
                    continue
                path = Path(directory) / name
                destination = source / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(regular(path))
        project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
        if project["version"] != VERSION or project["dependencies"]:
            raise ValueError(
                "kit requires the current cfdbender version with no runtime dependencies"
            )
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
        if metadata["Version"] != VERSION or any(
            "extra ==" not in req for req in metadata.get_all("Requires-Dist", [])
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
        mesh = workspace / "inputs/meshes/tiny.su2"
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
                "smoke/inputs/meshes/tiny.su2": mesh.read_bytes(),
            }
        )
        for name, data in members.items():
            check_public(name, data)
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
        print(f"members: {len(members)}; mesh cells: {cells}; runtime dependencies: none")
    return target


def main() -> int:
    """Build the kit in a named output directory, with a clear failure message."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="existing output directory")
    args = parser.parse_args()
    try:
        build(args.out.resolve())
    except (OSError, ValueError, ImportError, subprocess.TimeoutExpired) as error:
        print(f"kit not built: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
