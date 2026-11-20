"""P0000 item 4: build, unzip and use the delivered wheel without network.

Usage: python tools/build_kit.py --out OUTPUT [--wheels CACHE]; unzip, then paste
FIRST-PROMPT.md. The kit carries the wheels of every runtime dependency for Windows and
Linux. These tests build it from a local cache of stand-in dependency wheels, so they need
no network; CFDBENDER_KIT_NETWORK=1 opts in to one build that downloads the real wheels. The
fake SU2 is numerical test evidence only. The opt-in case uses the installed wheel.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
WHEEL = f"cfdbender-{VERSION}-py3-none-any.whl"
ZIP = f"cfdbender-kit-{VERSION}.zip"
FILES = {
    WHEEL,
    "README-FIRST.md",
    "FIRST-PROMPT.md",
    "SMOKE-CASE.md",
    "smoke/init.toml",
    "smoke/inputs/matrices/smoke.fs",
    "smoke/inputs/setups/s001.toml",
    "smoke/inputs/references/r001.toml",
    "smoke/inputs/pproc/p001.toml",
    "smoke/inputs/mesh/tiny.su2",
    "smoke/inputs/hpc/h001.toml",
}
DEPENDENCIES = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "dependencies"
]


def _name(requirement):
    return re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0]


def _stub_wheel(folder, requirement):
    """Write a deterministic stand-in wheel that satisfies ``requirement``."""
    name = _name(requirement)
    floor = re.search(r">=\s*([0-9][0-9.]*)", requirement)
    version = floor[1] if floor else "1.0"
    module = name.replace("-", "_")
    info = f"{module}-{version}.dist-info"
    files = {
        f"{module}_kit_stub/__init__.py": b"",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: kit-test\nRoot-Is-Purelib: true\n"
        b"Tag: py3-none-any\n",
    }
    record = []
    for member, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{member},sha256={digest},{len(data)}")
    record.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    target = folder / f"{module}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        for member, data in files.items():
            archive.writestr(zipfile.ZipInfo(member, (2026, 1, 1, 0, 0, 0)), data)
    return target.name


@pytest.fixture(scope="module")
def wheel_cache(tmp_path_factory):
    """A local cache with one stand-in wheel per runtime dependency (no network)."""
    folder = tmp_path_factory.mktemp("item4-wheels")
    names = sorted(_stub_wheel(folder, requirement) for requirement in DEPENDENCIES)
    return folder, names


def run(argv, *, cwd, env=None):
    result = subprocess.run(
        [str(a) for a in argv], cwd=cwd, env=env, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def kit(tmp_path_factory, wheel_cache):
    folder = tmp_path_factory.mktemp("item4-kit")
    builder = ROOT / "tools/build_kit.py"
    if builder.exists():
        run([sys.executable, builder, "--out", folder, "--wheels", wheel_cache[0]], cwd=ROOT)
    return folder / ZIP


@pytest.fixture(scope="module")
def installed(kit, tmp_path_factory):
    folder = tmp_path_factory.mktemp("item4-unzip")
    if not kit.is_file():
        return folder, folder, Path(sys.executable), {}
    with zipfile.ZipFile(kit) as archive:
        archive.extractall(folder)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PIP_NO_CACHE_DIR="1", PIP_NO_INDEX="1")
    run([sys.executable, "-m", "venv", folder / ".venv"], cwd=folder, env=env)
    scripts = folder / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    install = [python, "-I", "-m", "pip", "install", "--no-index", "--find-links", "wheels"]
    run([*install, folder / WHEEL], cwd=folder, env=env)
    return folder, scripts, python, env


def test_p0000_item4_deterministic(kit, tmp_path, wheel_cache):
    assert kit.is_file(), "the builder must produce the repeatable offline ZIP"
    builder = ROOT / "tools/build_kit.py"
    run([sys.executable, builder, "--out", tmp_path, "--wheels", wheel_cache[0]], cwd=ROOT)
    assert (tmp_path / ZIP).read_bytes() == kit.read_bytes()


def test_p0000_item4_exact_members_and_privacy(kit, wheel_cache):
    assert kit.is_file(), "the kit must carry only the intended portable payloads"
    bundled = {f"wheels/{name}" for name in wheel_cache[1]}
    with zipfile.ZipFile(kit) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == sorted(FILES | bundled)
        for name in archive.namelist():
            if name in bundled:
                continue  # upstream dependency wheels, copied as published
            assert ".git" not in Path(name).parts
            data = archive.read(name)
            if name.endswith(".whl"):
                import io

                with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                    for member in wheel.namelist():
                        assert ".git" not in Path(member).parts
                        payload = wheel.read(member)
                        assert not re.search(rb"\b[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+[\\/]", payload)
                continue
            assert not re.search(rb"\b[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+[\\/]", data)
            assert b"\xe2\x80\x93" not in data and b"\xe2\x80\x94" not in data
        assert len(archive.read("README-FIRST.md").decode().splitlines()) == 15
        mesh = archive.read("smoke/inputs/mesh/tiny.su2").decode()
        assert "NELEM= 1280" in mesh and "airfoil" in mesh and "farfield" in mesh
        assert b'route = "executable"' in archive.read("smoke/inputs/setups/s001.toml")
        profile = tomllib.loads(archive.read("smoke/inputs/hpc/h001.toml").decode())
        assert profile["submit"]["command"][0] == "FAKE_SUBMIT_REPLACE_ME"


def test_p0000_item4_wheel_metadata_and_offline_commands(installed):
    folder, scripts, python, env = installed
    assert (folder / WHEEL).is_file(), "the delivered wheel must install offline"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    with zipfile.ZipFile(folder / WHEEL) as archive:
        metadata = BytesParser().parsebytes(archive.read(f"cfdbender-{VERSION}.dist-info/METADATA"))
    assert metadata["Version"] == project["version"] == VERSION
    runtime = [r for r in metadata.get_all("Requires-Dist", []) if "extra ==" not in r]
    assert sorted(map(_name, runtime)) == sorted(map(_name, DEPENDENCIES))
    probe = run(
        [python, "-I", "-c", "import cfdbender; print(cfdbender.__version__, cfdbender.__file__)"],
        cwd=folder,
        env=env,
    )
    assert VERSION in probe and str(folder / ".venv").lower() in probe.lower()
    for command in ("cfd-workspace", "cfd-matrix"):
        exe = scripts / (command + (".exe" if os.name == "nt" else ""))
        assert command in run([exe, "--help"], cwd=folder, env=env)


def test_p0000_item4_su2_plan_without_mock_or_real_solver(installed):
    folder, scripts, python, original = installed
    assert (folder / "smoke/init.toml").is_file(), "the kit must carry a real SU2 workspace"
    env = dict(original, CFDBENDER_SU2_EXE=str(ROOT / "tests/fixtures/su2/fake_su2.py"))
    matrix = scripts / ("cfd-matrix.exe" if os.name == "nt" else "cfd-matrix")
    # Make mock imports impossible inside this installed interpreter, without editing its wheel.
    code = (
        "import sys,importlib.abc; "
        "exec('class DenyMock(importlib.abc.MetaPathFinder):\\n"
        " def find_spec(self, fullname, path=None, target=None):\\n"
        '  if fullname.startswith("cfdbender.solvers.mock"): '
        'raise ImportError("mock unavailable")\'); '
        "sys.meta_path.insert(0,DenyMock()); from cfdbender.matrix.cli import main; "
        "raise SystemExit(main(sys.argv[1:]))"
    )
    before = {
        str(p.relative_to(folder / "smoke")): p.read_bytes()
        for p in (folder / "smoke").rglob("*")
        if p.is_file()
    }
    output = run(
        [python, "-I", "-c", code, "plan", "smoke", "--local"], cwd=folder / "smoke", env=env
    )
    assert "solver: su2" in output and "1 point(s)" in output
    # The plan-receipt contract: plan writes post/<matrix>/plan.json and nothing else.
    after = {
        str(p.relative_to(folder / "smoke")): p.read_bytes()
        for p in (folder / "smoke").rglob("*")
        if p.is_file()
    }
    receipt = str(Path("post/smoke/plan.json"))
    assert receipt in after and receipt not in before
    assert before == {name: data for name, data in after.items() if name != receipt}
    # Fake execution verifies that planning wasn't quietly satisfied by the mock adapter.
    # run reports on stderr since the 0.8.0 console (the point line says COMPLETED, never the
    # classification), so the convergence is read from the record the fake solver's run left.
    result = subprocess.run(
        [str(matrix), "run", "smoke", "--local"],
        cwd=folder / "smoke",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETED" in result.stderr
    records = json.loads((folder / "smoke" / "runs.json").read_text(encoding="utf-8"))
    assert "COMPLETED" in json.dumps(records) and "converged" in json.dumps(records)
    run([matrix, "collect", "smoke", "--local"], cwd=folder / "smoke", env=env)
    summary = json.loads((folder / "smoke/post/smoke/campaign_summary.json").read_text())
    assert summary["solver"] == "su2"
    assert summary["points"][0]["result"]["solver_version"] == "8.6.0"


def test_p0000_item4_missing_gmsh_fails_loudly(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "tools/build_kit.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "gmsh Python API is required" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "payload", [bytes([90, 58, 47]) + b"FAKE_PATH", bytes([92, 92]) + b"FAKE_HOST/share"]
)
def test_p0000_item4_private_paths_refuse(payload):
    builder = ROOT / "tools/build_kit.py"
    api = SimpleNamespace(check_public=lambda name, data: None)
    if builder.is_file():
        spec = importlib.util.spec_from_file_location("item4_builder", builder)
        assert spec is not None and spec.loader is not None
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
    with pytest.raises(ValueError, match="private path"):
        api.check_public("payload.txt", payload)


@pytest.mark.skipif(not os.environ.get("CFDBENDER_SU2_EXE"), reason="set CFDBENDER_SU2_EXE")
def test_p0000_item4_real_installed_wheel(kit, tmp_path):
    print("kit sha256:", hashlib.sha256(kit.read_bytes()).hexdigest())
    with zipfile.ZipFile(kit) as archive:
        archive.extractall(tmp_path)
    print("FIRST-PROMPT step 1: Python", sys.version)
    assert sys.version_info >= (3, 12)
    temporary = tmp_path / ".tmp"
    temporary.mkdir()
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PIP_NO_CACHE_DIR="1",
        TEMP=str(temporary),
        TMP=str(temporary),
        TMPDIR=str(temporary),
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    print("FIRST-PROMPT step 2: create", tmp_path / ".venv")
    run([sys.executable, "-m", "venv", tmp_path / ".venv"], cwd=tmp_path, env=env)
    scripts = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    print(
        run(
            [
                python,
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                "wheels",
                tmp_path / WHEEL,
            ],
            cwd=tmp_path,
            env=env,
        )
    )
    print(
        run(
            [
                python,
                "-I",
                "-c",
                "import cfdbender; print(cfdbender.__version__, cfdbender.__file__)",
            ],
            cwd=tmp_path,
            env=env,
        )
    )
    with zipfile.ZipFile(tmp_path / WHEEL) as wheel:
        metadata = BytesParser().parsebytes(wheel.read(f"cfdbender-{VERSION}.dist-info/METADATA"))
    runtime = [r for r in metadata.get_all("Requires-Dist", []) if "extra ==" not in r]
    print("runtime dependencies:", ", ".join(runtime))
    matrix = scripts / ("cfd-matrix.exe" if os.name == "nt" else "cfd-matrix")
    workspace = scripts / ("cfd-workspace.exe" if os.name == "nt" else "cfd-workspace")
    for cli in (workspace, matrix):
        print("help command:", cli, "--help")
        print(run([cli, "--help"], cwd=tmp_path, env=env))
    finder = ["where.exe", "SU2_CFD"] if os.name == "nt" else ["which", "SU2_CFD"]
    found = subprocess.run(finder, capture_output=True, text=True, env=env)
    print("FIRST-PROMPT step 3:", " ".join(finder), "exit", found.returncode)
    print(found.stdout + found.stderr)
    executable = Path(env["CFDBENDER_SU2_EXE"])
    assert executable.is_file(), "STOP: step 3: supplied SU2 executable is unavailable"
    print("session CFDBENDER_SU2_EXE:", executable)
    print("FIRST-PROMPT step 4: one process, one OpenMP thread; RAM floor 0.5 GiB")
    for command in ("plan", "run", "collect"):
        print("command:", matrix, command, "smoke --local; cwd:", tmp_path / "smoke")
        print(run([matrix, command, "smoke", "--local"], cwd=tmp_path / "smoke", env=env))
    summary = json.loads((tmp_path / "smoke/post/smoke/campaign_summary.json").read_text())
    result = summary["points"][0]["result"]
    assert result["classification"] == "iteration_limit"
    assert summary["points"][0]["status"] == "COMPLETED"
    assert len(result["history"]) == 40
    print("installed wheel sha256:", hashlib.sha256((tmp_path / WHEEL).read_bytes()).hexdigest())
    print("FIRST-PROMPT step 5 summary:", json.dumps(summary, sort_keys=True))
    print("FIRST-PROMPT step 6:")
    prompt = (tmp_path / "FIRST-PROMPT.md").read_text()
    explanation = prompt.split("exactly these five lines explaining how to use my own case:\n")[1]
    print(explanation.split("\n\n---")[0])


def test_kit_name_is_a_single_builder_constant():
    spec = importlib.util.spec_from_file_location("kit_builder_id", ROOT / "tools/build_kit.py")
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.NAME == f"cfdbender-kit-{builder.VERSION}.zip"
    assert not hasattr(builder, "DELIVERABLE_ID")


def test_first_prompt_isolates_python_from_inherited_paths():
    prompt = (ROOT / "tools/kit/FIRST-PROMPT.md").read_text(encoding="utf-8")
    assert "PYTHONPATH" in prompt and "PYTHONHOME" in prompt
    assert "-I -m pip install --no-index --find-links wheels" in prompt


def test_kit_texts_ask_for_the_python_the_wheels_are_built_for():
    """RG-2: the bundled wheels are CPython 3.12 only; the kit texts must not accept newer."""
    spec = importlib.util.spec_from_file_location("kit_builder_py", ROOT / "tools/build_kit.py")
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    major, minor = builder.PYTHON_VERSION.split(".")
    readme = (ROOT / "tools/kit/README-FIRST.md").read_text(encoding="utf-8")
    prompt = (ROOT / "tools/kit/FIRST-PROMPT.md").read_text(encoding="utf-8")
    for text in (readme, prompt):
        assert f"Python {builder.PYTHON_VERSION}" in text
        assert "or newer" not in text
    assert f"sys.version_info[:2] == ({major}, {minor})" in prompt


def test_p0200_item5_kit_texts_match_current_release():
    readme = (ROOT / "tools/kit/README-FIRST.md").read_text(encoding="utf-8")
    prompt = (ROOT / "tools/kit/FIRST-PROMPT.md").read_text(encoding="utf-8")
    assert f"cfdbender {VERSION} offline kit" in readme
    assert f"cfdbender {VERSION} kit" in prompt
    assert f"installed version {VERSION}" in prompt
    assert WHEEL in prompt


def test_p0000_item4_a_cache_missing_a_dependency_refuses(tmp_path, wheel_cache):
    """A dependency wheel the cache lacks stops the build before any kit is written."""
    partial = tmp_path / "partial"
    partial.mkdir()
    for name in wheel_cache[1][1:]:
        (partial / name).write_bytes((wheel_cache[0] / name).read_bytes())
    out = tmp_path / "out"
    out.mkdir()
    builder = ROOT / "tools/build_kit.py"
    result = subprocess.run(
        [sys.executable, builder, "--out", out, "--wheels", partial],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "dependency wheels" in result.stderr
    assert not list(out.iterdir())


@pytest.mark.skipif(
    os.environ.get("CFDBENDER_KIT_NETWORK") != "1", reason="set CFDBENDER_KIT_NETWORK=1"
)
def test_p0000_item4_network_build_bundles_windows_and_linux_wheels(tmp_path):
    """Opt-in: the default build downloads every dependency wheel for both kit targets."""
    run([sys.executable, ROOT / "tools/build_kit.py", "--out", tmp_path], cwd=ROOT)
    with zipfile.ZipFile(tmp_path / ZIP) as archive:
        wheels = [n for n in archive.namelist() if n.startswith("wheels/")]
    for name in map(_name, DEPENDENCIES):
        found = [w for w in wheels if w.split("/")[1].lower().startswith(name.lower() + "-")]
        assert found, name
    numpy = [w for w in wheels if w.startswith("wheels/numpy-")]
    assert any("win_amd64" in w for w in numpy) and any("manylinux" in w for w in numpy)


def test_cfdpp_job_script_source_passes_kit_privacy_guard():
    """Shell quoting in shipped Python source must not resemble a private path."""
    spec = importlib.util.spec_from_file_location("item4_guard", ROOT / "tools/build_kit.py")
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = ROOT / "src/cfdbender/solvers/cfdpp/_execute.py"
    builder.check_public(source.name, source.read_bytes())
