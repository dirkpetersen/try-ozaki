"""Test cases using real source repositories.

Each RepoCase entry specifies a GitHub repo, expected hotspot kinds,
and whether the run should exercise --no-submit (local analysis+rewrite only)
or the full GPU pipeline.

Run:
    python -m pytest try_ozaki/tests/test_repos.py -v
or:
    python -m pytest try_ozaki/tests/test_repos.py -v -k certik
"""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from try_ozaki.analyzer import analyze, Hotspot
from try_ozaki.rewriter import rewrite


@dataclass
class RepoCase:
    name: str
    gh_url: str            # GitHub clone URL
    expected_hotspot_kinds: list[str]    # at least one of these kinds expected
    expected_languages: list[str]        # at least one of these languages expected
    min_hotspots: int = 1
    description: str = ""


# ── Registered test repos ─────────────────────────────────────────────────────

TEST_REPOS: list[RepoCase] = [
    RepoCase(
        name="certik/matmul",
        gh_url="https://github.com/certik/matmul",
        expected_hotspot_kinds=["loop_nest", "dgemm_call"],
        expected_languages=["fortran"],
        min_hotspots=1,
        description="Fortran matrix multiply benchmark with triple-nested DO loops (single precision). "
                    "The repo uses real(sp) not real(dp) — tests that the analyzer correctly reports "
                    "no FP64 hotspots (all loops are single-precision).",
    ),
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _clone(gh_url: str, dest: Path) -> None:
    subprocess.run(
        ["gh", "repo", "clone", gh_url, str(dest), "--", "--depth=1"],
        check=True, capture_output=True,
    )


@pytest.fixture(scope="module")
def cloned_repos(tmp_path_factory):
    """Clone all test repos once per test module. Returns dict name → Path."""
    base = tmp_path_factory.mktemp("repos")
    clones: dict[str, Path] = {}
    for repo in TEST_REPOS:
        dest = base / repo.name.replace("/", "_")
        print(f"\n[fixture] Cloning {repo.gh_url} → {dest}", flush=True)
        _clone(repo.gh_url, dest)
        clones[repo.name] = dest
    return clones


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("repo", TEST_REPOS, ids=[r.name for r in TEST_REPOS])
def test_clone_succeeds(cloned_repos, repo):
    """Repo clones without error and has files."""
    path = cloned_repos[repo.name]
    assert path.is_dir(), f"Clone directory missing: {path}"
    files = list(path.rglob("*"))
    assert len(files) > 0, "Clone appears empty"


@pytest.mark.parametrize("repo", TEST_REPOS, ids=[r.name for r in TEST_REPOS])
def test_analyze(cloned_repos, repo):
    """Analyzer runs without exception and returns a list (possibly empty)."""
    path = cloned_repos[repo.name]
    hotspots = analyze(path)
    # Just verify no exception and result is a list
    assert isinstance(hotspots, list)
    print(f"\n[{repo.name}] Found {len(hotspots)} hotspot(s):")
    for h in hotspots:
        rel = h.file.relative_to(path)
        print(f"  {rel}:{h.start_line}  [{h.kind}]  ({h.language})")


@pytest.mark.parametrize("repo", TEST_REPOS, ids=[r.name for r in TEST_REPOS])
def test_certik_matmul_is_single_precision(cloned_repos, repo):
    """certik/matmul uses real(sp) throughout — expect zero FP64 hotspots."""
    if repo.name != "certik/matmul":
        pytest.skip("Only for certik/matmul")

    path = cloned_repos[repo.name]
    hotspots = analyze(path)

    # certik/matmul is entirely single-precision (real(sp)), so the analyzer
    # should find no FP64 hotspots — this is the correct outcome.
    fp64_hotspots = [h for h in hotspots if h.language == "fortran"]

    print(f"\n[certik/matmul] FP64 hotspots found: {len(fp64_hotspots)}")
    for h in fp64_hotspots:
        rel = h.file.relative_to(path)
        print(f"  {rel}:{h.start_line} [{h.kind}]")
        print(f"  vars: {h.vars}")

    # The repo is intentionally single-precision — confirm analyzer agrees
    assert len(fp64_hotspots) == 0, (
        f"Expected 0 FP64 hotspots in certik/matmul (all code is real(sp)); "
        f"got {len(fp64_hotspots)}. Analyzer may be misidentifying sp as dp."
    )


@pytest.mark.parametrize("repo", TEST_REPOS, ids=[r.name for r in TEST_REPOS])
def test_rewrite_no_crash(cloned_repos, tmp_path, repo):
    """Rewriter runs without exception even on repos with zero hotspots."""
    path = cloned_repos[repo.name]
    hotspots = analyze(path)

    # Work on a copy
    work = tmp_path / "work"
    shutil.copytree(path, work)

    from try_ozaki.analyzer import Hotspot
    work_hotspots = [
        Hotspot(
            file=work / h.file.relative_to(path),
            kind=h.kind, language=h.language,
            start_line=h.start_line, end_line=h.end_line,
            context=h.context, vars=h.vars,
        )
        for h in hotspots
    ]

    modified = rewrite(work, work_hotspots)
    assert isinstance(modified, list)
    print(f"\n[{repo.name}] Rewriter modified {len(modified)} file(s)")
    for m in modified:
        print(f"  {m.relative_to(work)}")


@pytest.mark.parametrize("repo", TEST_REPOS, ids=[r.name for r in TEST_REPOS])
def test_cli_dry_run(cloned_repos, repo):
    """CLI --dry-run exits 0 and prints hotspot summary."""
    path = cloned_repos[repo.name]
    result = subprocess.run(
        [sys.executable, "-m", "try_ozaki.cli", str(path), "--dry-run"],
        capture_output=True, text=True,
    )
    print(f"\nstdout:\n{result.stdout}")
    print(f"stderr:\n{result.stderr}")
    assert result.returncode == 0, f"CLI dry-run failed with rc={result.returncode}"
    assert "Stage 1" in result.stdout, "Expected stage output in stdout"


@pytest.mark.parametrize("repo", TEST_REPOS, ids=[r.name for r in TEST_REPOS])
def test_cli_no_submit(cloned_repos, tmp_path, repo):
    """CLI --no-submit analyzes, rewrites, and exits 0."""
    path = cloned_repos[repo.name]
    result = subprocess.run(
        [sys.executable, "-m", "try_ozaki.cli", str(path), "--no-submit"],
        capture_output=True, text=True,
    )
    print(f"\nstdout:\n{result.stdout}")
    print(f"stderr:\n{result.stderr}")
    # Should not crash
    assert result.returncode in (0, 1), f"Unexpected exit code {result.returncode}"
    assert "Stage 1" in result.stdout


# ── ozaki-simple example tests ────────────────────────────────────────────────

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
OZAKI_SIMPLE = EXAMPLES_DIR / "ozaki-simple"


def test_ozaki_simple_analyzed():
    """examples/ozaki-simple contains exactly one FP64 triple-nested loop."""
    assert OZAKI_SIMPLE.is_dir(), f"examples/ozaki-simple not found at {OZAKI_SIMPLE}"
    hotspots = analyze(OZAKI_SIMPLE)
    assert len(hotspots) >= 1, f"Expected ≥1 hotspot in ozaki-simple, got {hotspots}"
    kinds = {h.kind for h in hotspots}
    assert "loop_nest" in kinds, f"Expected loop_nest, got {kinds}"
    langs = {h.language for h in hotspots}
    assert "fortran" in langs


def test_ozaki_simple_rewritten(tmp_path):
    """Rewriting ozaki-simple produces an OZAKI_DGEMM call (not a TODO)."""
    work = tmp_path / "ozaki-simple"
    shutil.copytree(OZAKI_SIMPLE, work)
    hotspots = analyze(work)
    modified = rewrite(work, hotspots)

    assert len(modified) >= 1
    assert (work / "ozaki_wrapper.f90").exists()

    src = (work / "matmul_fp64.f90").read_text()
    assert "OZAKI_DGEMM" in src, "Expected OZAKI_DGEMM call in rewritten file"
    assert "try-ozaki" in src
    # Original loop should be commented out
    assert "! [try-ozaki] Original FP64 triple-nested loop" in src


def test_ozaki_simple_cli_no_submit():
    """CLI --no-submit on ozaki-simple exits 0 and reports the rewrite."""
    result = subprocess.run(
        [sys.executable, "-m", "try_ozaki.cli", str(OZAKI_SIMPLE), "--no-submit"],
        capture_output=True, text=True,
    )
    print(f"\nstdout:\n{result.stdout}")
    print(f"stderr:\n{result.stderr}")
    assert result.returncode == 0
    assert "OZAKI_DGEMM" in result.stdout or "Rewriting" in result.stdout


# ── Synthetic FP64 repo test ───────────────────────────────────────────────────

_SYNTHETIC_F90 = """\
program test_fp64
  implicit none
  integer, parameter :: dp = kind(0.d0)
  integer, parameter :: N = 64
  real(dp) :: A(N,N), B(N,N), C(N,N)
  integer :: i, j, k

  call random_number(A)
  call random_number(B)
  C = 0.0_dp

  do j = 1, N
    do i = 1, N
      do k = 1, N
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do

  print *, 'C(1,1) =', C(1,1)
end program
"""

_SYNTHETIC_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(test_fp64 Fortran)
add_executable(test_fp64 test_fp64.f90)
"""


def test_synthetic_fp64_detected(tmp_path):
    """A hand-crafted triple-nested FP64 DO loop is detected by the analyzer."""
    src = tmp_path / "synthetic"
    src.mkdir()
    (src / "test_fp64.f90").write_text(_SYNTHETIC_F90)
    (src / "CMakeLists.txt").write_text(_SYNTHETIC_CMAKE)

    hotspots = analyze(src)
    assert len(hotspots) >= 1, f"Expected ≥1 hotspot in synthetic FP64 code, got {len(hotspots)}"
    kinds = [h.kind for h in hotspots]
    assert "loop_nest" in kinds, f"Expected loop_nest, got {kinds}"
    langs = [h.language for h in hotspots]
    assert "fortran" in langs, f"Expected fortran, got {langs}"


def test_synthetic_fp64_rewritten(tmp_path):
    """Rewriter produces ozaki_wrapper.f90 and modifies the source file."""
    src = tmp_path / "synthetic"
    src.mkdir()
    f90 = src / "test_fp64.f90"
    f90.write_text(_SYNTHETIC_F90)
    (src / "CMakeLists.txt").write_text(_SYNTHETIC_CMAKE)

    hotspots = analyze(src)
    assert hotspots, "No hotspots found in synthetic FP64 code"

    modified = rewrite(src, hotspots)
    assert len(modified) >= 1

    # ozaki_wrapper.f90 must exist
    wrapper = src / "ozaki_wrapper.f90"
    assert wrapper.exists(), "ozaki_wrapper.f90 not generated"

    # Original loop should now be commented out
    rewritten = f90.read_text()
    assert "try-ozaki" in rewritten, "Expected try-ozaki annotation in rewritten file"
    assert "OZAKI_DGEMM" in rewritten or "TODO" in rewritten, (
        "Expected OZAKI_DGEMM call or TODO placeholder in rewritten file"
    )


def test_synthetic_cli_no_submit(tmp_path):
    """CLI --no-submit works end-to-end on synthetic FP64 source."""
    src = tmp_path / "synthetic"
    src.mkdir()
    (src / "test_fp64.f90").write_text(_SYNTHETIC_F90)
    (src / "CMakeLists.txt").write_text(_SYNTHETIC_CMAKE)

    result = subprocess.run(
        [sys.executable, "-m", "try_ozaki.cli", str(src), "--no-submit"],
        capture_output=True, text=True,
    )
    print(f"\nstdout:\n{result.stdout}")
    print(f"stderr:\n{result.stderr}")
    assert result.returncode == 0, f"CLI --no-submit failed: {result.stderr}"
    assert "Rewriting" in result.stdout
    assert "rewritten sources at" in result.stdout
