"""AST analysis to find FP64 hotspots in source code.

Supports: Fortran (primary), C, C++.
Detects: triple-nested loops over double arrays, DGEMM/DGEMV calls.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Hotspot:
    file: Path
    kind: str          # "loop_nest" | "dgemm_call" | "dgemv_call"
    language: str      # "fortran" | "c" | "cpp"
    start_line: int
    end_line: int
    context: str = ""  # surrounding lines for rewrite context
    vars: list[str] = field(default_factory=list)  # detected double vars


# ── Fortran ──────────────────────────────────────────────────────────────────

_F_DOUBLE_DECL = re.compile(
    r"^\s*(real\s*\(\s*(?:kind\s*=\s*)?(?:8|dp|kind\(0\.d0\))\s*\)"
    r"|real\s*\*\s*8"
    r"|double\s+precision)",
    re.IGNORECASE,
)

_F_DO = re.compile(r"^\s*do\b", re.IGNORECASE)
_F_ENDDO = re.compile(r"^\s*end\s*do\b", re.IGNORECASE)
_F_DGEMM = re.compile(r"\bcall\s+dgemm\s*\(", re.IGNORECASE)
_F_DGEMV = re.compile(r"\bcall\s+dgemv\s*\(", re.IGNORECASE)


def _fortran_hotspots(path: Path) -> list[Hotspot]:
    lines = path.read_text(errors="replace").splitlines()
    hotspots: list[Hotspot] = []
    double_vars: list[str] = []

    # Collect declared double-precision variable names
    for line in lines:
        if _F_DOUBLE_DECL.match(line):
            # grab identifiers after "::" or after the type declaration
            decl = re.sub(r".*::\s*", "", line, count=1)
            decl = re.sub(r"\(.*?\)", "", decl)  # strip dimensions
            for v in re.split(r"[,\s]+", decl):
                v = v.strip()
                if v and re.match(r"[A-Za-z_]\w*", v):
                    double_vars.append(v.lower())

    # Find DGEMM/DGEMV calls
    for i, line in enumerate(lines):
        if _F_DGEMM.search(line):
            ctx = "\n".join(lines[max(0, i-2):i+6])
            hotspots.append(Hotspot(
                file=path, kind="dgemm_call", language="fortran",
                start_line=i+1, end_line=i+1, context=ctx, vars=double_vars,
            ))
        if _F_DGEMV.search(line):
            ctx = "\n".join(lines[max(0, i-2):i+4])
            hotspots.append(Hotspot(
                file=path, kind="dgemv_call", language="fortran",
                start_line=i+1, end_line=i+1, context=ctx, vars=double_vars,
            ))

    # Find triple-nested DO loops over double vars.
    # We track when depth first reaches 3 (innermost of a triple nest), record
    # the outermost loop start, then wait for the outermost end do.
    do_stack: list[int] = []
    nest_start: int | None = None  # outermost loop line when triple nest detected

    for i, line in enumerate(lines):
        if _F_DO.match(line):
            do_stack.append(i)
        elif _F_ENDDO.match(line) and do_stack:
            start = do_stack.pop()
            depth = len(do_stack) + 1  # depth of the loop we just closed

            # Mark start of the outermost loop the first time we see depth >= 3
            if depth >= 3 and nest_start is None:
                nest_start = do_stack[0] if do_stack else start

            # Once outermost closes (stack empty), evaluate the full nest
            if nest_start is not None and len(do_stack) == 0:
                body = "\n".join(lines[nest_start:i+1])
                uses_double = bool(double_vars) and any(
                    re.search(rf"\b{re.escape(v)}\b", body, re.IGNORECASE)
                    for v in double_vars
                )
                if uses_double:
                    ctx = "\n".join(lines[max(0, nest_start-3):i+2])
                    hotspots.append(Hotspot(
                        file=path, kind="loop_nest", language="fortran",
                        start_line=nest_start+1, end_line=i+1,
                        context=ctx, vars=double_vars,
                    ))
                nest_start = None
                break

    return hotspots


# ── C / C++ ──────────────────────────────────────────────────────────────────

_C_DOUBLE_DECL = re.compile(r"\bdouble\b")
_C_FOR = re.compile(r"^\s*for\s*\(")
_C_DGEMM = re.compile(r"\bcblas_dgemm\s*\(|\bdgemm_\s*\(|\bDGEMM\s*\(")
_C_DGEMV = re.compile(r"\bcblas_dgemv\s*\(|\bdgemv_\s*\(|\bDGEMV\s*\(")


def _c_hotspots(path: Path, lang: str) -> list[Hotspot]:
    lines = path.read_text(errors="replace").splitlines()
    hotspots: list[Hotspot] = []

    double_vars = [
        m.group(1)
        for line in lines
        for m in [re.search(r"\bdouble\s+\**(\w+)", line)]
        if m
    ]

    for i, line in enumerate(lines):
        for pat, kind in [(_C_DGEMM, "dgemm_call"), (_C_DGEMV, "dgemv_call")]:
            if pat.search(line):
                ctx = "\n".join(lines[max(0, i-2):i+4])
                hotspots.append(Hotspot(
                    file=path, kind=kind, language=lang,
                    start_line=i+1, end_line=i+1, context=ctx, vars=double_vars,
                ))

    # Simple triple-nested for-loop detection
    for_stack: list[int] = []
    brace_stack: list[Optional[int]] = []  # index into for_stack when '{' seen

    depth = 0
    for_starts: dict[int, int] = {}  # brace_depth → line index of matching for

    i = 0
    while i < len(lines):
        line = lines[i]
        if _C_FOR.match(line):
            for_stack.append(i)
        opens = line.count("{")
        closes = line.count("}")
        for _ in range(opens):
            if for_stack and (not for_starts or depth not in for_starts):
                for_starts[depth] = for_stack[-1]
            depth += 1
        for _ in range(closes):
            depth -= 1
            if depth in for_starts:
                start = for_starts.pop(depth)
                # depth was 3 meaning we just closed the innermost of a triple nest
                if depth == 2:
                    body = "\n".join(lines[start:i+1])
                    if _C_DOUBLE_DECL.search(body) or any(v in body for v in double_vars):
                        ctx = "\n".join(lines[max(0, start-2):i+2])
                        hotspots.append(Hotspot(
                            file=path, kind="loop_nest", language=lang,
                            start_line=start+1, end_line=i+1,
                            context=ctx, vars=double_vars,
                        ))
                        break
        i += 1

    return hotspots


# ── Public API ────────────────────────────────────────────────────────────────

_LANG_EXTS = {
    ".f90": "fortran", ".f95": "fortran", ".f03": "fortran",
    ".f": "fortran", ".F90": "fortran", ".F": "fortran",
    ".c": "c",
    ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp",
}

_LANG_PRIORITY = ["fortran", "c", "cpp"]


def analyze(repo_dir: Path) -> list[Hotspot]:
    """Return all FP64 hotspots in repo_dir, sorted by language priority."""
    by_lang: dict[str, list[Hotspot]] = {l: [] for l in _LANG_PRIORITY}

    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        lang = _LANG_EXTS.get(path.suffix) or _LANG_EXTS.get(path.name.split(".")[-1], "")
        if not lang:
            continue
        if lang == "fortran":
            by_lang["fortran"].extend(_fortran_hotspots(path))
        elif lang in ("c", "cpp"):
            by_lang[lang].extend(_c_hotspots(path, lang))

    result: list[Hotspot] = []
    for lang in _LANG_PRIORITY:
        result.extend(by_lang[lang])
    return result
