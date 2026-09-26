"""Static checks on the MATLAB sources, for use before MATLAB is available.

Not a MATLAB parser -- a tokenizer that catches the mistakes most likely in code
written without an interpreter to hand:

* block structure: every if/for/while/switch/try/function/classdef/properties/
  methods/parfor is closed by exactly one ``end`` (``end`` inside an index
  expression is recognised and ignored), functions included, and each ``end``
  on its own line is indented like its opener (catches a misplaced ``end`` that
  still balances);
* brackets balanced within each statement (continuations followed), strings
  terminated;
* every ``drscreen.<pkg>.<fn>`` / ``drscreen.<Class>.<method>`` reference in code
  resolves to a file (and, for class references, to a method of that class).

    python matlab/tools/lint_matlab.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENERS = {"if", "for", "while", "switch", "try", "function", "classdef", "parfor",
           "spmd", "properties", "methods", "events", "enumeration"}
TRANSPOSE_AFTER = set(")]}_.'\"")


def strip_line(line: str) -> tuple[str, bool]:
    """Remove comments and string contents; return (code, continues).

    MATLAB's rule for a quote: it is a transpose when it immediately follows
    (no whitespace) an identifier character, a closing bracket, a dot or
    another quote; otherwise it opens a character vector.
    """
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "%":
            return "".join(out), False
        if line.startswith("...", i):
            return "".join(out), True
        if c == '"':
            j = i + 1
            while j < n:
                if line[j] == '"':
                    if j + 1 < n and line[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise ValueError("unterminated double-quoted string")
            out.append('""')
            i = j + 1
            continue
        if c == "'":
            prev = line[i - 1] if i > 0 else ""
            if prev and (prev.isalnum() or prev in TRANSPOSE_AFTER):
                out.append("'")
                i += 1
                continue
            j = i + 1
            while j < n:
                if line[j] == "'":
                    if j + 1 < n and line[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise ValueError("unterminated single-quoted string")
            out.append("''")
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), False


TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[()\[\]{},;]|\S")


def check_file(path: Path) -> list[str]:
    errs = []
    stack: list[tuple[str, int, int]] = []
    brackets: list[str] = []
    in_block_comment = False
    stmt_start = True
    for ln, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if s == "%{":
            in_block_comment = True
            continue
        if s == "%}":
            in_block_comment = False
            continue
        if in_block_comment:
            continue
        try:
            code, cont = strip_line(raw)
        except ValueError as e:
            errs.append(f"{path.name}:{ln}: {e}")
            continue
        prev = None
        indent = len(raw) - len(raw.lstrip(" "))
        for t in TOKEN.findall(code):
            depth = len(brackets)
            if t in "([{":
                brackets.append(t)
                stmt_start = False
            elif t in ")]}":
                if not brackets or brackets[-1] != {")": "(", "]": "[", "}": "{"}[t]:
                    errs.append(f"{path.name}:{ln}: mismatched closing '{t}'")
                    brackets.clear()
                else:
                    brackets.pop()
                stmt_start = False
            elif depth == 0 and t in (",", ";"):
                stmt_start = True
            else:
                if depth == 0 and stmt_start and prev != ".":
                    if t in OPENERS:
                        stack.append((t, ln, indent))
                    elif t == "end":
                        if not stack:
                            errs.append(f"{path.name}:{ln}: 'end' without an open block")
                        else:
                            kw, oln, oind = stack.pop()
                            # house style: an 'end' on its own line sits under its opener
                            if oln != ln and code.strip() == "end" and indent != oind:
                                errs.append(f"{path.name}:{ln}: 'end' indented {indent}, "
                                            f"its '{kw}' (line {oln}) {oind} -- misplaced end?")
                # after a block keyword the next tokens are its expression
                stmt_start = False
            prev = t
        # A newline inside [] or {} is a row separator; inside () it is an error.
        if not cont and brackets and brackets[-1] == "(":
            errs.append(f"{path.name}:{ln}: '(' not closed by end of line (missing '...'?)")
            brackets.clear()
        if not brackets:
            stmt_start = True
    if brackets:
        errs.append(f"{path.name}: brackets still open at EOF: {brackets}")
    if stack:
        # house style: every function is closed by its own 'end'
        errs.append(f"{path.name}: unclosed blocks at EOF: {[(k, l) for k, l, _ in stack]}")
    return errs


def code_text(f: Path) -> str:
    lines = []
    for raw in f.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(strip_line(raw)[0])
        except ValueError:
            lines.append("")
    return "\n".join(lines)


def resolve_refs(files: list[Path]) -> list[str]:
    errs = []
    pkg = ROOT / "+drscreen"
    ref = re.compile(r"\bdrscreen\.([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?")
    for f in files:
        text = code_text(f)
        for m in ref.finditer(text):
            a, b = m.group(1), m.group(2)
            line = text.count("\n", 0, m.start()) + 1
            if (pkg / f"{a}.m").exists():
                if b:
                    cls = (pkg / f"{a}.m").read_text(encoding="utf-8")
                    if not cls.lstrip().startswith("classdef"):
                        errs.append(f"{f.name}:{line}: drscreen.{a} is a function, used as drscreen.{a}.{b}")
                    elif not re.search(rf"^\s*function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?{b}\b", cls, re.M):
                        errs.append(f"{f.name}:{line}: drscreen.{a} has no method {b}")
                continue
            if (pkg / f"+{a}").is_dir():
                if b and not (pkg / f"+{a}" / f"{b}.m").exists():
                    errs.append(f"{f.name}:{line}: missing drscreen.{a}.{b}")
                continue
            errs.append(f"{f.name}:{line}: unknown drscreen.{a}")
    return errs


def logical_lines(f: Path) -> list[tuple[int, str]]:
    """Stripped code with '...' continuations joined; (first line number, text)."""
    out, buf, start = [], "", None
    for ln, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        try:
            code, cont = strip_line(raw)
        except ValueError:
            code, cont = "", False
        if start is None:
            start = ln
        buf += code + " "
        if not cont:
            out.append((start, buf))
            buf, start = "", None
    return out


SIG = re.compile(r"^\s*function\s+(?:(\[[^\]]*\]|\w+)\s*=\s*)?(\w+)\s*(?:\(([^)]*)\))?")


def signatures() -> dict[str, tuple[int, int]]:
    """drscreen.<pkg>.<fn> -> (max inputs, max outputs); -1 means varargin/varargout."""
    sigs = {}
    pkg = ROOT / "+drscreen"
    for f in pkg.rglob("*.m"):
        text = f.read_text(encoding="utf-8")
        if text.lstrip().startswith("classdef"):
            continue
        rel = f.relative_to(pkg).with_suffix("")
        name = ".".join(["drscreen"] + [p.lstrip("+") for p in rel.parts])
        for _, line in logical_lines(f):
            m = SIG.match(line)
            if m:
                outs = [o for o in re.split(r"[\s,]+", (m.group(1) or "").strip("[] ")) if o]
                ins = [a for a in re.split(r"\s*,\s*", (m.group(3) or "").strip()) if a]
                sigs[name] = (-1 if "varargin" in ins else len(ins),
                              -1 if "varargout" in outs else len(outs))
                break
    return sigs


def count_args(text: str, open_idx: int) -> int:
    depth, n, seen = 0, 0, False
    for c in text[open_idx:]:
        if c in "([{":
            depth += 1
            if depth == 1:
                continue
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return n + (1 if seen else 0)
        if depth == 1:
            if c == ",":
                n += 1
                seen = False
                continue
            if not c.isspace():
                seen = True
    return -1


def check_calls(files: list[Path]) -> list[str]:
    errs = []
    sigs = signatures()
    call = re.compile(r"(@?)\b(drscreen(?:\.\w+){1,2})\b(\s*\()?")
    for f in files:
        for ln, line in logical_lines(f):
            for m in call.finditer(line):
                name = m.group(2)
                if m.group(1) or name not in sigs:
                    continue
                max_in, max_out = sigs[name]
                n_in = count_args(line, m.end() - 1) if m.group(3) else 0
                if max_in >= 0 and n_in > max_in:
                    errs.append(f"{f.name}:{ln}: {name} called with {n_in} inputs, takes {max_in}")
                lhs = re.match(r"^\s*\[([^\]]*)\]\s*=\s*$", line[:m.start()])
                if lhs and max_out >= 0:
                    n_out = len([o for o in re.split(r"[\s,]+", lhs.group(1).strip()) if o])
                    if n_out > max_out:
                        errs.append(f"{f.name}:{ln}: {name} asked for {n_out} outputs, returns {max_out}")
    return errs


def main() -> int:
    files = sorted(p for p in ROOT.rglob("*.m") if "tools" not in p.parts)
    errs = []
    for f in files:
        errs += check_file(f)
    errs += resolve_refs(files)
    errs += check_calls(files)
    print(f"checked {len(files)} MATLAB files")
    for e in errs:
        print("  " + e)
    print("no problems found" if not errs else f"{len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
