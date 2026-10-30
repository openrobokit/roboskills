"""Hold the README's promise using only the stdlib: each workflow (src/roboskills/workflows/<name>/) stays under LIMIT
lines of Python; tools are not counted. Python counts code lines, including docstrings, excluding blanks and
comment-only lines. Files directly under workflows/ (the package marker) belong to no workflow."""

import io
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1000


def python_loc(source):
    """Count physical code lines, including docstrings, without blanks or comments."""
    lines = source.splitlines()
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    code = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in ignored:
            code.update(row for row in range(token.start[0], token.end[0] + 1) if lines[row - 1].strip())
    return len(code)


def counts(root):
    """Discover tracked and untracked nonignored source; count each path toward its workflow."""
    argv = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"]
    paths = subprocess.check_output([*argv, "src/roboskills/workflows"], cwd=root).decode()
    totals = defaultdict(int)
    for name in sorted(set(paths.split("\0")) - {""}):
        path, parts = root / name, Path(name).parts
        if path.is_file() and path.suffix == ".py" and len(parts) > 4:
            totals[parts[3]] += python_loc(path.read_text(encoding="utf-8"))
    return dict(sorted(totals.items()))


def main(root=ROOT):
    found = counts(root)
    for name, code in sorted(found.items(), key=lambda item: (-item[1], item[0])):
        print(f"{code:>6} {name}")
    over = [name for name, code in found.items() if code >= LIMIT]
    if over:
        print(f"{LIMIT} or more lines of Python: {', '.join(over)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
