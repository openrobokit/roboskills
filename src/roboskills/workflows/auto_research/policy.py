"""Validating a saved `solve(bot)` program: exactly one `solve(bot)`, on `bot` only the given public names, only math
and NumPy, and no files, native libraries, private names, classes, global mutation or attribute assignment. The AST
restrictions constrain the generated interface, not hostile Python."""

import ast

from roboskills.tools.robokit.bot import Tools


VERBS = frozenset(n for n, f in vars(Tools).items() if not n.startswith("_") and callable(f))  # the bot verbs

ALLOWED_IMPORTS = frozenset({"math", "numpy"})
FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "input",
    "breakpoint",
    "help",
    "type",
    "object",
    "super",
}
IO_ATTRIBUTES = {
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "savetxt",
    "genfromtxt",
    "fromfile",
    "tofile",
    "memmap",
    "ctypeslib",
    "f2py",
    "DataSource",
}


def validate(source, public):
    """The parsed tree of `source`, a saved policy: exactly one solve(bot), and `bot.` only for names in `public` (e.g.
    VERBS); ValueError otherwise. It cannot end or score an episode."""
    tree = ast.parse(source)
    entry = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    if len(entry) != 1 or [arg.arg for arg in entry[0].args.args] != ["bot"]:
        raise ValueError("Define exactly one solve(bot) entrypoint")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name not in ALLOWED_IMPORTS for name in names) or getattr(node, "level", 0):
                raise ValueError("Only math and numpy imports are available")
            if isinstance(node, ast.ImportFrom) and any(n.name in IO_ATTRIBUTES or n.name == "*" for n in node.names):
                raise ValueError("Numerical imports cannot access files or native libraries")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            raise ValueError(f"Unavailable name {node.id!r}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in IO_ATTRIBUTES or isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError("Private attributes and attribute mutation are unavailable")
            if isinstance(node.value, ast.Name) and node.value.id == "bot" and node.attr not in public:
                raise ValueError(f"Unavailable robot tool {node.attr!r}")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError("Global mutation and classes are unavailable")
    return tree
