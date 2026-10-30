import subprocess

from scripts.count_loc import LIMIT, counts, main, python_loc


def test_python_loc_handles_strings_comments_and_multiline_code():
    source = '''# comment
"""Documentation.

# Still a string.
"""
value = (
    "# not a comment"  # trailing comment
    # comment inside expression
)
'''
    assert python_loc(source) == 6
    assert python_loc("") == 0
    assert python_loc("# comment\n\n") == 0
    assert python_loc("value = 1") == 1


def test_discovery_and_the_limit(tmp_path, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    files = {
        "src/roboskills/workflows/demo/glue.py": "# comment\nvalue = 1\n",
        "src/roboskills/workflows/demo/SKILL.md": "# Prompt\n\nInstructions\n",
        "src/roboskills/workflows/__init__.py": '"""Workflows."""\n',
        "src/roboskills/tools/__init__.py": '"""Tools."""\n',
        "src/roboskills/workflows/saw/saw.py": "value = 1\nother = 2\n",
        "src/roboskills/tools/hammer/hammer.py": "value = 1\n",
        "src/roboskills/tools/docs_only/SKILL.md": "# Tool\n",
        "src/roboskills/workflows/demo/ignored.py": "pass\n",
        "src/roboskills/eval/runtime.py": "pass\n",
        ".gitignore": "ignored.py\n",
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    subprocess.run(["git", "add", "src/roboskills/workflows/demo/glue.py"], cwd=tmp_path, check=True)
    assert counts(tmp_path) == {"demo": 1, "saw": 2}
    assert main(root=tmp_path) == 0
    assert "2 saw" in capsys.readouterr().out
    (tmp_path / "src/roboskills/workflows/saw/saw.py").write_text("value = 1\n" * LIMIT)
    assert main(root=tmp_path) == 1
    assert "saw" in capsys.readouterr().err
