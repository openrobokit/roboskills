"""Model CLIs: one-shot `claude -p` and `codex exec` calls with tools, settings, memory and project files off, and the
DeepSeek chat API (`deepseek`). Any
`(system, messages) -> str` callable can stand in for `claude`. Every `claude` system prompt ends with PRIVATE, and
`codex` callers append it to their instructions: agents never see artifacts/."""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from roboskills.core.process import terminate


LAST = threading.local()  # .usage: this thread's latest `claude` call's tokens, cost and durations (agent.run logs it)
USAGE_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens",
              "total_cost_usd", "duration_ms", "duration_api_ms")  # fmt: skip
LONG_SYSTEM = 100_000  # characters; Linux refuses a single argument over 128 KiB
PRIVATE = ("Do not look through `artifacts/`: never read, open, import or run a file there "
           "that your instructions do not name, in tool calls or in code you write. They hold evaluation code, results "
           "and other tasks' solutions. Do not retrieve them through other branches, worktrees, Git history or remote "
           "links. Current-run feedback and relevant debugging logs in runs/ are allowed; do not read prior solutions "
           "or private evaluation data from older runs unless your instructions explicitly provide those inputs.")  # fmt: skip
ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"  # a tool-using agent's Read tool is denied them


def claude_command(system, model="claude-opus-5-5", effort=None, *, system_file=None):
    """`claude -p` taking one stream-json user message: no tools, MCP, settings or session. ``system_file`` passes the
    system prompt as a file instead (no argument-length limit; the CLI caches it across calls like any system prompt)."""
    command = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--no-session-persistence", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               *(["--system-prompt-file", str(system_file)] if system_file else ["--system-prompt", system]),
               "--append-system-prompt", PRIVATE]  # fmt: skip
    return command + (["--effort", effort] if effort is not None else [])


def agent_command(prompt, model="claude-opus-5-5", effort=None, *, session=None, resume=None):
    """`claude -p` with its tools and no permission prompts, except reading artifacts/; user settings, hooks and
    CLAUDE.md stay off. ``prompt`` and PRIVATE follow the default system prompt, read from a file: on the command line,
    another agent's `pkill -f` on a command the prompt quotes would stop this one. ``session`` starts a session that
    is kept so ``resume`` can continue it later; with neither, nothing is kept."""
    text = f"{prompt}\n{PRIVATE}"
    system = Path(tempfile.gettempdir()) / f"roboskills-agent-{hashlib.sha256(text.encode()).hexdigest()[:16]}.md"
    system.write_text(text)
    keep = ["--session-id", session] if session else ["--resume", resume] if resume else ["--no-session-persistence"]
    command = ["claude", "-p", "--safe-mode", "--model", model, "--permission-mode", "bypassPermissions",
               "--disallowedTools", f"Read(/{ARTIFACTS}/**)", "--setting-sources", "", *keep,
               "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--append-system-prompt-file", str(system)]  # fmt: skip
    return command + (["--effort", effort] if effort is not None else [])


def codex_agent_command(model, effort=None, *, reply, images=()):
    """`codex exec` as a tool-using agent, ``agent_command``'s counterpart for a GPT model: no approvals or sandbox,
    the caller's environment, user config, rules and project docs off, ChatGPT login (the account is CODEX_HOME's).
    The prompt (instructions with PRIVATE, then the request) comes on stdin; the last reply goes to ``reply``."""
    command = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "--json", "--model", model,
               "-c", 'forced_login_method="chatgpt"', "-c", "shell_environment_policy.inherit=all",
               "-c", "project_doc_max_bytes=0"]  # fmt: skip
    if effort is not None:
        command += ["-c", "model_reasoning_effort=" + json.dumps(effort)]
    for path in images:
        command += ["--image", str(path)]
    return command + ["-o", str(reply), "-"]


def run_cli(command, stdin, directory, *, env=None, timeout=900, grace=10, cwd=None):
    """Run a model CLI in `cwd` (default `directory`, where stdout.jsonl and stderr.txt go). On a timeout or any
    interruption the CLI's whole process group is stopped and the error raised. Returns (returncode, stdout rows)."""
    directory = Path(directory)
    with (directory / "stdout.jsonl").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                                   cwd=cwd or directory, env=env, start_new_session=True)  # fmt: skip
        try:
            process.communicate(stdin, timeout=timeout)
        except BaseException:
            terminate(process, grace)
            raise
    rows = [json.loads(line) for line in (directory / "stdout.jsonl").read_text().splitlines() if line.startswith("{")]
    return process.returncode, rows


def user_message(content):
    return json.dumps(dict(type="user", message=dict(role="user", content=content))) + "\n"


def claude(system, messages, model="claude-opus-5-5", cost=None, timeout=900, effort=None):
    """`claude -p` on a conversation: each message is a `<role>` text block, its `images` [(name, base64 PNG)] image
    blocks after it. Runs in an empty directory, deleted afterwards. `LAST.usage` gets the provider's token counts and
    reported cost (absent fields stay absent); ``cost`` = {"usd": running total}, including charged failed calls."""
    content = []
    for m in messages:
        content.append(dict(type="text", text=f"<{m['role']}>\n{m['content']}\n</{m['role']}>"))
        for name, data in m.get("images", []):
            content += [
                dict(type="text", text=f"image {name}:"),
                dict(type="image", source=dict(type="base64", media_type="image/png", data=data)),
            ]
    with tempfile.TemporaryDirectory() as d:
        system_file = None
        if len(system) > LONG_SYSTEM:  # longer than one command-line argument may be
            system_file = Path(d) / "system.txt"
            system_file.write_text(system)
        command = claude_command(system, model, effort, system_file=system_file)
        returncode, rows = run_cli(command, user_message(content), d, timeout=timeout)
        stderr = (Path(d) / "stderr.txt").read_text() if (Path(d) / "stderr.txt").exists() else ""
    j = rows[-1] if rows and rows[-1].get("type") == "result" else {}
    u = j.get("usage") if isinstance(j.get("usage"), dict) else {}
    LAST.usage = {k: v for k, v in {**u, **j}.items() if k in USAGE_KEYS}
    if cost is not None:
        cost["usd"] += j.get("total_cost_usd") or 0.0
    if returncode or j.get("is_error") or "result" not in j:
        raise RuntimeError(f"claude -p failed: {(stderr or json.dumps(rows[-3:]))[-500:]}")
    return j["result"]


def codex_command(directory, model, effort, images=()):
    """`codex exec` without tools in a read-only sandbox, reading the prompt from stdin and its instructions from
    `directory`/system.txt, replying into `directory`/reply.txt: ChatGPT login only, no user config, rules, web,
    memory or session."""
    command = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--sandbox",
               "read-only", "--json", "--model", model, "-c", 'forced_login_method="chatgpt"',
               "-c", "model_reasoning_effort=" + json.dumps(effort), "-c", "project_doc_max_bytes=0",
               "-c", 'web_search="disabled"', "-c", "model_instructions_file=" + json.dumps(str(Path(directory) / "system.txt")),
               "--ephemeral"]  # fmt: skip
    for feature in ("shell_tool", "multi_agent", "hooks", "plugins", "apps", "remote_plugin", "memories", "goals",
                    "code_mode", "view_image", "image_generation", "computer_use", "browser_use", "skill_search",
                    "sleep_tool"):  # fmt: skip
        command += ["--disable", feature]
    for path in images:
        command += ["--image", str(path)]
    return command + ["-o", str(Path(directory) / "reply.txt"), "-"]


def codex(system, messages, model="gpt-6.1-sol", timeout=900, usage=None, effort="medium", *, directory=None, env=None):
    """One independent, tool-free multimodal ChatGPT call, matching ``claude``'s conversation contract.

    Image names are mapped to attached files in conversation order. Optional ``directory``
    retains inputs, raw CLI events and the reply after the call. The CLI always runs
    in a neutral temporary directory, hiding method/task identities from model context.
    Subscription admission and billing verification remain the caller's responsibility.
    """
    retained = Path(directory).resolve() if directory is not None else None
    # Codex includes its cwd in model context even with tools disabled. Keep all
    # provider-visible paths neutral, then preserve diagnostics outside that cwd.
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name).resolve()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "system.txt").write_text(system + "\n" + PRIVATE)
            content, images = [], []
            for message in messages:
                content.append(f"<{message['role']}>\n{message['content']}\n</{message['role']}>")
                for label, data in message.get("images", []):
                    path = directory / f"image_{len(images):03d}.png"
                    path.write_bytes(base64.b64decode(data, validate=True))
                    images.append(path)
                    content.append(f"Attached image {len(images)}: {label}")
            prompt = "\n".join(content)
            (directory / "request.txt").write_text(prompt)
            command = codex_command(directory, model, effort or "medium", images)
            returncode, rows = run_cli(command, prompt, directory, env=env, timeout=timeout)
            terminal = next((r for r in reversed(rows) if r.get("type") in {"turn.completed", "turn.failed"}), None)
            completed = terminal if terminal and terminal["type"] == "turn.completed" else None
            unexpected = [
                r
                for r in rows
                if r.get("type", "").startswith("item.")
                and r.get("item", {}).get("type") not in {"agent_message", "reasoning", "error"}
            ]
            failed = [r for r in rows if r.get("type") in {"error", "turn.failed"}]
            value = completed.get("usage", {}) if completed else {}
            LAST.usage = value
            if usage is not None:
                usage.clear()
                usage.update(value)
            (directory / "receipt.json").write_text(
                json.dumps(
                    dict(
                        model=model,
                        returncode=returncode,
                        cli=completed,
                        errors=failed,
                        unexpected_tools=unexpected,
                    ),
                    indent=2,
                )
                + "\n"
            )
            if returncode or not completed or unexpected or not (directory / "reply.txt").exists():
                raise RuntimeError(f"codex exec failed: {json.dumps(rows[-3:])[-500:]}")
            return (directory / "reply.txt").read_text()
        finally:
            if retained is not None:
                shutil.copytree(directory, retained, dirs_exist_ok=True)


# USD per 1M tokens (cache hit, cache miss, output), off-peak; peak (UTC weekdays 01-04, 06-10) doubles it
DEEPSEEK_PRICE = {"deepseek-flash": (0.003, 0.15, 0.60)}


def deepseek(system, messages, model="deepseek-flash", cost=None, timeout=900, effort=None, *, key=None):
    """One tool-free DeepSeek chat completion (OpenAI format), matching ``claude``'s conversation contract: one user
    message of `<role>` text parts, each message's images as base64 PNG parts after it. ``key`` defaults to
    $DEEPSEEK_API_KEY; ``effort``: none, low, high or max. `LAST.usage` gets the provider's token counts and, for a
    priced model, ``cost`` = {"usd": running total}. Of the API's models only deepseek-flash sees images;
    deepseek-v4-pro silently drops them."""
    content = []
    for m in messages:
        content.append(dict(type="text", text=f"<{m['role']}>\n{m['content']}\n</{m['role']}>"))
        for name, data in m.get("images", []):
            content += [dict(type="text", text=f"image {name}:"),
                        dict(type="image_url", image_url=dict(url="data:image/png;base64," + data))]  # fmt: skip
    body = dict(model=model, messages=[dict(role="system", content=system + "\n" + PRIVATE),
                                       dict(role="user", content=content)],
                stop=["<user>", "</assistant>"], max_tokens=16384)  # else it writes the next turn, or "</assistant>" on and on  # fmt: skip
    if effort is not None:
        body["reasoning_effort"] = effort
    request = urllib.request.Request("https://api.deepseek.com/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json",
                                      "Authorization": "Bearer " + (key or os.environ["DEEPSEEK_API_KEY"])})  # fmt: skip
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            j = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"deepseek failed: {error.code} {error.read().decode(errors='replace')[-500:]}") from error
    u = j.get("usage") or {}
    names = dict(prompt_cache_miss_tokens="input_tokens", prompt_cache_hit_tokens="cache_read_input_tokens",
                 completion_tokens="output_tokens")  # fmt: skip
    LAST.usage = {names[k]: v for k, v in u.items() if k in names} | dict(
        duration_ms=int(1000 * (time.monotonic() - started))
    )
    if model in DEEPSEEK_PRICE:
        now = time.gmtime()
        peak = now.tm_wday < 5 and (1 <= now.tm_hour < 4 or 6 <= now.tm_hour < 10)
        hit, miss, out = DEEPSEEK_PRICE[model]
        usd = (hit * u.get("prompt_cache_hit_tokens", 0) + miss * u.get("prompt_cache_miss_tokens", 0)
               + out * u.get("completion_tokens", 0)) * (2 if peak else 1) / 1e6  # fmt: skip
        LAST.usage["total_cost_usd"] = usd
        if cost is not None:
            cost["usd"] += usd
    choice = (j.get("choices") or [{}])[0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise RuntimeError(f"deepseek failed: {json.dumps(j)[-500:]}")
    # it sometimes stops after thinking alone or runs on to max_tokens: a turn, cut short, not a failed call
    return choice["message"].get("content") or ""
