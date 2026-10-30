"""The model CLI wrapper: usage and cost survive failed calls; absent usage stays absent."""

from pathlib import Path

import pytest

from roboskills.core.llm import LAST, claude


def test_codex_runs_without_tools_or_session(tmp_path):
    from roboskills.core import llm

    command = llm.codex_command(tmp_path, "model", "medium", [tmp_path / "a.png"])
    assert command[command.index("--sandbox") + 1] == "read-only" and "--ephemeral" in command
    assert {"shell_tool", "view_image"} <= {command[i + 1] for i, c in enumerate(command) if c == "--disable"}
    assert command[command.index("--image") + 1] == str(tmp_path / "a.png") and command[-3:] == [
        "-o",
        str(tmp_path / "reply.txt"),
        "-",
    ]


def test_failed_provider_call_retains_charged_usage(monkeypatch):
    result = dict(
        type="result",
        is_error=True,
        result="backend error",
        usage=dict(input_tokens=17, output_tokens=9),
        total_cost_usd=0.12,
        modelUsage={"example": dict(inputTokens=17, outputTokens=9)},
    )
    monkeypatch.setattr("roboskills.core.llm.run_cli", lambda *a, **k: (1, [result]))
    cost = {"usd": 1.0}
    with pytest.raises(RuntimeError):
        claude("system", [], cost=cost)
    assert LAST.usage == dict(input_tokens=17, output_tokens=9, total_cost_usd=0.12)
    assert cost["usd"] == pytest.approx(1.12)


def test_missing_usage_is_not_zero_tokens(monkeypatch):
    result = dict(type="result", result="ok")
    monkeypatch.setattr("roboskills.core.llm.run_cli", lambda *a, **k: (0, [result]))
    assert claude("system", [], cost={"usd": 0.0}) == "ok"
    assert LAST.usage == {}


def test_agent_calls_keep_the_plain_invocation(monkeypatch):
    from roboskills.core import llm

    seen = []
    monkeypatch.setattr(
        llm, "run_cli", lambda command, *a, **k: seen.append(command) or (0, [dict(type="result", result="ok")])
    )
    assert llm.claude("system", [dict(role="user", content="hi")]) == "ok"
    assert "--safe-mode" not in seen[0] and seen[0][seen[0].index("--system-prompt") + 1] == "system"


def test_long_system_prompt_goes_through_a_file(monkeypatch):
    from roboskills.core import llm

    seen = []

    def run(command, *a, **k):
        path = command[command.index("--system-prompt-file") + 1]
        seen.append(Path(path).read_text())
        return 0, [dict(type="result", result="ok")]

    monkeypatch.setattr(llm, "run_cli", run)
    system = "docs " * llm.LONG_SYSTEM
    assert llm.claude(system, [dict(role="user", content="hi")]) == "ok" and seen == [system]


def test_codex_preserves_named_image_order_private_instructions_and_usage(tmp_path, monkeypatch):
    import base64

    from roboskills.core import llm

    def run(command, prompt, directory, **kwargs):
        assert (directory / "system.txt").read_text() == "system\n" + llm.PRIVATE
        assert prompt.index("first view") < prompt.index("second view")
        attached = [command[i + 1] for i, x in enumerate(command) if x == "--image"]
        assert [Path(p).read_bytes() for p in attached] == [b"first", b"second"]
        assert "shell_tool" in command and "--ephemeral" in command
        (directory / "reply.txt").write_text("answer")
        return 0, [dict(type="turn.completed", usage=dict(input_tokens=7, output_tokens=3))]

    monkeypatch.setattr(llm, "run_cli", run)
    usage = {"old": True}
    message = dict(
        role="user",
        content="compare",
        images=[
            ("first view", base64.b64encode(b"first").decode()),
            ("second view", base64.b64encode(b"second").decode()),
        ],
    )
    assert llm.codex("system", [message], directory=tmp_path, usage=usage) == "answer"
    assert usage == dict(input_tokens=7, output_tokens=3)


@pytest.mark.parametrize("failure", ["tool", "missing_turn", "nonzero", "failed_turn"])
def test_codex_rejects_incomplete_failed_or_tool_using_calls(tmp_path, monkeypatch, failure):
    from roboskills.core import llm

    def run(command, prompt, directory, **kwargs):
        (directory / "reply.txt").write_text("untrusted")
        rows = [] if failure == "missing_turn" else [dict(type="turn.completed")]
        if failure == "tool":
            rows.append(dict(type="item.completed", item=dict(type="command_execution")))
        if failure == "failed_turn":
            rows.append(dict(type="turn.failed", error={"message": "denied"}))
        return int(failure == "nonzero"), rows

    monkeypatch.setattr(llm, "run_cli", run)
    with pytest.raises(RuntimeError, match="codex exec failed"):
        llm.codex("system", [], directory=tmp_path)


def test_codex_accepts_a_completed_reply_after_transport_recovery(tmp_path, monkeypatch):
    import json

    from roboskills.core import llm

    diagnostic = dict(type="error", message="Reconnecting... stream disconnected before completion")

    def run(command, prompt, directory, **kwargs):
        (directory / "reply.txt").write_text("recovered answer")
        return 0, [diagnostic, dict(type="turn.completed", usage=dict(output_tokens=9))]

    monkeypatch.setattr(llm, "run_cli", run)
    assert llm.codex("system", [], directory=tmp_path) == "recovered answer"
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["errors"] == [diagnostic] and receipt["cli"]["type"] == "turn.completed"


def test_codex_hides_method_paths_and_preserves_failure_logs(tmp_path, monkeypatch):
    import subprocess

    from roboskills.core import llm

    retained = tmp_path / "method_identity_marker" / "calls"

    def run(command, prompt, directory, **kwargs):
        assert directory != retained and not directory.is_relative_to(tmp_path)
        assert "method_identity_marker" not in str(command) + prompt + str(directory)
        (directory / "stderr.txt").write_text("retained failure evidence")
        raise subprocess.TimeoutExpired("codex", 1)

    monkeypatch.setattr(llm, "run_cli", run)
    with pytest.raises(subprocess.TimeoutExpired):
        llm.codex("system", [], directory=retained)
    assert (retained / "stderr.txt").read_text() == "retained failure evidence"
    assert llm.PRIVATE in (retained / "system.txt").read_text()


def test_deepseek_sends_images_private_instructions_and_logs_usage(monkeypatch):
    import io
    import json

    from roboskills.core import llm

    seen = []

    def urlopen(request, timeout):
        seen.append(json.loads(request.data))
        reply = dict(choices=[dict(finish_reason="stop", message=dict(content="ok"))],
                     usage=dict(prompt_cache_miss_tokens=5, prompt_cache_hit_tokens=3, completion_tokens=2))  # fmt: skip
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    message = dict(role="user", content="hi", images=[("front", "QUJD")])
    cost = {"usd": 0.0}
    assert llm.deepseek("system", [message], cost=cost, key="k", effort="low") == "ok"
    assert cost["usd"] in (pytest.approx(1.959e-6), pytest.approx(3.918e-6))  # off-peak or peak
    system, user = seen[0]["messages"]
    assert system["content"].endswith(llm.PRIVATE) and seen[0]["reasoning_effort"] == "low"
    assert seen[0]["stop"] == ["<user>", "</assistant>"] and seen[0]["max_tokens"] == 16384
    assert user["content"][-1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert {k: LAST.usage[k] for k in ("input_tokens", "cache_read_input_tokens", "output_tokens")} == dict(
        input_tokens=5, cache_read_input_tokens=3, output_tokens=2
    )
