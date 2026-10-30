"""Agents that act by writing Python: the turn loop (`run`), where the model writes one ```python block per turn
against the tools in a `Repl`'s persistent namespace and gets back a receipt (printed output, error, status, images);
every turn is logged to log.jsonl.
"""

import base64
import inspect
import io
import itertools
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from roboskills.core.llm import LAST, claude


CODE = re.compile(r"```python\n(.*?)```", re.S)


class ThreadOut(io.TextIOBase):
    """sys.stdout that sends each thread's prints to that thread's buffer while it runs a turn (several episodes
    run turns at once in one process; redirect_stdout would swap one process-wide stdout under them)."""

    def __init__(self, real):
        self.real, self.local = real, threading.local()

    def write(self, s):
        return (getattr(self.local, "buf", None) or self.real).write(s)

    def flush(self):
        (getattr(self.local, "buf", None) or self.real).flush()


def execute(code, ns, limit=4000):
    """Run code in namespace `ns`; returns its printed output plus the traceback tail on error (clipped)."""
    if not isinstance(sys.stdout, ThreadOut):
        sys.stdout = ThreadOut(sys.stdout)
    sys.stdout.local.buf = buf = io.StringIO()
    try:
        exec(compile(code, "<turn>", "exec"), ns)
    except Exception:  # noqa: BLE001  (the model reads the error and corrects itself)
        print("ERROR:\n" + "".join(traceback.format_exc().splitlines(True)[-6:]))
    finally:
        sys.stdout.local.buf = None
    out = buf.getvalue()
    return out if len(out) <= limit else out[: limit // 2] + "\n...[clipped]...\n" + out[-limit // 2 :]


def append_only(messages, block=20):
    """Prompt view that changes no message once sent, except to drop images: the whole log, with images dropped oldest
    first in whole blocks of `block` once there are two (a request takes about 32 MB; a 640x480 PNG is 0.5 MB). Each
    call reuses the prompt cache up to its newest turn and only a dropped block rewrites it (a windowed view would edit
    earlier turns every call, so only the system prompt and the task are reused); the text grows with every turn."""
    drop, out = max(0, sum(len(m.get("images", [])) for m in messages) // block - 1) * block, []
    for m in messages:
        k = min(drop, len(m.get("images", [])))
        out.append(dict(m, images=m["images"][k:]) if k else m)
        drop -= k
    return out


def png64(rgb):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode()


class Repl:
    """The tools in a persistent namespace, in this process; any object with the same `exec` and `api` can stand in.
    `episode`: what ends the episode, when not the tools themselves: an object with `finish()`, which the namespace
    offers as `finish`, and `result` (None until it ran). `show(images)` puts camera images in the turn's receipt."""

    def __init__(self, tools, episode=None):
        self.tools, self.episode, self.images = tools, episode or tools, []
        self.ns = {"bot": tools, "np": __import__("numpy"), "show": self.show}
        if episode is not None:
            self.ns["finish"] = episode.finish

    def show(self, images):
        """Show camera images with this turn's output: a View (from bot.observe) or {name: (H, W, 3) uint8}."""
        self.images += [(n, getattr(c, "rgb", c)) for n, c in getattr(images, "cams", images).items()]

    def exec(self, code):
        """(printed output, episode finished, images [(name, base64 PNG)] the code showed)"""
        out = execute(code, self.ns)
        shown, self.images = [(n, png64(rgb)) for n, rgb in self.images], []
        return out, self.episode.result is not None, shown

    def api(self):
        t, names = self.tools, {k: v for k, v in self.ns.items() if k in ("finish", "show")}
        return "\n".join([f"- bot.{n}{inspect.signature(getattr(t, n))}: {getattr(t, n).__doc__}" for n in dir(t)
                           if not n.startswith("_") and callable(getattr(t, n))]
                          + [f"- {n}{inspect.signature(f)}: {f.__doc__}" for n, f in names.items()])  # fmt: skip


def run(repl, instruction, out, llm=claude, max_turns=None, max_bad=None, context="", *, prompt,
        initial_code="print(bot.status())", status_code="print(bot.status())"):  # fmt: skip
    """Drive one episode through `repl`. Returns (finished, turns); every turn is logged to out/log.jsonl, with the
    LLM call's seconds and (from `claude`) its usage. `prompt`: the system prompt, with `{api}` where the tools' API
    listing goes (e.g. roboskills/workflows/agent_as_policy/SKILL.md).
    `context`: text given with the instruction: the robot's ROBOT.md from its eval runtime, or what a launcher itself
    produced (frozen research); never a file from artifacts/, which agents do not see. No limit by default
    on turns (`max_turns`) or consecutive bad turns (`max_bad`): the episode ends when the model calls finish.
    `initial_code`: public observations collected before the first model request, including images.
    `status_code`: what is run after every turn for the receipt's status.
    The LLM sees the log through `append_only`.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    system = prompt.format(api=repl.api())
    notes = f"\nNotes on this task:\n{context.strip()}\n\n" if context.strip() else ""
    initial, ended, images = repl.exec(initial_code)
    messages = [dict(role="user", content=f"Task: {instruction}\n{notes}" + initial, images=images)]
    if ended:
        return True, 0
    bad, done, turns = 0, False, 0
    with open(out / "log.jsonl", "a") as log:
        for turn in range(max_turns) if max_turns else itertools.count():
            LAST.usage, t = None, time.monotonic()
            turns += 1
            reply = llm(system, append_only(messages))
            llm_s = round(time.monotonic() - t, 1)
            messages.append(dict(role="assistant", content=reply))
            m = CODE.search(reply)
            receipt, done, imgs = repl.exec(m.group(1)) if m else ("No ```python block found. Reply with exactly one.", False, [])  # fmt: skip
            bad = bad + 1 if not m or receipt.startswith("ERROR") else 0
            if not done:
                receipt += "[status] " + repl.exec(status_code)[0]
            messages.append(dict(role="user", content=receipt, images=imgs))
            row = dict(turn=turn, reply=reply, receipt=receipt, images=[n for n, _ in imgs], llm_s=llm_s)
            log.write(json.dumps(dict(row, usage=LAST.usage) if LAST.usage else row) + "\n")
            log.flush()
            if done or (max_bad and bad >= max_bad):
                break
    return done, turns
