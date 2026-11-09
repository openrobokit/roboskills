"""The person at the bench: a viser web panel (http://localhost:PORT) with start, execute, stop, the verdict and release.

Start lets the agent begin; Execute runs the one plan waiting for review, and only then is it enabled. The 3D view
shows the arms where they stand (grey), the fixed cameras' depth clouds and, when a plan waits for review, its arms
(orange, a slider and Play scrub along it) and TCP path: nothing moves until Execute. Rest opens the hands where they
are (what they hold drops), then takes the arms to the rig's `rest`: before Start, instead of the next plan, or before
the verdict.
The episode server's own output goes to a log, not a terminal, so the operator talks to it here.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np


VERDICTS = ("success", "failure", "void")
STRIDE = 4  # px between a depth cloud's points, both ways
PLAN_RGB, PLAN_OPACITY = (255, 140, 0), 0.45
PLAY_S = (3.0, 6.0)  # s Play takes over a plan, at least and at most, whatever the plan's own duration
CLIP = 6000  # characters of a reply or a receipt shown
TURNS_WIDTH = 480  # px of the agent turns' column


def read_turns(folder):
    """The agent's turns in `folder` (core.agent's log.jsonl, then reply.json while a turn's block runs)."""
    turns = []
    for f in ("log.jsonl", "reply.json"):
        path = Path(folder) / f
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line still being written
                continue
            if f == "log.jsonl" or not turns or row["turn"] != turns[-1].get("turn"):
                turns.append(row)
    return turns


def clip(text):
    text = text if len(text) <= CLIP else text[: CLIP // 2] + "\n...[clipped]...\n" + text[-CLIP // 2 :]
    return text + "\n```" if text.count("```") % 2 else text


def turn_text(row):
    """One turn as markdown: the model's reply (its reasoning and code), then what came back."""
    if row.get("turn") == "last":
        return f"**After the last turn**: the block still running finished\n\n```text\n{clip(row['receipt'])}\n```"
    head = f"**Turn {row['turn']}** (the model thought {row.get('llm_s', '?')} s)"
    if "receipt" not in row:
        return f"{head}, its block is running\n\n{clip(row['reply'])}"
    return f"{head}\n\n{clip(row['reply'])}\n\n**What came back**\n\n```text\n{clip(row['receipt'])}\n```"


class Operator:
    def __init__(self, port=8080, title="Real episode"):
        import viser

        self.server = viser.ViserServer(port=int(port), label=title)
        gui = self.server.gui
        self.start, self.execute, self.stop, self.release, self.judged = (threading.Event() for _ in range(5))
        self.verdict, self.note_text, self.closed = None, "", threading.Event()
        self.status = gui.add_markdown(f"**{title}**: starting")
        self.rejected, self.rest = threading.Event(), threading.Event()
        self.starter = gui.add_button("Start: the agent observes and plans", color="green")
        self.starter.on_click(lambda _: (self.start.set(), setattr(self.starter, "disabled", True)))
        gui.add_button("Stop", color="red").on_click(lambda _: self.stop.set())
        gui.add_button("Rest: open the hands, then move the arms to the rig's rest").on_click(lambda _: self.rest.set())
        with gui.add_folder("Plan review"):
            self.executer = gui.add_button("Execute: run the plan shown", color="green", disabled=True)
            self.reject = gui.add_button("Reject: nothing moves", disabled=True)
            self.scrub = gui.add_slider("Along the plan", 0.0, 1.0, 0.01, 1.0, disabled=True)
            self.play = gui.add_button("Play", disabled=True)
            self.live = gui.add_checkbox("Show the arms where they stand (grey)", True)
            self.queue = gui.add_markdown("Nothing queued")
        self.live.on_update(lambda _: self._show_live(self.live.value))
        self.executer.on_click(lambda _: self.execute.set())
        self.reject.on_click(lambda _: self.rejected.set())
        self.scrub.on_update(lambda _: self._scrub_to(self.scrub.value))
        self.play.on_click(lambda _: threading.Thread(target=self._play, daemon=True).start())
        self.meshes, self.frames, self.duration, self.marks = {}, [], 0.0, {}
        self.note = gui.add_text("Verdict note", initial_value="")
        for v in VERDICTS:
            gui.add_button(v.capitalize()).on_click(lambda _, v=v: self._judge(v))
        self.releaser = gui.add_button("Release: let the arms go where they are (support them)", disabled=True)
        self.releaser.on_click(lambda _: self.release.set())

    def watch(self, folder):
        """Show the agent's turns from `folder` (`read_turns`), the latest one unless an earlier one is picked."""
        gui, turns, seen = self.server.gui, [], None
        side = gui.add_panel()  # its own column on the left, beside the 3D view; the controls stay on the right
        with side.add_tab("Agent turns"):
            latest = gui.add_checkbox("Follow the latest turn", True)
            pick = gui.add_slider("Turn", 0, 1, 1, 0)
            shown = gui.add_markdown("No turn yet")
        side.dock_left()
        side.set_width(TURNS_WIDTH)

        def show(_=None):
            if turns:
                i = min(int(pick.value), len(turns) - 1)
                if i != len(turns) - 1:
                    latest.value = False
                shown.content = turn_text(turns[i])

        def poll():
            nonlocal seen
            while not self.closed.wait(1.0):
                files = [Path(folder) / f for f in ("log.jsonl", "reply.json")]
                key = [(f.stat().st_mtime_ns, f.stat().st_size) if f.exists() else None for f in files]
                if key != seen:
                    seen, turns[:] = key, read_turns(folder)
                    if turns:
                        pick.max = max(1, len(turns) - 1)
                        if latest.value:
                            pick.value = len(turns) - 1
                        show()

        pick.on_update(show)
        latest.on_update(lambda _: latest.value and turns and setattr(pick, "value", len(turns) - 1))
        threading.Thread(target=poll, daemon=True, name="turns").start()

    def _judge(self, verdict):
        self.verdict, self.note_text = verdict, self.note.value
        self.judged.set()
        self.stop.set()  # a verdict given mid-run ends the episode there, as Stop does

    def say(self, text):
        self.status.content = text
        print(f"[operator] {text}", flush=True)

    def wait(self, event, text, timeout=None):
        """Show `text` and wait for `event`; False on `timeout` s."""
        self.say(text)
        return event.wait(timeout)

    def released(self, text):
        """Show `text` and wait for Release, enabled only now: a press before this moment never counts."""
        self.release.clear()
        self.releaser.disabled = False
        try:
            self.wait(self.release, text)
        finally:
            self.releaser.disabled = True

    def show_path(self, name, points):
        """Draw a planned TCP path (N, 3) in the panel's 3D view."""
        if len(points) > 1:
            self.server.scene.add_spline_catmull_rom(
                f"/paths/{name}", points, thickness=3.0, thickness_units="screen", color=(255, 120, 0)
            )

    def add_arm(self, name, meshes):
        """Two copies of arm `name`'s `meshes` {geom: (V, F, rgb)}: /live (where it stands) and /plan (hidden)."""
        for layer in ("live", "plan"):
            for g, (V, F, rgb) in meshes.items():
                plan = layer == "plan"
                self.meshes[layer, name, g] = self.server.scene.add_mesh_simple(
                    f"/{layer}/{name}/g{g}", V, F, color=PLAN_RGB if plan else rgb,
                    opacity=PLAN_OPACITY if plan else None, visible=not plan,
                )  # fmt: skip

    def pose_arms(self, layer, poses, visible=True):
        """Set the `layer` arms to `poses` {arm: {geom: (4, 4) world}}."""
        import viser.transforms as vtf

        for (lay, a, g), h in self.meshes.items():
            if lay == layer and g in poses.get(a, {}):
                T = poses[a][g]
                h.wxyz, h.position = vtf.SO3.from_matrix(T[:3, :3]).wxyz, T[:3, 3]
                h.visible = visible and (layer != "live" or self.live.value)

    def show_cloud(self, name, rgb, depth, K, T_world_cam):
        """A camera's depth as coloured points in the world."""
        v, u = np.nonzero((depth > 0) & (depth < 2.5))
        keep = (v % STRIDE == 0) & (u % STRIDE == 0)
        v, u = v[keep], u[keep]
        z = depth[v, u]
        C = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
        P = C @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        self.server.scene.add_point_cloud(f"/clouds/{name}", P, rgb[v, u], point_size=0.004)

    def _scrub_to(self, x):
        if self.frames:
            self.pose_arms("plan", self.frames[int(round(x * (len(self.frames) - 1)))])

    def _play(self):
        n = len(self.frames)
        for i in range(n):
            if not self.frames or self.execute.is_set() or self.rejected.is_set():
                return
            self.scrub.value = i / max(1, n - 1)  # its on_update poses the arms
            time.sleep(min(max(self.duration, PLAY_S[0]), PLAY_S[1]) / max(1, n))

    def _show_live(self, visible):
        for (lay, _, _), h in self.meshes.items():
            if lay == "live":
                h.visible = visible

    def first(self, text, answers):
        """Show `text`, wait for the first of `answers` {name: event}, clear it and return its name."""
        self.say(text)
        while True:
            for name, event in answers.items():
                if event.wait(0.05):
                    if name != "stop":
                        event.clear()
                    return name

    def show_queue(self, text):
        """The queued plans and their state, under the plan review."""
        self.queue.content = text

    def review(self, text, frames, duration, cancel=None):
        """Show a plan (`frames` [{arm: {geom: 4x4}}] along it, `duration` s) and wait: "approve" (Execute),
        "reject", "rest", "stop" or, once `cancel` (an Event) is set, "cancel": the plan was dropped meanwhile."""
        self.frames, self.duration = frames, duration
        self.execute.clear(), self.rejected.clear()  # not rest: one pressed while the agent thought counts here
        for h in (self.executer, self.reject, self.scrub, self.play):
            h.disabled = False
        self.scrub.value = 1.0
        self._scrub_to(1.0)
        try:
            answers = dict(approve=self.execute, reject=self.rejected, rest=self.rest, stop=self.stop)
            if cancel is not None:
                answers["cancel"] = cancel
            return self.first(text, answers)
        finally:
            for h in (self.executer, self.reject, self.scrub, self.play):
                h.disabled = True
            self.frames = []
            for (lay, _, _), h in self.meshes.items():
                if lay == "plan":
                    h.visible = False

    def close(self):
        self.closed.set()
        self.server.stop()
