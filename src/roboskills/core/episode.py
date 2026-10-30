"""One live episode per env, shared by the launchers that play them: the adapter's robots behind Tools (`make_envs`),
how an episode ends (`Episode`, `end`), the planner lock motions let go (`Lock`, `locked`, `unlocked`) and the
robot's ROBOT.md (`robot`)."""

import threading
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path


OUTCOME = ("success", "score", "reason", "steps", "task", "layout", "void")


class Lock:
    """A lock that knows which thread holds it, so `unlocked` lets go only a lock its caller holds."""

    def __init__(self):
        self.lock, self.owner = threading.Lock(), None

    def acquire(self):
        self.lock.acquire()
        self.owner = threading.get_ident()

    def release(self):
        self.owner = None
        self.lock.release()

    def mine(self):
        return self.owner == threading.get_ident()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


def locked(fn, lock):
    """`fn` holding `lock` (a resource one thread uses at a time)."""

    def call(*args, **kw):
        with lock:
            return fn(*args, **kw)

    return call


def unlocked(fn, lock):
    """`fn` with `lock` let go while it runs (a robot executing, observing or ending: no GPU work of ours). A caller
    that does not hold it (let go further up, e.g. an observation inside a motion) runs `fn` as it is."""

    def call(*args, **kw):
        if not lock.mine():
            return fn(*args, **kw)
        lock.release()
        try:
            return fn(*args, **kw)
        finally:
            lock.acquire()

    return call


def to_json(v):
    return v.tolist() if hasattr(v, "tolist") else str(v)


STATUS = (
    "with np.printoptions(precision=3, suppress=True):\n    print(bot.status())"  # after every turn: no camera, compact
)


class Episode:
    """One env's `bot` (`tools`) and its end: `finish()`, the model's or the runner's, scores it once through
    `robot.finish` (Tools never ends an episode); `result` is the native result, None until then."""

    def __init__(self, tools):
        self.tools, self.out, self.result = tools, tools.out, None

    def finish(self):
        """End and score the episode, at the end; calling it again changes nothing."""
        if self.result is None:  # robot.finish is called once: a second one would score and save the episode again
            self.result = self.tools.robot.finish(self.out)
        return "episode finished"  # the score stays with the launcher, not the model


def end(episode, void=None):
    """The runner ends an episode the model left open. `void`: why it does not count (the model could not play it:
    its LLM failed); the episode is still finished, to free the simulator."""
    if episode.result is not None:
        return {}
    msg = episode.finish()
    if void:
        episode.result["void"] = void
    return dict(finished_by="runner", msg=msg, void=void)


@contextmanager
def make_envs(mod, out, kw, plan, sam=None):
    """The adapter's robots, each behind its own Tools in out/env<i> (an `Episode` each), and the one SAM 3 they share, started after the
    robots so a simulator that fails to start never loads it. Releases SAM (unless the caller passed its own `sam`) and
    the adapter on exit, including startup failures. Motions, observations and SAM let `plan` go."""
    from roboskills.tools.robokit.bot import Tools
    from roboskills.tools.sam3.sam3 import Sam

    with ExitStack() as cleanup:
        cleanup.callback(getattr(mod, "close", lambda: None))
        robots = mod.make_vector(**{"out": str(out), **kw}) if hasattr(mod, "make_vector") else [mod.make(**kw)]
        if sam is None:
            sam = cleanup.enter_context(closing(Sam()))
            sam.detect = unlocked(locked(sam.detect, Lock()), plan)  # others plan while one segments
        bots = []
        for i, robot in enumerate(robots):
            robot.follow, robot.finish, robot.obs = (unlocked(f, plan) for f in (robot.follow, robot.finish, robot.obs))
            tools = Tools(robot, out=out / f"env{i}", sam=sam)
            tools.out.mkdir(exist_ok=True)
            bots.append(Episode(tools))
        yield bots, sam


def robot(mod):
    """The adapter's ROBOT.md (its robot, for every agent); none for an adapter under artifacts/."""
    path = Path(mod.__file__).resolve().with_name("ROBOT.md")
    return path.read_text() if path.is_file() and "artifacts" not in path.parts else ""
