"""Every safety check of a real bench, for any arm and hand, on RoboKit's model of its arms (the planner's): the bench's
driver is the only code that sends to the motors, and everything that reaches it passes here first.

Safety, in three layers, all ending in a hold (never a dropped arm): `Limits.check` refuses a whole path before it
moves, `Limits.tracking` stops a motion that leaves its targets, and the caller holds the arm wherever it is when a
motion ends early.

A driver (one arm and its hand: `eval/yam/arm.py`) has `n` (its arm joints), `target` (its
last command: n arm joints, then the hand command), `read()` (the same, measured), `send(q, u, contact=False)`,
`hold(u=None)` (hold where it is measured now), `alive()`, `holds()` (its hand holds something) and `close()`.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

from roboskills.eval.interface import Trajectory


class Body:
    """One arm of a RoboKit `RobotModel` (tools/robokit/robot_model.py, the model the planner plans on), for the checks:
    its link poses and surface points in the world, from RoboKit's forward kinematics on the CPU. The hand is the arm's
    `ee_link` and every link hanging off it; `skip`: links that cannot reach the table or another arm, never checked.
    Surface points are about `hand_spacing` m apart on the hand, `arm_spacing` m on the rest. Thread-safe: the queue's
    executor and reviewer and the agent's calls share the model, so each pose is computed under `lock`."""

    def __init__(self, kin, arm, skip=(), hand_spacing=0.01, arm_spacing=0.02):
        self.kin, self.arm, self.i = kin, arm, kin.info[arm]
        self.hand_q = kin.robot.arms[arm].hand.model_q
        names, hand = self.i.M.rk.link_names, {int(k) for k in self.i.hand_links}
        self.hand_links = [k for k in range(len(names)) if k in hand and names[k] not in skip]
        self.arm_links = [k for k in range(len(names)) if k not in hand and names[k] not in skip]
        self.spacing = hand_spacing, arm_spacing
        lim = np.asarray(self.i.M.rk.spec.actuated_joint_limits, float)[self.i.idx]
        self.lo, self.hi = lim[:, 0], lim[:, 1]  # the model's joint limits, in command order
        self.lock = threading.RLock()

    @property
    def T(self):
        return np.asarray(self.kin.robot.arms[self.arm].T_world_base, float)

    def _fq(self, q, u=None):
        f = np.zeros(len(self.i.M.names), np.float32)
        f[self.i.idx] = q
        if u is not None and self.i.hidx:
            f[self.i.hidx] = self.hand_q(u) if self.hand_q else np.asarray(u, float)
        return f

    def ee_pose(self, q):
        """(4, 4) the `ee_link` in the world at joints `q`."""
        with self.lock:
            return self.kin.links(self.arm, self._fq(q))[self.i.ee]

    def surface(self, q, u=None):
        """(points (N, 3) in the world, hand (N,) bool: on the hand) at joints `q`, hand command `u`."""
        f = self._fq(q, u)
        with self.lock:
            H = self.kin.points(self.arm, f, self.hand_links, self.spacing[0])
            A = self.kin.points(self.arm, f, self.arm_links, self.spacing[1])
        return np.concatenate([A, H]), np.r_[np.zeros(len(A), bool), np.ones(len(H), bool)]

    def meshes(self):
        """{link: (vertices, faces, rgb 0..255)}: every link's visual mesh in its own frame (the operator's 3D view)."""
        spec, out = self.i.M.rk.spec, {}
        for k, name in enumerate(self.i.M.rk.link_names):
            g = spec.link_visual_geometries[name]
            if g.geometry:
                mesh = g.to_mesh()
                rgb = tuple(int(c) for c in getattr(mesh.visual, "main_color", (160, 160, 160))[:3])
                out[k] = (np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int), rgb)
        return out

    def poses(self, q, u=None):
        """{link: (4, 4) in the world} for every link at joints `q`, hand command `u`."""
        with self.lock:
            return dict(enumerate(self.kin.links(self.arm, self._fq(q, u))))


@dataclass
class Limits:
    """The rig's safety envelope, checked on every path before it moves (all metres and radians)."""

    lo: dict  # {arm: (n,)} joint limits
    hi: dict
    vmax: float  # rad/s: faster paths are slowed down (retimed), never refused for speed alone
    table_z: float
    box_lo: np.ndarray  # (3,) the TCP stays inside this world box
    box_hi: np.ndarray
    clearance: float = 0.005  # m every checked point stays above the table
    contact_depth: float = 0.01  # m the hand may press into the table on a contact waypoint
    separation: float = 0.03  # m between two arms' surfaces
    start_tol: float = 0.1  # rad the path's first waypoint may lie from the measured joints
    track: float = 0.15  # rad a joint may lag its target before the motion stops
    track_contact: float = 0.35  # ... on a contact waypoint (it may stop on contact by design)

    @classmethod
    def from_rig(cls, rig, lo, hi, vmax):
        """The rig's envelope: each arm's joint limits `lo`, `hi` {arm: (n,)}, its table and workspace box, and any of
        the margins above the rig sets under `limits`."""
        table_z, box = float(rig.get("table_z", 0.0)), rig.get("workspace", {})
        return cls(lo, hi, float(vmax), table_z, np.asarray(box.get("lo", [-1.5, -1.5, table_z - 0.02]), float),
                   np.asarray(box.get("hi", [1.5, 1.5, table_z + 1.0]), float),
                   **{k: float(v) for k, v in (rig.get("limits") or {}).items()})  # fmt: skip

    def retime(self, path):
        """`path.t` stretched so no joint moves faster than vmax (as planned when it already is)."""
        t = np.r_[0.0, np.asarray(path.t, float)]
        scale = 1.0
        for Q in path.q.values():
            Q = np.asarray(Q, float)
            speed = np.abs(np.diff(Q, axis=0)).max(1) / np.diff(t[1:]) if len(Q) > 1 else np.zeros(0)
            scale = max(scale, float(speed.max(initial=0.0)) / self.vmax)
        return np.asarray(path.t, float) * scale

    def check(self, path, q_now, u_now, bodies, tcp):
        """Why `path` must not run (None: it may), from the measured joints `q_now` {arm: (n,)} and hand `u_now`
        {arm: (m,)}; `bodies` {arm: Body}, `tcp` {arm: (4, 4) TCP in its `ee_link`}."""
        contact = np.zeros(len(path.t), bool) if path.contact is None else np.asarray(path.contact, bool)
        for a, Q in path.q.items():
            Q = np.asarray(Q, float)
            if np.abs(Q[0] - q_now[a]).max() > self.start_tol:
                return f"the {a} arm's path starts {np.abs(Q[0] - q_now[a]).max():.2f} rad from where it stands: plan again"
            lo, hi = (
                np.minimum(self.lo[a], q_now[a]),
                np.maximum(self.hi[a], q_now[a]),
            )  # from a rest on a limit: inward
            bad = np.argwhere((Q < lo) | (Q > hi))
            if len(bad):
                i, j = bad[0]
                return f"{a} joint{j + 1} at waypoint {i} is {Q[i, j]:.2f} rad, outside [{lo[j]:.2f}, {hi[j]:.2f}]"
        if not path.q:
            return None
        arms = list(bodies)
        # from under the table limit (a hand that sank onto the table): never lower than it stands, so it can rise out
        start = {a: float(bodies[a].surface(q_now[a], u_now[a])[0][:, 2].min()) - 0.001 for a in path.q}
        for i in range(len(path.t)):
            pts = {}
            for a in arms:
                q = np.asarray(path.q[a][i], float) if a in path.q else q_now[a]
                u = np.asarray(path.hand[a][min(i, len(path.hand[a]) - 1)], float) if a in path.hand else u_now[a]
                pts[a] = bodies[a].surface(q, u)
                if a not in path.q:
                    continue
                T = bodies[a].ee_pose(q) @ tcp[a]
                p = T[:3, 3]
                if (p < self.box_lo).any() or (p > self.box_hi).any():
                    return f"the {a} TCP at waypoint {i} is at {np.round(p, 3).tolist()}, outside the workspace box"
                P, hand = pts[a]
                floor = self.table_z + self.clearance
                low = np.minimum(np.where(hand & contact[i], self.table_z - self.contact_depth, floor), start[a])
                if (P[:, 2] < low).any():
                    k = int(np.argmax(low - P[:, 2]))
                    what = "hand" if hand[k] else "arm"
                    return (
                        f"the {a} {what} at waypoint {i} reaches z = {P[k, 2]:.3f}, below the table limit {low[k]:.3f}"
                    )
            if len(arms) == 2:
                from scipy.spatial import cKDTree

                d, _ = cKDTree(pts[arms[0]][0]).query(pts[arms[1]][0], k=1)
                if d.min() < self.separation:
                    return f"the arms come {d.min() * 100:.1f} cm apart at waypoint {i} (limit {self.separation * 100:.0f} cm)"
        return None

    def tracking(self, q_cmd, q_meas, contact):
        """Why a motion must stop now (None: it goes on): a joint lagging its target (a collision, a blocked arm)."""
        lag = np.abs(np.asarray(q_meas, float) - np.asarray(q_cmd, float))
        limit = self.track_contact if contact else self.track
        if lag.max() > limit:
            return f"joint{int(np.argmax(lag)) + 1} lags its target by {lag.max():.2f} rad (limit {limit:.2f})"
        return None


def run_path(hw, limits, bodies, tcp, path, dt):
    """Run `path` on the arms `hw` {arm: driver} at one command per tick `dt` if it passes `limits.check` whole (slowed
    to vmax); a joint leaving its target or a stopped driver ends it in a hold, as does any exception (re-raised,
    Ctrl+C included). Returns why it did not run to the end (None: it did)."""
    now = {a: h.read() for a, h in hw.items()}
    why = limits.check(path, {a: v[: hw[a].n] for a, v in now.items()}, {a: v[hw[a].n :] for a, v in now.items()},
                       bodies, tcp)  # fmt: skip
    if why:
        return why
    t = np.r_[0.0, limits.retime(path)]
    rows = {a: np.vstack([hw[a].target[: hw[a].n], np.asarray(Q, float)]) for a, Q in path.q.items()}
    hands = {a: np.vstack([hw[a].target[hw[a].n :], np.asarray(U, float)]) for a, U in path.hand.items() if a in hw}
    t0 = time.monotonic()
    try:
        for k in range(1, int(np.ceil(t[-1] / dt - 1e-9)) + 1):
            s = min(k * dt, t[-1])
            for a, h in hw.items():
                q = interp(s, t, rows[a]) if a in rows else h.target[: h.n]
                h.send(q, interp(s, t, hands[a]) if a in hands else h.target[h.n :])
            time.sleep(max(0.0, t0 + k * dt - time.monotonic()))
            for a in rows:
                lag = limits.tracking(hw[a].target[: hw[a].n], hw[a].read()[: hw[a].n], contact=False)
                why = f"{a}: {lag}" if lag else None if hw[a].alive() else f"{a}: the driver stopped"
                if why:
                    for h in hw.values():
                        h.hold()
                    return why
    except BaseException:
        for h in hw.values():
            h.hold()
        raise
    return None


def interp(s, t, rows):
    """Each column of `rows` (len(t), k) at time `s`."""
    return np.array([np.interp(s, t, rows[:, j]) for j in range(rows.shape[1])])


def opening(hw, opened, speed, dt):
    """A path that opens each hand to `opened` {arm: hand command} at `speed` command units per s, each arm held at its
    target (None: every hand is open already)."""
    u0 = {a: hw[a].target[hw[a].n :] for a in opened}
    far = max(float(np.abs(np.asarray(opened[a], float) - u0[a]).max()) for a in opened)
    if far < 1e-6:
        return None
    n = max(2, int(np.ceil(far / speed / dt)))
    s = np.linspace(0, 1, n + 1)[1:, None]
    return Trajectory(t=s[:, 0] * n * dt, q={}, hand={a: u0[a] + s * (np.asarray(opened[a], float) - u0[a])
                                                     for a in opened}, T={})  # fmt: skip


def park(hw, limits, bodies, tcp, rest, home, dt, opened=None, hand_speed=0.5):
    """Hold, then take the arms to `rest` {arm: (n,)} on checked straight paths at half speed, directly or through
    `home`; returns why they are not there (None: parked). `opened` {arm: hand command}: first open the hands there at
    `hand_speed` per s, the arms held (what they hold drops). Ctrl+C or SIGTERM during it leaves them held."""

    def go(targets):  # a straight joint line there at half speed, a waypoint per tick
        now = {a: h.read()[: h.n] for a, h in hw.items()}
        n = max(
            2,
            int(np.ceil(max(np.abs(np.asarray(targets[a]) - now[a]).max() for a in targets) / (limits.vmax / 2) / dt)),
        )
        s = np.linspace(0, 1, n + 1)[1:, None]
        line = Trajectory(t=s[:, 0] * n * dt, q={a: now[a] + s * (np.asarray(targets[a]) - now[a]) for a in targets},
                          hand={}, T={})  # fmt: skip
        return run_path(hw, limits, bodies, tcp, line, dt)

    try:
        for h in hw.values():
            h.hold()
        path = opening(hw, opened, hand_speed, dt) if opened else None
        why = run_path(hw, limits, bodies, tcp, path, dt) if path else None
        if why:
            return f"opening the hands: {why}"
        if go(rest) is None:
            return None
        return go(home) or go(rest)  # rest refused straight: through home
    except KeyboardInterrupt:
        for h in hw.values():
            h.hold()
        return "interrupted while parking"
