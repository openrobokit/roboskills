"""The contract between a benchmark adapter and the shared tools in `roboskills.tools.robokit`.

An adapter is one class implementing `Robot`, plus a factory
`make(**kw) -> Robot` that the launcher calls with the `key=value` pairs after `--` on its command line.
Everything the skills know about a benchmark comes through here; nothing else touches the env.

The simulator never runs in the launcher's process: `make` starts it as a server and returns the host's end, speaking
`roboskills.eval.wire`. An adapter whose simulator runs in Python of its own keeps its Robot class and serves it with
`roboskills.eval.serve`; the Isaac Sim benchmarks (RoboDojo, BEHAVIOR, EmbodiedSWE, HumanoidArena) serve native
actions from their own `sim/server.py`. Either runs natively or in the benchmark's image (`roboskills.eval.runtime`),
and `roboskills.eval.check` validates a runtime without a model.

Frames and units: metres, radians, the benchmark's world frame with +z up (the table is below the
arms). Poses are 4x4 homogeneous matrices.

A mobile robot has `Robot.base` (a `Base`): its arms ride on a holonomic base that drives on the floor plane. `Obs.base`
is the base's measured [x, y, yaw] in the world, `Trajectory.base` drives it, and every `Arm.T_world_base` moves with it
(the adapter keeps them current: the skills read them again after every motion). A fixed robot has no `base`.
A legged robot walks as its `base` (the adapter turns base waypoints into steps) and may also have `Robot.body` (a
`Body`): the height its pelvis stands, crouches or sits at, measured in `Obs.body` and moved by `Trajectory.body`.

The TCP (tool centre point) frame is the same for every hand, so verbs mean the same thing on every robot:
  origin  where a held object's centre sits: between the jaw pads for a parallel gripper; a few cm in
          front of the palm, among the fingers, for a dexterous hand
  +x      approach: the direction the hand moves to reach into an object (out of the palm / along the jaws)
  +y      squeeze axis: the jaws close along it; for a dexterous hand, the thumb-to-fingers direction the
          fingers close along (the palm's finger direction)
  +z      x cross y; a bar held in the hand lies along z
The adapter gives it as `Hand.T_ee_tcp` (the TCP in the arm model's `ee_link` frame).

`roboskills.workflows.agent_as_policy.run` loads the adapter. ROBOT.md states the arms' names
and poses, hand kinds and presets, cameras, control rate, and step-budget units.

Hand commands `u` are vectors: a parallel gripper has one value in [0, 1] (0 closed, 1 open to
`Hand.width` metres between the pads); a dexterous hand has one joint target per actuated finger joint,
in the benchmark's action order. `Obs.hand` should be the measured hand state, not the last command, where
the simulator gives it: fingers that stop short of a closing command tell the skills something is held.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class Camera:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32 metres along the optical axis; 0, nan or inf where nothing
    #                                     is seen. None, and K, T_world_cam None: the env gives RGB only (the adapter
    #                                     estimates them, or the skills cannot use this camera in 3D)
    K: Optional[np.ndarray] = None  # (3, 3) pinhole intrinsics
    T_world_cam: Optional[np.ndarray] = None  # (4, 4) OpenCV optical frame (x right, y down, z forward) in the world


@dataclass
class Hand:
    kind: Literal["jaw", "dex"]  # parallel gripper or multi-finger hand
    T_ee_tcp: np.ndarray  # (4, 4) the TCP (see above) in the arm's `ee_link` frame
    open: np.ndarray  # command: fully open (jaw: [1.0]; dex: fingers straight/spread, ready to grasp)
    closed: np.ndarray  # command: fully closed (jaw: [0.0]; dex: a fist)
    joints: list = field(default_factory=list)  # hand joint names in the arm model (FK of fingers)
    model_q: Optional[Callable] = None  # command u -> values of `joints` in the model (None: u itself)
    width: float = 0.0  # jaw: distance between the pads at command 1.0, m (command = width / this)
    presets: dict = field(default_factory=dict)  # more named commands, e.g. dex "point" (index out), "flat"
    press: str = "closed"  # the preset (or "closed") to press/push with
    press_point: np.ndarray = field(default_factory=lambda: np.zeros(3))  # contact point in the TCP frame then


@dataclass
class Arm:
    name: str  # "left" / "right"
    model: object  # RoboKit-loadable description (URDF/MJCF path, or a yourdfpy.URDF) holding this arm
    joints: list  # this arm's joint names in the model, in command order
    ee_link: str  # model link the hand hangs off (Hand.T_ee_tcp is relative to it)
    T_world_base: np.ndarray  # (4, 4) the model's root link in the world, now (it moves with a mobile base and torso)
    lo: np.ndarray  # joint limits used for IK and planning (may be tighter than the model's)
    hi: np.ndarray
    home: np.ndarray  # a safe posture with the hand out of the cameras' view of the workspace
    hand: Hand
    vmax: float = 1.0  # joint speed for retiming, rad/s
    collision_spheres: object = None  # RoboKit collision-sphere YAML of the model (e.g. a robokit asset's
    # COLLISION_SPHERE_PATH); None: generated from its collision meshes


@dataclass
class Base:
    """A holonomic mobile base: it drives along x, y and turns about z in the world, carrying the arms and cameras."""

    vmax: float = 0.3  # planar speed for retiming base waypoints, m/s
    wmax: float = 0.5  # turning speed, rad/s
    radius: float = 0.35  # footprint radius about the base origin, m (clearance from what the cameras see)
    floor_z: float = 0.0  # height of the floor it drives on, m
    tol: float = 0.02  # a drive ending this close to its goal has reached it, m (a walking robot's is looser)
    tol_yaw: float = 0.0524  # ... and this close to its heading, rad (3 degrees)


@dataclass
class Body:
    """A legged robot's posture: its pelvis height above the floor, from a deep squat or sitting to standing."""

    lo: float  # lowest pelvis height, m
    hi: float  # standing pelvis height, m
    vmax: float = 0.2  # speed for retiming height changes, m/s
    tol: float = 0.02  # a posture ending this close to its height has reached it, m


@dataclass
class Obs:
    cams: dict  # {name: Camera}; empty when asked for proprioception only
    q: dict  # {arm: (dof,) measured joints}
    hand: dict  # {arm: measured hand state: jaw (1,) opening 0..1; dex (n,) joint angles}
    steps: int  # control steps used so far
    step_limit: int  # the episode ends after this many
    instruction: str  # the task's native instruction
    base: Optional[np.ndarray] = None  # (3,) measured [x, y, yaw] of a mobile base in the world; None: fixed arms
    body: Optional[float] = None  # measured pelvis height above the floor of a robot with a `Body`, m


@dataclass
class Trajectory:
    """A time-parameterized motion for one or both arms (arms not in `q` hold still)."""

    t: np.ndarray  # (N,) seconds from now, increasing, t[0] > 0
    q: dict  # {arm: (N, dof) joint targets}
    hand: dict  # {arm: (N, n) hand commands}; an arm in `hand` but not in `q` moves only its hand
    T: dict  # {arm: (N, 4, 4) ee_link poses in the world} (FK of q; for pose-controlled arms)
    base: Optional[np.ndarray] = None  # (N, 3) [x, y, yaw] world targets of a mobile base (yaw unwrapped); None:
    #                                    it holds still. The arms keep their joints, riding along
    body: Optional[np.ndarray] = None  # (N,) pelvis-height targets of a robot with a `Body`; None: it holds its height


@dataclass
class Follow:
    steps: int  # control steps this call used
    done: bool  # the step budget is spent
    stopped: bool  # `stop` ended it early
    obs: Obs  # proprioception at the end (cams empty)


@runtime_checkable
class Robot(Protocol):
    arms: dict  # {name: Arm}, e.g. {"left": ..., "right": ...} or {"right": ...}
    dt: float  # seconds of one budget step (nominal where one step is a whole motion)
    ended: bool  # the env ended the episode (its own terminal or the step limit); Tools stops moving and reports it.
    # Unlike `Follow.done`, it also turns True on the env's own terminal between motions.
    # optional `table_z` (float): the table top the arms stand on, for cameras without depth
    # optional `base` (Base): a mobile robot's base (see above); absent on fixed robots
    # optional `body` (Body): a legged robot's pelvis height (see above)

    def obs(self, cams: bool = True) -> Obs:
        """Current observation; `cams=False` skips rendering (proprioception only, cheap)."""

    def follow(self, path: Trajectory, stop: Optional[Callable] = None) -> Follow:
        """Execute `path` at the env's native control rate and end at its last waypoint (settle and correct
        for controller sag if the arm needs it). Hold the other arms and hands at their last command, and a mobile
        base where it is unless `path.base` drives it (a base blocked short of its path ends the motion there), and a
        body at its height unless `path.body` moves it.
        `stop(obs)` (proprioception only) is checked at least every few control steps; True ends the
        motion there, holding the current measured posture. Stop at the step limit."""

    def robot_mask(self, cam: str, obs: Obs) -> Optional[np.ndarray]:
        """(H, W) bool, True where camera `cam` sees the robot; None to let the skills use the model."""

    def finish(self, out) -> dict:
        """End the episode: hold still to the step limit (or the env's own end), score it, save
        `video.mp4` and `actions.npy` in directory `out`. Returns at least {"success": bool}. Called once,
        by the server only."""
