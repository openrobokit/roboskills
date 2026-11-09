"""YAM arms on i2rt: the arm model and TCP, and the connection, the only code that sends to the motors (the safety
checks are eval/real/safety.py's).

Each arm is i2rt's `MotorChainRobot` (or its MuJoCo `SimRobot`) on one CAN channel, its own thread running the PD loop
and gravity compensation; we only set joint targets. Units are i2rt's: arm joints in radians, the gripper a seventh
joint read and commanded 0 (closed) .. 1 (open).
"""

import shutil
import threading
from pathlib import Path

import numpy as np

from roboskills.eval.real import safety


STROKE = 0.0475  # m each linear_4310 finger slides from open (joint7 = joint8 = 0) to closed
WIDTH = 2 * STROKE  # m between the pads at command 1.0
# The TCP in the arm's `gripper` body (the flange): between the fingertip facets 144.65 mm out along the flange's +z
# (i2rt's `grasp_site`), +x that approach, +y the axis the jaws close along (flange x), +z = x cross y (flange y).
T_EE_TCP = np.array([[0.0, 1, 0, 0], [0, 0, 1, 1e-4], [1, 0, 0, 0.14465], [0, 0, 0, 1]])
EE_LINK, JOINTS = "gripper", [f"joint{i}" for i in range(1, 7)]
HAND_JOINTS = ["joint8"]  # the model's actuated finger joint (joint7 mimics it)
SKIP = ("base", "link1")  # links that cannot reach the table or the other arm: never checked
HOLDS = 0.03  # a jaw commanded at least this far closed, and measured this far open, may hold something


def jaw_q_robokit(u):
    """The finger joint (joint8; joint7 mimics it) of jaw command `u` (0 closed .. 1 open) in RoboKit's reading of the
    model (`Hand.model_q`, for the planner, the safety checks, the robot mask and the hand's points): its slide joints
    run 0 closed (MuJoCo's 0 is open), so a jaw `u` open is STROKE * u."""
    return np.array([STROKE * float(np.clip(np.asarray(u, float).reshape(-1)[0], 0.0, 1.0))])


def model(arm, gripper, out):
    """i2rt's arm and gripper combined into one MJCF (meshes by absolute path), copied to `out`; returns its path."""
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    path = Path(out) / f"{arm}_{gripper}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        combine_arm_and_gripper_xml(ArmType.from_string_name(arm), GripperType.from_string_name(gripper)), path
    )
    return str(path)


def limits(rig, bodies, vmax):
    """The rig's `safety.Limits`, each arm's joint limits its model's (`bodies` {arm: safety.Body})."""
    return safety.Limits.from_rig(rig, {a: b.lo for a, b in bodies.items()}, {a: b.hi for a, b in bodies.items()}, vmax)


class Arm:
    """One YAM on i2rt, a `safety` driver: holds from the moment it connects; `send` streams targets, `hold` keeps it
    where it is."""

    n = 6  # arm joints; then the jaw

    def __init__(self, channel, arm="yam_ultra", gripper="linear_4310", sim=False, contact_kp=0.3):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.robot = get_yam_robot(channel=channel, arm_type=ArmType.from_string_name(arm), gripper_type=GripperType.from_string_name(gripper),
                                   zero_gravity_mode=False, sim=sim)  # fmt: skip
        self.sim, self.contact_kp = sim, float(contact_kp)
        self.kp = np.asarray(getattr(self.robot, "_kp", np.zeros(7)), float).copy()
        self.kd = np.asarray(getattr(self.robot, "_kd", np.zeros(7)), float).copy()
        self.target = self.read()
        self.hold()

    def read(self):
        """(7,) measured: six joints, then the gripper 0 closed .. 1 open."""
        return np.asarray(self.robot.get_joint_pos(), float).copy()

    def send(self, q, u, contact=False):
        """Target joints `q` (6,) and jaw `u` (1,); `contact`: the arm joints at contact_kp of their stiffness (compliant)."""
        self.target = np.r_[np.asarray(q, float), np.clip(np.asarray(u, float).reshape(-1)[:1], 0.0, 1.0)]
        if contact and not self.sim and self.kp.any():
            kp = self.kp.copy()
            kp[:6] *= self.contact_kp
            self.robot.command_joint_state(dict(pos=self.target.copy(), vel=np.zeros(7), kp=kp, kd=self.kd.copy()))
        else:
            self.robot.command_joint_pos(self.target.copy())

    def hold(self, u=None):
        """Hold the arm where it is measured now, at full stiffness (the jaw at `u`, else its last target)."""
        now = self.read()
        self.send(now[:6], self.target[6:] if u is None else u)

    def holds(self):
        """The jaw may hold something: a jaw left open, or shut all the way, holds nothing; one stopped anywhere between
        may (a soft object, a plastic cup, gives way to the command, so a jaw at its command can still hold it)."""
        return bool(self.target[6] < 1.0 - HOLDS and self.read()[6] > HOLDS)

    def alive(self):
        chain = getattr(self.robot, "motor_chain", None)
        return chain is None or bool(getattr(chain, "running", True))

    def close(self):
        """i2rt's close shuts the CAN bus under its still-running control loop (a ValueError traceback per arm): stop
        its server, then the loop, and wait for both before it closes the bus."""
        chain, stop, server = (getattr(self.robot, k, None) for k in ("motor_chain", "_stop_event", "_server_thread"))
        if chain is not None and stop is not None and server is not None:
            stop.set()
            server.join(timeout=2)
            loops = [
                t for t in threading.enumerate() if getattr(getattr(t, "_target", None), "__self__", None) is chain
            ]
            chain.running = False
            for t in loops:
                t.join(timeout=2)
        self.robot.close()
