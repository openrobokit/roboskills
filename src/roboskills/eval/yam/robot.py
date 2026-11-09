"""Real I2RT YAM arms and RealSense cameras behind the shared Robot contract: the real bench (eval/real/robot.py) with
YAM's arms (arm.py: i2rt, the arm model and its jaw).
"""

import numpy as np

from roboskills.eval.interface import Arm, Hand
from roboskills.eval.real import safety
from roboskills.eval.real.robot import RealRobot
from roboskills.eval.yam import arm as yam
from roboskills.tools.robokit.robot_model import RobotModel


class Yam(RealRobot):
    NAME = "YAM"

    def _arms(self, names, vmax):
        rig = self.rig
        for name in names:
            spec = rig["arms"][name]
            kind, gripper = spec.get("arm", "yam_ultra"), spec.get("gripper", "linear_4310")
            xml = yam.model(kind, gripper, self.out / "model")
            T = np.asarray(spec["T_world_base"], float)
            self.hw[name] = yam.Arm(spec["channel"], kind, gripper, sim=self.sim, contact_kp=rig.get("contact_kp", 0.3))
            hand = Hand("jaw", yam.T_EE_TCP.copy(), np.ones(1), np.zeros(1), joints=list(yam.HAND_JOINTS), model_q=yam.jaw_q_robokit,
                        width=float(spec.get("width", yam.WIDTH)))  # fmt: skip
            self.arms[name] = Arm(name, xml, list(yam.JOINTS), yam.EE_LINK, T, None, None, None, hand, vmax=vmax)
        kin = RobotModel(self)  # RoboKit's model of the arms, the planner's (built once: it fits collision spheres)
        self.bodies = {a: safety.Body(kin, a, yam.SKIP) for a in self.arms}
        for name, arm in self.arms.items():  # the model's joint limits; i2rt clips 0.15 rad out
            body = self.bodies[name]
            arm.lo, arm.hi = body.lo.copy(), body.hi.copy()
            # the rig's `rest` (default all zeros, folded): where the arms go home, and where they park before their
            # motors turn off
            arm.home = np.clip(np.asarray(rig["arms"][name].get("rest", np.zeros(6)), float), body.lo, body.hi)
        self.limits = yam.limits(rig, self.bodies, vmax)
