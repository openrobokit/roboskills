"""The G1's whole-body reference at 50 Hz: GEAR-SONIC's kinematic planner (`planner_sonic.onnx`) makes it walk and squat.

SONIC tracks the 29 joint angles and the pelvis orientation of a reference but does not walk by itself: the robot moves
over the floor only by stepping, through the leg joints of the reference. The planner (GR00T-WholeBodyControl,
`docs/source/references/planner_onnx.md`) generates such a reference: from four 30 Hz context frames of MuJoCo qpos
(pelvis position, wxyz quaternion, the 29 joints in MuJoCo order) and a command (a mode, a speed, a world movement
direction, a world facing direction, a pelvis height) it predicts 0.8-2.1 s of whole-body poses, which `Gait` resamples to
50 Hz and cross-fades into the running reference over 8 frames, as GR00T's deployment does: the context is sampled from
the reference itself 2 frames ahead of now, so successive plans join smoothly.

`Gait.step(cmd)` advances one control step and returns the reference frame for it; a command that differs from the one
being played (mode, facing, height, direction, speed), a moving command every REPLAN steps, or a plan running out
triggers a new plan. Commands come from `command`: a world [x, y, yaw] target is approached by slow walking along the
measured offset (feedback from the pelvis), facing the target's heading, and stopped (idle) within POS_TOL; a pelvis
height below standing squats to it; nothing holds the stance. Pure NumPy and onnxruntime: it runs in the simulator image.
"""

import math

import numpy as np


MUJOCO = tuple(f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
               "ankle_pitch", "ankle_roll")) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                                           "wrist_roll", "wrist_pitch", "wrist_yaw"))  # fmt: skip
CANONICAL = (  # HumanoidArena's canonical order (Isaac Lab's): SONIC's and the semantic40 action's
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)  # fmt: skip
TO_CANONICAL = np.array([MUJOCO.index(n) for n in CANONICAL])  # canonical = mujoco[TO_CANONICAL]
TO_MUJOCO = np.array([CANONICAL.index(n) for n in MUJOCO])  # mujoco = canonical[TO_MUJOCO]
IDLE, SLOW_WALK, SQUAT = 0, 1, 4  # planner modes
LOOK_AHEAD, BLEND = 2, 8  # 50 Hz frames: the next plan starts this far ahead of now; old and new cross-fade this long
REPLAN = 50  # steps between plans while moving (feedback on the measured pelvis; GR00T's deployment: 1 s)
TURN = 30  # degrees the walking direction may change before a plan is made sooner (after REPLAN / 4 steps)
LOW = 25  # plan again when fewer frames than this remain
POS_TOL, YAW_TOL = 0.04, math.radians(4)  # within these of a base target: stand (idle) facing its heading
SPEED = (0.15, 0.5, 1.2)  # m/s: slowest and fastest walk; speed per metre of remaining distance
STAND = 0.74  # m: a pelvis height target below this squats (the G1 stands at about 0.79)


def qmul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w * v - x * i - y * j - z * k, w * i + x * v + y * k - z * j, w * j - x * k + y * v + z * i,
                     w * k + x * j - y * i + z * v])  # fmt: skip


def qconj(q):
    return np.asarray(q, float) * [1, -1, -1, -1]


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qyaw(a):
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def rot6d(q):
    """The first two columns of q's rotation matrix, row-major: semantic40's [3:9] layout ("row")."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip
    return R[:, :2].reshape(-1)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def resample(frames, src_hz, dst_hz, t0=0.0, n=None):
    """Frames (k, 36) of qpos sampled at src_hz, at times t0 + i / dst_hz: positions and joints linear, quaternions
    slerped; clamped at the ends."""
    frames = np.asarray(frames, float)
    n = int(math.floor((len(frames) - 1) / src_hz * dst_hz)) + 1 if n is None else n
    out = np.empty((n, frames.shape[1]))
    for i in range(n):
        f = min(max((t0 + i / dst_hz) * src_hz, 0.0), len(frames) - 1.0)
        f0 = min(int(f), len(frames) - 1)
        f1, w = min(f0 + 1, len(frames) - 1), f - f0
        out[i] = (1 - w) * frames[f0] + w * frames[f1]
        out[i, 3:7] = slerp(frames[f0, 3:7], frames[f1, 3:7], w)
    return out


def command(base, target=None, body=None, hold_yaw=None, walking=False):
    """The planner command (mode, speed, movement, facing yaw, height) for a measured pelvis `base` [x, y, yaw] and a
    world base `target` [x, y, yaw] or a pelvis height `body` (else stand, facing `hold_yaw`). A robot standing starts
    walking past 1.5 POS_TOL (`walking`: it is walking now), and turns in place (slow walk, no movement) to a heading
    more than YAW_TOL off (2 YAW_TOL when standing)."""
    none = np.zeros(2)
    if target is not None:
        d = np.asarray(target[:2], float) - base[:2]
        dist = float(np.hypot(*d))
        if dist > POS_TOL * (1.0 if walking else 1.5):
            speed = float(np.clip(SPEED[2] * dist, SPEED[0], SPEED[1]))
            return (SLOW_WALK, speed, d / dist, float(target[2]), -1.0)
        if abs(wrap(float(target[2]) - base[2])) > YAW_TOL * (1.0 if walking else 2.0):
            return (SLOW_WALK, -1.0, none, float(target[2]), -1.0)
        return (IDLE, 0.0, none, float(target[2]), -1.0)
    face = base[2] if hold_yaw is None else hold_yaw
    if body is not None and body < STAND:
        return (SQUAT, -1.0, none, face, float(body))
    return (IDLE, 0.0, none, face, -1.0)


class Planner:
    """planner_sonic.onnx on onnxruntime (CUDA): one call per plan."""

    def __init__(self, path, device=0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 4, 1
        self.session = ort.InferenceSession(path, sess_options=so, providers=[("CUDAExecutionProvider", {
            "device_id": int(device)}), "CPUExecutionProvider"])  # fmt: skip
        self.inputs = {i.name: i.shape for i in self.session.get_inputs()}
        self.seed = 1234

    def __call__(self, context, mode, speed, move, face, height):
        """Frames (n, 36) at 30 Hz, from 4 context frames at 30 Hz and a command."""
        feed = {"context_mujoco_qpos": np.asarray(context, np.float32)[None], "target_vel": np.array([speed], np.float32),
                "mode": np.array([mode], np.int64), "movement_direction": np.array([[move[0], move[1], 0.0]], np.float32),
                "facing_direction": np.array([[math.cos(face), math.sin(face), 0.0]], np.float32),
                "random_seed": np.array([self.seed], np.int64)}  # fmt: skip
        extra = {"height": np.array([height], np.float32), "has_specific_target": np.zeros((1, 1), np.int64),
                 "specific_target_positions": np.zeros((1, 4, 3), np.float32),
                 "specific_target_headings": np.zeros((1, 4), np.float32)}  # fmt: skip
        if "allowed_pred_num_tokens" in self.inputs:
            k = self.inputs["allowed_pred_num_tokens"][-1]
            extra["allowed_pred_num_tokens"] = np.ones((1, k if isinstance(k, int) else 11), np.int64)
        feed.update({k: v for k, v in extra.items() if k in self.inputs})
        qpos, n = self.session.run(None, feed)
        out = np.asarray(qpos, float)[0, : int(np.asarray(n).reshape(-1)[0])]
        if not len(out) or not np.isfinite(out).all():
            raise RuntimeError("the kinematic planner returned no valid frames")
        return out


class Gait:
    """The running reference: 50 Hz qpos frames (pelvis position, wxyz quaternion, joints in MuJoCo order)."""

    def __init__(self, planner):
        self.planner, self.plans, self.plan_s = planner, 0, []

    def reset(self, pos, quat, joints):
        """Start from the measured pelvis pose and joints (canonical order), standing still."""
        frame = np.r_[np.asarray(pos, float), np.asarray(quat, float), np.asarray(joints, float)[TO_MUJOCO]]
        self.buf, self.i, self.cmd, self.since = frame[None].repeat(LOOK_AHEAD + 4, 0), 0, None, 0

    def frame(self):
        """The reference now: pelvis wxyz quaternion (world), the 29 joints (canonical order), pelvis height."""
        f = self.buf[min(self.i, len(self.buf) - 1)]
        return f[3:7].copy(), f[7:][TO_CANONICAL], float(f[2])

    def _changed(self, cmd):
        if self.cmd is None:
            return True
        mode, speed, move, face, height = cmd
        m0, s0, mv0, f0, h0 = self.cmd
        if mode != m0 or abs(wrap(face - f0)) > math.radians(3) or abs(height - h0) > 0.02:
            return True
        moving = mode not in (IDLE, SQUAT)
        turned = float(np.dot(move, mv0)) < math.cos(math.radians(TURN))
        return moving and (self.since >= REPLAN or (self.since >= REPLAN // 4 and (turned or abs(speed - s0) > 0.1)))

    def _plan(self, cmd):
        """A new plan from 2 frames ahead of now, cross-faded into the running one."""
        import time

        t = time.perf_counter()
        start = self.i + LOOK_AHEAD
        ctx = resample(self.buf, 50, 30, t0=start / 50, n=4)
        new = resample(self.planner(ctx, *cmd), 30, 50)
        old = self.buf[start : start + BLEND]
        if len(old) < BLEND:
            old = np.vstack([old, self.buf[-1:].repeat(BLEND - len(old), 0)]) if len(old) else new[:BLEND]
        w = (np.arange(1, BLEND + 1) / (BLEND + 1))[:, None]
        head = (1 - w) * old + w * new[:BLEND]
        for k in range(BLEND):
            head[k, 3:7] = slerp(old[k, 3:7], new[k, 3:7], w[k, 0])
        self.buf = np.vstack([self.buf[self.i : start], head, new[BLEND:]])
        self.i, self.cmd, self.since = 0, cmd, 0
        self.plans += 1
        self.plan_s.append(time.perf_counter() - t)

    def settled(self):
        """The reference stands still: a static command, and its pelvis moves less than 5 mm and 0.5 degrees over the
        rest of the plan."""
        rest = self.buf[self.i :]
        if self.moving() or len(rest) < 2:
            return not self.moving()
        yaw = np.unwrap([yaw_of(q) for q in rest[:, 3:7]])
        return bool(np.ptp(rest[:, 0]) < 0.005 and np.ptp(rest[:, 1]) < 0.005 and np.ptp(yaw) < math.radians(0.5))

    def moving(self):
        return self.cmd is not None and self.cmd[0] not in (IDLE, SQUAT)

    def step(self, cmd):
        """Advance one control step under `cmd` (from `command`); returns `frame()` for this step. A static plan (stand,
        squat) holds its last frame when it ends; a moving one is planned again."""
        if self._changed(cmd) or (self.moving() and len(self.buf) - self.i < LOW):
            self._plan(cmd)
        out = self.frame()
        self.i, self.since = min(self.i + 1, len(self.buf) - 1), self.since + 1
        return out
