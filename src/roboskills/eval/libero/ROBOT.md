# Robot: one Panda arm with a parallel gripper (LIBERO / LIBERO-Pro, MuJoCo)

World frame: metres, +z up. The arm, `"right"`, stands at the near edge of the table and reaches forward (+x);
+y is to the robot's left. Robot kinematics give its base pose and joint limits; the table height is `table_z`
in the `update` summary.

- Seven Panda joints with a parallel gripper: 8 cm between the pads when open, closing along the TCP's `squeeze`
  axis. Hand commands are one value, 0 closed and 1 open; the measured opening says whether something is between
  the pads. Closure alone does not establish a hold: check with fresh observations.
- Cameras: `agentview` looks at the table from in front of the robot; `robot0_eye_in_hand` is on the wrist.
  RGB, metric depth and native camera calibration are available; object locations come only from the cameras.
- Control: native joint-position control at 20 Hz. One step is 0.05 s; every motion and settling step spends the
  episode's 990-step budget.
- Drawers, cabinet doors, the microwave and the stove knob are articulated: pull handles along their opening
  direction, press or turn knobs.

LIBERO's own predicate judges the instruction; it is not shown during the episode. The episode ends the moment
the predicate holds (status shows `ended`), or when the steps run out.
