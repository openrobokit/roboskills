"""The aloha-agilex jaw's command-to-finger map, without the simulator (the host imports it for `Hand.model_q`)."""

import numpy as np


WIDTH = 0.09  # m between the pads at command 1.0; each finger joint then stands at WIDTH / 2
SQUEEZE = 0.2  # below this command the finger target goes on past touching (to RoboTwin's closed, -1 cm)


def finger(u, lo):
    """Finger joint target for a jaw command u (0..1, the fraction of WIDTH between the pads): WIDTH * u / 2 over
    most of the range (a width command is that width), bending down to RoboTwin's own closed target `lo` (past
    touching, so a closed jaw squeezes) at 0."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return WIDTH / 2 * u + lo * np.clip(1.0 - u / SQUEEZE, 0.0, 1.0)


def model_q(u, g_lo):
    """`Hand.model_q` (as a partial with RoboTwin's closed finger target `g_lo`): the URDF's two finger joints for jaw
    command `u`, held at 0 where the target goes past touching."""
    return np.repeat(np.clip(finger(np.asarray(u, float)[:1], g_lo), 0.0, None), 2)
