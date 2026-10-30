"""The native Panda gripper: its axes, width and finger map."""

import xml.etree.ElementTree as ET

import numpy as np


AXES = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
TOUCH = 0.001  # m of Panda finger travel (finger joint 1 - finger joint 2) where the pads meet
WIDTH = 0.08 - TOUCH  # m between the pads fully open (each finger joint 0..0.04)


def opening(fingers):
    """Measured jaw opening 0..1 from the two Panda finger joints: the pad gap over the gap fully open."""
    return np.clip([(fingers[0] - fingers[1] - TOUCH) / WIDTH], 0.0, 1.0)


def finger_q(u):
    """The two Panda finger joints at jaw opening u (`opening` inverted): Hand.model_q."""
    return (TOUCH + WIDTH * np.asarray(u, float).reshape(-1)[0]) / 2 * np.array([1.0, -1.0])


def radians_xml(source):
    root = ET.fromstring(source)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    return root
