"""CaP-X's Robosuite tasks behind the shared Robot contract; see SKILL.md."""

TASKS = {  # task: (CaP-X's step limit, its instruction without the API hints)
    "cube_lifting": (1500, "Pick up the red cube and lift it."),
    "cube_stack": (1500, "Pick up the red cube and gently stack it on top of the green cube, then release it."),
    "cube_restack": (
        1500,
        "Gently place the red cube on top of the green cube and then open the gripper. "
        "Nothing should be dropped from a height.",
    ),
    "spill_wipe": (4000, "Wipe up the brown spill. A sponge is already attached to the end-effector."),
    "nut_assembly": (1000, "Grasp the brown square nut by its handle and insert it onto the brown square peg."),
    "two_arm_lift": (
        5000,
        "The two arms should coordinate to lift the pot: each arm grasps the handle on its side, "
        "then both arms lift the pot up at the same height.",
    ),
    "two_arm_handover": (
        5000,
        "The left arm should pick up the hammer, lift it, and hand it over to the right arm. "
        "The right arm should then grasp the hammer by its handle, not its head.",
    ),
}
