"""RoboCasa365 kitchens with RoboCasa's PandaOmron mobile manipulator behind the shared Robot contract; see SKILL.md."""

TASKS = {  # RoboCasa environment: (step limit at 20 Hz, what RoboCasa's _check_success requires)
    "NavigateKitchen": (
        2000,
        "the arm's shoulder within 0.2 m of RoboCasa's parking spot at the named fixture, facing it",
    ),
    "DeliverBrewedCoffee": (
        3000,
        "the mug on the dining counter within 0.3 m of the breakfast plate, both touching the counter, the gripper away",
    ),
    "PickPlaceCounterToSink": (2000, "the object inside the sink basin, the gripper away from it"),
    "CloseDrawer": (1500, "the drawer at most 5 % open"),
    "TurnOnMicrowave": (1500, "the microwave turned on (its start button pressed), the gripper away from the buttons"),
}
