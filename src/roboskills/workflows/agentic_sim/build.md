# Build a simulation task

Turn the instruction into a simulation task: the whole scene, built by you in a simulator you choose. The
interpreter named in the request has MuJoCo, PyBullet and Newton (with MuJoCo Warp), Blender (`bpy`) and, where
installed, NVIDIA's RTX (`ovrtx`), and robokit, whose robot modules give each robot's URDF (`URDF_PATH`, some
`MJCF_PATH`) and rest pose (`REST_Q`). Work in the current directory; write anything you need there. Nothing is
provided beyond this: build the room, furniture, lighting, camera, objects and robot yourself, as the instruction
describes them. Make the scene look like a photograph of a real place, and render it with a path tracer, not a
simulator's own renderer.

Deliver `task.py`, defining `make(seed) -> env`. Each seed is one episode; the instruction says how episodes vary.
The env has:

- `robot`: the robokit robot modules it uses, e.g. `["robokit.assets.robots.arms.franka_panda"]`
- `max_steps`: the episode's step limit
- `reset()`: put the episode in its start state
- `step(action=None)`: advance one control step; `None` holds the robot still
- `observe()`: `{"robot_q": {joint: position}, "objects": {name: [x, y, z, qw, qx, qy, qz]}}`
- `render()`: an HxWx3 uint8 image of the scene from a camera that shows the task
- `success()`: whether the instruction's goal holds now, measured from simulator state

Deliver `demo.py` beside it, defining `solve(env)`: a scripted robot that does the task on a fresh env from
`make(seed)`, right after `reset()`, acting only through `env.step`. It may read the env's model and state but never
write them, move objects or decide from `env.success()`. It shows the task can be done; if the robot cannot do it,
change the task (robot, objects, physics or step limit) until it can, within the instruction. It is withheld from
whoever solves the task later.

Checks, which the harness runs on `task.py` when you call `finish()` (run them yourself first, through `sh`):

    PYTHON -m roboskills.workflows.agentic_sim.check task.py check

Every seed builds; the seeds start in different states; holding still never succeeds; the robot starts at robokit's
rest pose (`REST_Q`) for every joint it shares with it; the start frame shows a scene, not mostly one backdrop
colour; and `demo.py` solves at least one seed. Look at `check/start-*.png`, `check/hold.mp4` and `check/demo.mp4`
and fix what looks wrong. Physics must hold: objects rest
on surfaces without falling through or drifting, and holding still leaves the scene still. Write `notes.md`: the
simulator you chose and why, the scene, and how success is measured. Leave no background process running when you
call `finish()`. Other agents share this machine and run the same commands: stop only
processes you started, by the PID you saved when you started them, never by a name or pattern (`pkill`, `pgrep -f`),
and use only the GPU(s) in `CUDA_VISIBLE_DEVICES`.
