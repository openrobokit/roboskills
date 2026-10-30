# Solve a simulation task

Another agent built a simulation task from the instruction in the request: `task.py` in the task directory the
request names, with its own notes in `notes.md` there. The task is frozen: read it, import it and run it, but never
change, copy over or re-create anything in that directory. Work in the current directory; write anything you need
there. The interpreter named in the request has the task's simulator and robokit.

Deliver `solution.py`, defining `solve(env)`. The harness calls it on a fresh env from `make(seed)`, right after
`reset()`, and it drives the robot with `env.step(action)`, in the action format `task.py` defines, until the task
is done. A seed counts as solved once `env.success()` holds within `env.max_steps` steps; the harness then ends the
episode. Every seed must be solved, from its own start.

Act only through `env.step`. Read anything you like: `env.observe()`, and the simulator's model and state that the
env holds, for kinematics, planning or object poses. Never write simulator state, move objects, change the task's
parameters, call `reset()` or `make()` inside `solve`, or decide from `env.success()`. Plan the motion the way a
robot would execute it: approach, grasp or contact, move, release; robokit has inverse kinematics.

The harness's check, which it runs when you call `finish()` (run it yourself first, through `sh`):

    PYTHON -m roboskills.workflows.agentic_sim.check TASK_DIR/task.py check --solution solution.py

It runs every seed and writes `check/receipt.json` (each seed's steps, whether it was solved, any error) and
`check/solve.mp4` (seed 0). Watch the video: the motion should look like a robot doing the task, without objects
passing through each other or jumping. Write `notes.md`: the approach and each seed's result. Leave no background
process running when you finish. Other agents share this machine and run the same commands: stop only processes you
started, by the PID you saved when you started them, never by a name or pattern (`pkill`, `pgrep -f`), and use only
the GPU(s) in `CUDA_VISIBLE_DEVICES`.
