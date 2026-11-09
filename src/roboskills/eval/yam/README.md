# Real I2RT YAM

Two real [I2RT YAM](https://github.com/i2rt-robotics/i2rt) arms with parallel jaws and four RealSense cameras, run by
`agent_as_policy` behind the same `Robot` contract as the simulated benchmarks. `launch.make(instruction, rig=...)`
starts an episode server on this machine (it owns the CAN buses and the cameras); an operator reviews each plan in a
web panel and gives the verdict. What every real bench shares is in [`../real/`](../real/). See [SKILL.md](SKILL.md)
for setup, the rig file and safety, and [ROBOT.md](ROBOT.md) for what the agent is told.

## Run

```bash
uv sync --locked --group dev --group yam      # i2rt, pyrealsense2, imageio
python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.yam.launch \
  --out runs/agent_as_policy/yam/put_cup-001 -- rig=~/.config/roboskills/yam.yaml instruction="put the red cup on the plate"
```

Then open http://localhost:8080, reset the scene and press **Start**. Keep a hand on the e-stop.

Without hardware, `sim=true` runs i2rt's MuJoCo arms behind the same panel.
