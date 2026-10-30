<p align="center">
  <a href="https://openrobokit.com/roboskills"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/roboskills-banner.webp" alt="RoboSkills" width="100%"></a>
</p>

<p align="center">
  <a href="https://openrobokit.com/roboskills"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="https://openrobokit.com/roboskills"><img src="https://img.shields.io/github/actions/workflow/status/openrobokit/roboskills/build.yml.svg" alt="Build"></a>
  <a href="https://openrobokit.com/roboskills"><img src="https://img.shields.io/badge/homepage-RoboSkills-blue" alt="Homepage"></a>
  <a href="https://openrobokit.com/roboskills"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
</p>

RoboSkills is a **low-cost**, **minimal** set of agent workflows and tools for robotics.

<p align="center">
  <a href="https://openrobokit.com/roboskills"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/roboskills-teaser.webp" alt="Two Sharpa robot hands holding a red string in a cat's cradle" width="100%"></a><br>
  <em>Sharpa hands playing cat's cradle, reconstructed from a human video.</em>
</p>

## Workflows

<table>
  <tr>
    <td width="33%" valign="top">
      <a href="https://openrobokit.com/roboskills/showcase/run/?id=auto-research/bimanual-lifting"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/workflows/showcase/auto_research.webp" alt="Researched bimanual lifting: 0/2 initially, 2/2 after revision, 3/3 on fresh checks; Blender replay" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/auto_research/README.md">Auto Research</a></strong><br>
      Improve policies and reusable skills through repeated simulation experiments.
      <p><strong>Setting:</strong> Auto-research; no direct privileged simulator state in the policy API.</p>
    </td>
    <td width="33%" valign="top">
      <a href="https://openrobokit.com/roboskills/showcase/run/?id=libero/libero_goal_swap-2-s5"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/workflows/showcase/agent_as_policy.webp" alt="Blender render of an LLM agent putting a wine bottle on a drawer in LIBERO" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/agent_as_policy/README.md">Agent as Policy</a></strong><br>
      An LLM observes an episode and chooses actions turn by turn.
      <p><strong>Setting:</strong> Zero/one-shot demonstrations; no auto-research.</p>
    </td>
    <td width="33%" valign="top">
      <a href="https://openrobokit.com/roboskills/showcase/run/?id=agentic-simulation/thread-rope"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/workflows/showcase/agentic_sim.webp" alt="Two YAM arms thread a rope through a ring in an agent-built simulation" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/agentic_sim/README.md">Agentic Simulation</a></strong><br>
      Solve simulation tasks from natural-language instructions.
      <p><strong>Setting:</strong> Privileged simulator information available.</p>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <a href="https://openrobokit.com/roboskills/showcase/run/?id=real2sim/droid-dev-11"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/workflows/showcase/real2sim.webp" alt="A DROID demonstration split on a diagonal with its simulation, a Panda taking a marker out of a mug" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/real2sim/README.md">Real2Sim</a></strong><br>
      Build interactive simulations from demonstrations.
      <p><strong>Setting:</strong> Physics simulation included.</p>
    </td>
    <td width="33%" valign="top">
      <a href="https://openrobokit.com/roboskills/showcase/run/?id=motion-generation/street-dance"><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/workflows/showcase/motion_gen.webp" alt="Studio dance video split on a diagonal with Leonard and Michelle repeating both dancers' motion" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/motion_gen/README.md">Motion Generation</a></strong><br>
      One agent turns a text, image or video request into a motion video.
      <p><strong>Setting:</strong> Kinematic motion and rendering; no physics.</p>
    </td>
    <td width="33%" valign="top"></td>
  </tr>
</table>

## Features

### Low cost

With suitable tools, agents reach comparable performance on fewer, cheaper tokens.

<table>
  <tr>
    <th width="50%">Real2Sim · DexYCB potted meat can</th>
    <th width="50%">Motion Generation · Tennis serve</th>
  </tr>
  <tr>
    <td valign="top">Cost <strong>$0.09</strong>. 11% of Codex.</td>
    <td valign="top">Cost <strong>$0.00</strong>. 0% of Claude Code.</td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/cost/real2sim_potted_meat.webp" alt="A hand lifting a can beside a Sharpa hand lifting it in physics, split on a diagonal" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/cost/motion_gen_tennis.webp" alt="A tennis serve beside the same motion on a Mixamo character, split on a diagonal" width="100%"></td>
  </tr>
  <tr>
    <th width="50%">Agent as Policy · RoboDojo stack blocks</th>
    <th width="50%">Auto Research · RoboDojo press by number</th>
  </tr>
  <tr>
    <td valign="top">Cost <strong>$0.16</strong> with DeepSeek. 7% of Claude Code. Both 5/5 on 5 seeds.</td>
    <td valign="top">Cost <strong>$0.14</strong> with DeepSeek. 13% of Claude Code. Both 3/3 on 3 seeds.</td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/cost/agent_as_policy_stack_blocks.webp" alt="Blender render of DeepSeek as the policy stacking blocks in RoboDojo, layout 0" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/cost/auto_research_press_by_number.webp" alt="Blender render of DeepSeek's frozen auto-research solution pressing buttons by number in RoboDojo, test layout 2" width="100%"></td>
  </tr>
</table>

### Minimal

We commit to keeping each workflow under 1,000 lines of code.

## Tools

| Tool | Does | Reference |
| --- | --- | --- |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/robokit.svg" alt="" height="20"> [RoboKit](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/robokit/README.md) | kinematics, motion planning, robot hands | [Code](https://github.com/openrobokit/robokit) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/robokit.svg" alt="" height="20"> [GraspKit](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/graspkit/README.md) | learned grasps | [Code](https://github.com/openrobokit/graspkit) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/meta.svg" alt="" height="20"> [SAM 3](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/sam3/README.md) | text → masks | [Paper](https://arxiv.org/abs/2511.16719) · [Code](https://github.com/facebookresearch/sam3) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/meta.svg" alt="" height="20"> [SAM 3D](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/sam3d/README.md) | image → object mesh | [Paper](https://arxiv.org/abs/2511.16624) · [Code](https://github.com/facebookresearch/sam-3d-objects) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/microsoft.png" alt="" height="20"> [MoGe](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/moge3/README.md) | image → depth | [Paper](https://arxiv.org/abs/2607.17967) · [Code](https://github.com/microsoft/MoGe) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/calibration.svg" alt="" height="20"> [GeoCalib](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/geocalib/README.md) | image → gravity | [Paper](https://arxiv.org/abs/2409.06704) · [Code](https://github.com/cvg/GeoCalib) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/hand.svg" alt="" height="20"> [HaWoR](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/hawor/README.md) | video → hand motion | [Paper](https://arxiv.org/abs/2501.02973) · [Code](https://github.com/ThunderVVV/HaWoR) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/hand.svg" alt="" height="20"> [WiLoR](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/wilor/README.md) | image → hands (MANO) | [Paper](https://arxiv.org/abs/2409.12259) · [Code](https://github.com/rolpotamias/WiLoR) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/human.svg" alt="" height="20"> [GVHMR](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/gvhmr/README.md) | video → human motion | [Paper](https://arxiv.org/abs/2409.06662) · [Code](https://github.com/zju3dv/GVHMR) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/nvidia.png" alt="" height="20"> [Kimodo](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/kimodo/README.md) | text → G1 or human motion | [Paper](https://arxiv.org/abs/2603.15546) · [Code](https://github.com/nv-tlabs/kimodo) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/blender.svg" alt="" height="20"> [Blender](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/blender/README.md) | rendering | [Site](https://www.blender.org) · [Code](https://projects.blender.org/blender/blender) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/mesh.svg" alt="" height="20"> [CoACD](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/coacd/README.md) | mesh → convex collision parts | [Paper](https://arxiv.org/abs/2205.02961) · [Code](https://github.com/SarahWeiii/CoACD) |
| <img src="https://raw.githubusercontent.com/openrobokit/roboskills/main/assets/tools/logos/mujoco.png" alt="" height="20"> [MuJoCo](https://github.com/openrobokit/roboskills/blob/main/src/roboskills/tools/mujoco/README.md) | physics, IK, contact feedback, rendering | [Paper](https://doi.org/10.1109/IROS.2012.6386109) · [Code](https://github.com/google-deepmind/mujoco) |

## Installation

Requires Python 3.10 or newer:

```shell
uv add roboskills      # or: pip install roboskills
```

## Quick links

- [Homepage](https://openrobokit.com/roboskills)
- [RoboKit](https://github.com/openrobokit/robokit)
- [GraspKit](https://github.com/openrobokit/graspkit)
- [RetargetingKit](https://github.com/openrobokit/retargetingkit)
