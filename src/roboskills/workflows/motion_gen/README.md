# Motion Generation

Text, image or video in, rendered motion out. The agent ([SKILL.md](SKILL.md), the `motion-gen` skill) picks the models, writes its own scripts,
renders, looks at the frames and revises. A video's person on a Mixamo character also runs alone as code, agent-free.

## Any request

The agent calls these tools and writes the rest in its run directory.

| Tool                                    | Call                                                     |
| --------------------------------------- | -------------------------------------------------------- |
| [Kimodo](../../tools/kimodo/SKILL.md)   | text to G1 or human motion                               |
| [GVHMR](../../tools/gvhmr/SKILL.md)     | a video's person as SMPL-X body motion                   |
| [WiLoR](../../tools/wilor/SKILL.md)     | a video's hands                                          |
| [SAM 3](../../tools/sam3/SKILL.md)      | an object's masks from text                              |
| [Character](character/SKILL.md)         | motion onto a Mixamo character, its joints and contact   |
| [RoboKit](../../tools/robokit/SKILL.md) | robot kinematics, retargeting and motion planning        |
| [Blender](../../tools/blender/SKILL.md) | `motion_studio.py`: an animated GLB and its saved motion |

```sh
bash src/roboskills/workflows/motion_gen/setup.sh  # once per machine
# The agent in this repo (the motion-gen skill): interactive, or one instruction
claude  # then: /motion-gen make Ch21.fbx wave.
claude -p "/motion-gen make Ch21.fbx wave."
codex exec '$motion-gen make Ch21.fbx wave.'
# The agent anywhere
claude -p "Follow the motion-gen skill in github.com/openrobokit/roboskills to make Ch21.fbx wave."
```

## Video to character

Recover the person, put the motion on the character, render: [`pipeline.py`](pipeline.py), a folder per tool, each with
a `status.json`. One person, body only.

| Tool                                             | Call                                                                           | Writes      |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | ----------- |
| [GVHMR](../../tools/gvhmr/SKILL.md)              | `recover_motion`: the person's SMPL-X body per frame                           | `motion/`   |
| [`character/retarget.py`](character/retarget.py) | `retarget_motion`: the motion on the character's own proportions               | `retarget/` |
| [Blender](../../tools/blender/SKILL.md)          | `render_motion`: the animated GLB, every frame, source and render side by side | `blender/`  |

```sh
# A video on a Mixamo character (FBX, or a GLB Blender wrote)
python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --video assets/example_data/tennis.mp4 --character Shannon.fbx
# A moving camera
python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/walk --video walk.mp4 --character Shannon.fbx --moving-camera
# The same motion on another character
python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --character Josh.fbx --tools retarget blender --redo retarget blender
# A faster render
python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --tools blender --redo blender --engine eevee
```
