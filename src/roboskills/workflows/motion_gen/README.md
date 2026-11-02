# Motion Generation

Text, image or video in, rendered motion out, on a 3D character or a humanoid robot (such as the Unitree G1). Two ways
to run it:

- **With an agent** (Claude Code or Codex, the `motion-gen` skill in [SKILL.md](SKILL.md)): you write a request in
  plain words, from a text, an image or a video; the agent picks the models, writes its own scripts, renders, looks at
  the frames and revises.
- **Without an agent**: one command puts a video's person or a text's motion on a character or robot.

## Setup

Runs from a clone of this repository on Linux with an NVIDIA GPU (we use 24 GB cards), [uv](https://docs.astral.sh/uv/)
and ffmpeg, in about 60 GB of disk (about 30 GB of it model weights, downloaded on first use). For the agent route, also
install [Claude Code](https://claude.com/claude-code) or [Codex](https://github.com/openai/codex) and log in.

Some models are licensed or gated and need your own account. Get only the ones for what you want to make:

| To make                                    | Needs                                                                                        |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| a video's person on a character or robot   | [SMPL-X](https://smpl-x.is.tue.mpg.de)                                                       |
| ... with the hands                         | and [MANO](https://mano.is.tue.mpg.de) (without it, the body alone)                          |
| a motion from a text on a character or robot | SMPL-X and [Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| anything, with an agent                    | whatever the request needs, plus [SAM 3](https://huggingface.co/facebook/sam3) for objects   |

1. Download SMPL-X (`models_smplx_v1_1.zip`, a standard 3D human body model) and MANO (`mano_v1_2.zip`, its hand
   model) from their sites, an account each. For texts and for objects, request access to Llama 3 (used to turn text
   into motion) and SAM 3 on Hugging Face first: approval can take a while.
2. Clone the repository; every other command in this README runs from its root:

   ```sh
   git clone https://github.com/openrobokit/roboskills && cd roboskills && uv sync
   ```

3. Once per machine, run setup with the paths of the zips you downloaded (leave out `MANO_ZIP=...` if you skipped
   MANO). If you were granted Llama 3 or SAM 3, log in to Hugging Face first, with a read token from
   https://huggingface.co/settings/tokens:

   ```sh
   uv run hf auth login  # only if you were granted Llama 3 or SAM 3
   SMPLX_ZIP=path/to/models_smplx_v1_1.zip MANO_ZIP=path/to/mano_v1_2.zip bash src/roboskills/workflows/motion_gen/setup.sh
   ```

   Setup puts SMPL-X and MANO in `~/.robokit/body_models/`, makes each tool's environment, downloads the ready
   characters (below) and ends by naming what is still missing. Each missing item only blocks what needs it (the
   table above); rerun setup once you have it.

4. Check the setup: put the person in the example video on the dancer. It takes a few minutes, plus the weight
   download on the first run.

   ```sh
   uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --video assets/example_data/tennis.mp4 --character runs/motion_gen/characters/dancer.glb
   ```

   Then watch `runs/motion_gen/tennis/blender/compare.mp4`, the video beside the dancer. If a step fails, the run
   stops and that step's `status.json` (for example `runs/motion_gen/tennis/motion/status.json`) says why.

## Characters and robots

Setup downloads four ready characters into `runs/motion_gen/characters/`:

| File          | Character                                                    |
| ------------- | ------------------------------------------------------------ |
| `athlete.glb` | a man in a tank top and track trousers                       |
| `dancer.glb`  | a woman in a cropped jacket and shorts                       |
| `hoodie.glb`  | a man in a hoodie, cap and cargo trousers                    |
| `bob.glb`     | a stylized woman with a bob, in a jacket and flared trousers |

Each was generated with [Tripo](https://www.tripo3d.ai) from a text description and rigged with Mixamo's skeleton;
MIT, like RoboSkills. Six more are in the Hugging Face dataset
[openrobokit/roboskills](https://huggingface.co/datasets/openrobokit/roboskills) under `motion_gen/characters/`
(`blademaster`, `swordswoman`, `chef`, `elf_ranger`, `idol`, `wavy_hair`); fetch one into the same folder with:

```sh
uv run hf download openrobokit/roboskills motion_gen/characters/chef.glb --repo-type dataset --local-dir runs --revision 5e6107d0d19df2c3542206437d922560d604320e
```

Use a character by its path: `--character runs/motion_gen/characters/dancer.glb` without an agent, or name the file in
an agent's request. Try the ready characters, or bring your own: any character with a Mixamo skeleton works, whether
you download one from [Mixamo](https://www.mixamo.com) or make it elsewhere and rig it with Mixamo's skeleton, as an
FBX or a GLB exported from Blender.

A humanoid robot goes by name instead, `--robot NAME` or in an agent's request: `g1_inspire` is a Unitree G1 with
Inspire dexterous hands, whose fingers follow the person's; `g1`, `h1`, `h1_2`, `gr3` and `bhl` work the same way, but
their hands don't follow the person's fingers ([tools/robokit/SKILL.md](../../tools/robokit/SKILL.md)).

## Run with an agent

### Requests

Write the request as you would to a person, naming the files it uses: a character or a robot (see Characters and
robots), and the video or image it starts from. The agent builds the objects the action needs and keeps the hands on
them. For example:

| From              | Request                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| a text            | `make runs/motion_gen/characters/dancer.glb pick up a mug from a table and drink from it.`                                     |
| a video           | `put the person in assets/example_data/tennis.mp4 on runs/motion_gen/characters/dancer.glb, handling the same objects the same way.` |
| an image          | `make runs/motion_gen/characters/dancer.glb do what the person in path/to/photo.jpg is doing, with the same objects.`         |
| a text, on a robot | `make the G1 with Inspire hands pick up a mug from a table and drink from it.`                                               |

Add `Deliver within 30 minutes.` to a request only if you want a time limit.

### How to run it

Pass the request to the skill: `/motion-gen` in Claude Code, `$motion-gen` in Codex. `--permission-mode auto` has
Claude check each command before it runs; Codex's `--dangerously-bypass-approvals-and-sandbox`, like Claude's
`--permission-mode bypassPermissions`, runs every command without asking, so use them only on a machine you trust.

| Run it                   | How                                                                           |
| ------------------------ | ----------------------------------------------------------------------------- |
| Claude Code, interactive | start `claude` in the repository, then type `/motion-gen REQUEST`             |
| Claude Code, one command | `claude -p --permission-mode auto "/motion-gen REQUEST"`                      |
| Codex, one command       | `codex exec --dangerously-bypass-approvals-and-sandbox '$motion-gen REQUEST'` |

For example, a quick first request without objects:

```sh
claude -p --permission-mode auto "/motion-gen make runs/motion_gen/characters/dancer.glb wave."
```

Without a clone, from any directory, the agent can clone and set up the repository itself; have the files from Setup
step 1 ready:

```sh
claude -p --permission-mode auto "Follow the motion-gen skill in github.com/openrobokit/roboskills to make its dancer character wave."
```

The agent works in `runs/motion_gen/NAME/` and delivers `final/`: `video.mp4`, `motion.npz`, a contact sheet of stills
and `notes.md` (what it built, how to reproduce it, what remains).

### What the agent uses

It calls these tools and writes the rest itself:

| Tool                                    | Used for                                                 |
| --------------------------------------- | -------------------------------------------------------- |
| [Kimodo](../../tools/kimodo/SKILL.md)   | text to G1 or human motion                               |
| [GVHMR](../../tools/gvhmr/SKILL.md)     | a video's person as SMPL-X body motion                   |
| [WiLoR](../../tools/wilor/SKILL.md)     | a video's hands                                          |
| [SAM 3](../../tools/sam3/SKILL.md)      | an object's masks from text                              |
| [Character](character/SKILL.md)         | motion onto a Mixamo character, its joints and contact   |
| [RoboKit](../../tools/robokit/SKILL.md) | robot kinematics, retargeting and motion planning        |
| [Blender](../../tools/blender/SKILL.md) | `motion_studio.py`: an animated GLB and its saved motion |

## Run without an agent

`uv run python -m roboskills.workflows.motion_gen` (code in [`pipeline.py`](pipeline.py)) takes a video or a text and
a character or a robot. Its results go into the `--workspace` folder, one folder per step (see Steps), each with a
`status.json`.

### From a video

On a character: the setup check above. On a robot:

```sh
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis-g1 --video assets/example_data/tennis.mp4 --robot g1_inspire
```

Your own video, filmed with a moving camera:

```sh
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/walk --video path/to/your_video.mp4 --character runs/motion_gen/characters/dancer.glb --moving-camera
```

Each writes `blender/video.mp4` and `blender/compare.mp4`, the video beside the render. On a robot, `robot/motion.npz`
and `blender/scene.glb` (its links animated) line up with the video frame for frame.

### From a text

`--duration` is in seconds. These write `blender/video.mp4` (on a robot also `robot/motion.npz` and
`blender/scene.glb`); there is no source video to compare with.

```sh
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/wave --text "a person waves with the right hand" --duration 5 --character runs/motion_gen/characters/dancer.glb
```

```sh
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/wave-g1 --text "a person waves with the right hand" --duration 5 --robot g1_inspire
```

### Rerunning steps

`--tools` runs only the named steps (see Steps) and `--redo` reruns steps that already have results. The same motion
on another character (this replaces the dancer's results in that workspace):

```sh
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --character runs/motion_gen/characters/athlete.glb --tools retarget fingers blender --redo retarget fingers blender
```

For a quicker preview render, use EEVEE instead of the default Cycles:

```sh
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --tools blender --redo blender --engine eevee
```

### Steps

| Step       | Folder      | What it does                                                                                     |
| ---------- | ----------- | ------------------------------------------------------------------------------------------------ |
| `gvhmr`    | `motion/`   | [GVHMR](../../tools/gvhmr/SKILL.md): the video's person, an SMPL-X body per frame                 |
| `kimodo`   | `kimodo/`   | [Kimodo](../../tools/kimodo/SKILL.md): a text's motion, an SMPL-X body                            |
| `wilor`    | `hands/`    | [WiLoR](../../tools/wilor/SKILL.md): the video's hands per frame (needs MANO, else left out)      |
| `retarget` | `retarget/` | [`character/retarget.py`](character/retarget.py): the motion on the character's own proportions   |
| `fingers`  | `fingers/`  | [`character/fingers.py`](character/fingers.py): the wrists' rotation and the finger poses on the character's hands |
| `robot`    | `robot/`    | [RoboKit](../../tools/robokit/SKILL.md): with `--robot`, the person or the text's motion on the robot, in place of `retarget` and `fingers` |
| `blender`  | `blender/`  | [Blender](../../tools/blender/SKILL.md): every frame rendered (`video.mp4`); from a video, also the video beside it (`compare.mp4`) |
