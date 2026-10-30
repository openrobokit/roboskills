# Showcase previews

The root README uses these five looping WebP previews. Sources
are the site's existing reviewed clips;
no simulator, model or Blender job runs to make a preview.

| Preview | Site clip | Playback |
| --- | --- | --- |
| `auto_research.webp` | `showcase/auto-research/bimanual-lifting.mp4` | Complete 20 s clip in 8 s; the revised policy on fresh-check seed 5 |
| `agent_as_policy.webp` | `showcase/agent-as-policy/libero/libero_goal_swap-2-s5.mp4` | Complete 20 s clip in 8 s at 960 × 540; Pillow at quality 80 and `kmax=2`, since FFmpeg's settings leave blocks on its wall |
| `agentic_sim.webp` | `showcase/agentic-simulation/thread-rope.mp4` | Complete 14.5 s clip in 8 s |
| `real2sim.webp` | `roboskills/stages/real2sim/droid-dev-11-input.mp4` and `droid-dev-11.mp4` | DROID split on the site's diagonal, complete 8.5 s clip in 8 s |
| `motion_gen.webp` | `roboskills/bento/motion-generation.mp4` | The site's diagonal source and render, complete 15 s clip in 8 s |

All previews are 480 × 270 at 15 fps, encoded with FFmpeg's `libwebp_anim`,
quality 85, compression level 5 and `-loop 0`. Clips longer than 8 s use the filter below; the others drop its
`setpts`:

```sh
ffmpeg -i SOURCE.mp4 \
  -vf 'setpts=8/DURATION*(PTS-STARTPTS),fps=15,scale=480:270:flags=lanczos' \
  -an -c:v libwebp_anim -quality 85 -compression_level 5 -loop 0 PREVIEW.webp
```

Replace `DURATION` with the source duration in seconds (`ffprobe`), and `SOURCE`
and `PREVIEW` with the mapped paths.

Showcase links retain `openrobokit.com` for future deployment.


`../../roboskills-teaser.webp` (README) is the cat's-cradle render, 1920 × 1080 at WebP quality 85.

`../../cost/auto_research_press_by_number.webp` (README, Low cost) is DeepSeek's frozen Auto Research solution on
RoboDojo press by number, test layout 2. Its batch's event log, cut to that layout's env (each `act` keeps env 0's
action, an empty one where it had none), replayed in Isaac Sim (native success, same 279 steps; the three-env batch
crashes PhysX under the recorder), then rendered in Blender: its 20.03 s `clip.mp4` in 8 s, with the same WebP settings.

`../../cost/agent_as_policy_stack_blocks.webp` (README, Low cost) is DeepSeek's Agent as Policy episode on RoboDojo
stack blocks, layout 0, made the same way from its five-layout batch's event log (native success, the same 274 steps).

`../../cost/motion_gen_tennis.webp` (README, Low cost) is `assets/example_data/tennis.mp4` and the agent-free motion_gen
run's `blender/video.mp4` on Shannon, split on the bento's diagonal, in the 480 × 270, 15 fps encoding above at real
speed.

`../../cost/real2sim_potted_meat.webp` (README, Low cost) is DexYCB `s02-potted-meat-can-0861` from frame 10 and its
physics rollout, rendered like the EgoDex clip with exposure -0.85 and a darker hand, from a slow follow camera, split
on the bento's diagonal in the same encoding.
