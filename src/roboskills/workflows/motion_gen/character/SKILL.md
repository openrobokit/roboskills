---
name: character
description: A person's SMPL-X motion (a video's from GVHMR, or Kimodo's) on a Mixamo character through RoboKit's humanoid retargeting; functions make the character a RoboKit robot with a person's joint ranges and measure its joints and skin against an object; the rest is what to know to do the remainder yourself.
---

# Motion onto a character

Source `runs/motion_gen/env/character.sh`: it sets `ROBOSKILLS_PYTHON` (this repository's Python: RoboKit, warp, SciPy),
`GVHMR_PYTHON`, `BLENDER_PYTHON` and `CHARACTER_SKILL`. `wilor.sh` sets `WILOR_PYTHON` (a video's hands) and
`WILOR`, WiLoR's checkout. The functions below are in
`roboskills.workflows.motion_gen.character`.

## The person

- GVHMR: `save_npz` writes `world_*` and `camera_*` SMPL-X parameters, `intrinsics` and `fps`.
  `gvhmr.utils.smplx_utils.make_smplx("supermotion")` is the SMPL-X model: on the `world_*` parameters it gives the
  joints, on the `camera_*` ones the same joints in the camera (x right, y down, z ahead); its `.bm` has
  `v_template`, `shapedirs`, `J_regressor` and `vertex_joint_selector.extra_joints_idxs` (the last 10 are the
  fingertips). GVHMR has no fingers (`bm.left_hand_mean`, `bm.right_hand_mean` are a relaxed open hand), and its
  wrists are guesses, often tens of degrees off.
- RoboKit's `robokit.helpers.humanoid_retargeting.loaders.smplx.load_smplx_file(path, compute_vertices=False)` reads
  an AMASS SMPL-X `.npz` (`root_orient`, `pose_body` [T, 63], `trans`, `betas`, `mocap_frame_rate`; no `gender`: the
  neutral model). Kimodo's SMPL-X model writes one in a z-up world; GVHMR's `world_global_orient`, `world_body_pose`,
  `world_transl` and the mean of `world_betas` [T, 10] make one in GVHMR's y-up world. Its `T_world_joint_xyz_wxyz`
  [T, 55, 7] are the joints' world poses, named by `spec.joint_names` (`pelvis`, `left_hip`, ...); `human_height` is
  the person's T-pose skin height. It zeroes the hands: a `pose_hand` in the file is not read. A joint's world turn
  is the identity at SMPL-X's T-pose in its own y-up frame; Kimodo's z-up file stands the body up through
  `root_orient`, so its turns carry that turn A (+90 degrees about x) on their left.
- WiLoR: per detected hand MANO's `global_orient` and `hand_pose` and `pred_keypoints_3d` (OpenPose order: wrist, then
  thumb, index, middle, ring, pinky from the root out); `wilor.utils.renderer.cam_crop_to_full` places them in the image
  after flipping `pred_cam`'s x for a left hand. MANO's 15 finger joints are SMPL-X's hand joints, in the same order and
  from the same flat rest: a right hand's `hand_pose` fills the right-hand slots of an SMPL-X hand pose (AMASS
  `pose_hand`: the left hand's 45 values, then the right's) as it is; a left hand is read mirrored, so its turns, and
  its `global_orient`, are F r F with F = diag(-1, 1, 1) (axis-angle (x, -y, -z)). `global_orient` is the hand's turn in
  the camera: the wrist's turn in its parent is that parent's camera turn (down GVHMR's `camera_*` chain) transposed
  times it. An SMPL-X hand pose's turns are absolute (zero: a flat hand); GVHMR's SMPL-X model adds
  `left_hand_mean`/`right_hand_mean` to the hand pose it is given, so give it the pose minus those for the joints. The
  detector's left/right label is often wrong: a detection belongs to the hand whose wrist, projected into the picture
  (the camera joints through the intrinsics), lies nearest its wrist keypoint. A detection whose label disagrees with
  that hand was read with the wrong side, and one turned far from its neighbours was read flipped (palm in for palm
  out); neither is usable as it is.

## The character

- `character.load_character(path)` (a Mixamo `.glb`): `joint_names`, `parents`, `rest_joints` [J, 3], `verts`, `faces`,
  `skin_weights`, `bone(role)`. Its world is Blender's after importing the GLB (z up, metres, +X its left, -Y its front)
  with the lowest skin vertex moved to z = 0. A pose is each bone's world turn from its rest, `turns` [T, J, 3, 3];
  `character.positions(character, turns, root)` gives the joints [T, J, 3] and `character.skinned(character, turns,
  joints)` the posed skin [T, V, 3].
- `robot.robot(character)` is the character as a RoboKit `Robot`: three joints per bone about its anatomical axes, each
  limited to a person's range (`contact.RANGES`), so an elbow or knee bends only as a hinge; each bone is a link of its
  full name (`c.joint_names[c.bone("Hips")]`, prefix included); the fingers are held unless `fingers=True` (63 joints,
  153 with them); held fingers stay at rest, for turns you set on them afterwards. `robot.turns(character, r, qpos)`
  turns RoboKit's solutions [T, 7 + dofs] (base xyz + wxyz, then the joints) into `turns`;
  `character.positions(character, turns, qpos[:, :3])` places the joints. `robot.SMPLX` names the SMPL-X joint each
  body bone answers, by role.
- RoboKit's `HumanoidRetargeterOnline(config, robot=r)` (with `HumanoidRetargeterOnlineConfig` and `LinkMapping`, from
  `robokit.helpers.humanoid_retargeting`) solves frame by frame, each from the last: `solver.warmup(1)`, then
  `solver.solve_numpy(frames[t : t + 1], human_heights=ones(1))`; about 20 ms a frame without the fingers, 0.7 s with
  them. `HumanoidRetargeterOffline` solves the whole clip at once, much slower. The first solve with a new set of
  targets compiles GPU kernels for minutes; later ones reuse them. Targets that track well:
  - The person's turns and positions in the character's z-up world, the T-pose as the identity as a Mixamo rest is:
    A R Aᵀ and A p for anything in GVHMR's y-up world (an AMASS file made from it included), R Aᵀ and p as they are
    for Kimodo's z-up file; then the floor at z = 0. Without the change of basis every target is a quarter turn off.
  - On the character's own proportions: each bone turns as its SMPL-X joint (a bone without one, as its parent), and
    `character.positions(character, turns, root)` runs the positions down the character's bones from the person's
    pelvis position (all three axes) times s, the character's rest hips height over the person's standing pelvis
    height. Scaling the person's joints instead puts a neck or head where the character's cannot reach, and bends
    the torso.
  - Each bone its own target: `link_mapping={bone: LinkMapping(bone, zeros(3), zeros(4))}` for the bones weighted
    below, `human_root_name` the Hips bone, `human_height_assumption=1.0`, and `frames[t, k]` the position and the
    turn (wxyz) of bone `solver.human_joint_names[k]`. The `target` task's `position_weight` and
    `orientation_weight` per bone (full names): Hips (30, 2), Spine2 (0.7, 0.7), each UpLeg (1.5, 1), Foot (30, 2),
    ToeBase (30, 1), Arm (1.5, 1), Hand (2, 1.2).
  - The `rest` task, once weighted, pulls toward the middle of each range, which bends the knees: set its
    `rest_q_arg` to None (the robot's zero pose, the T-pose) and its `weight` to 0.05 (setting `rest_q` is ignored).
- A Mixamo rest hand's fingers lie parallel and its thumb runs along the index; a person's (SMPL-X's, MANO's) fan
  out, with the thumb pointing away from the palm. A finger turn copied from the person's rest keeps that difference:
  fingers pressed together, the thumb tucked in. SMPL-X's `left_index1`-`3` (and middle, ring, pinky, thumb) are the
  Mixamo `LeftHandIndex1`-`3`.

`retarget.retarget_motion(motion, character, out)` runs these targets on GVHMR's `smplx.npz` into
`out/motion.npz`, the renderer's MOTION; `check_motion` reports how far the limbs strayed. The first of a clip's solves
starts from the T-pose: solved once, the arms end short of it, so solve the first frame a few times first. RoboKit's
first solve failed with `may need newer version of nvJitLink library` on driver 580.178.04 (Warp 1.18.0) and worked on
580.82.07.

## Joints and contact (`contact`)

- `contact.joints(character, turns)`: per joint `bend`, `sideways` and `twist` from rest (degrees, about the joint's
  own axes: an elbow bends but does not bend sideways, a wrist bends toward the palm much further than toward the
  thumb), `past`, how far beyond a person's range (`contact.RANGES`) on each frame, and `step`, how far it turned
  since the previous frame.
- `contact.touch(character, turns, joints, side, vertices, faces, poses, device="cuda:0")` (`side` "Left" or
  "Right"): per hand bone (`Hand` the palm,
  `HandIndex2`...) its skin's distance to an object [T] (mm, negative inside), for object poses [T, 4, 4] (world from
  object). It signs by winding number, so an open object mesh works, and on the GPU it takes every frame
  of a clip at once. An object rests on the palm where the palm's distance (`Hand`) is zero: moving it along the
  palm's normal by about that distance puts it there. A distance to a surface cannot see through a thin wall: a
  finger that went through a cup's side, or into an open container, is still only millimetres from the surface and
  reads as touching. How deep it went shows against a closed solid, such as the object's convex hull when the hand
  stays outside it.

## What to know

- A bone's turn moves every bone below it; a child's turn in its parent is `turns[parent].T @ turns[child]`.
- A fix made frame by frame makes limbs shake; an edit eased over neighbouring frames, or a filter over time, does
  not.
- A hand's twist about the forearm is the forearm's: with the twist on the hand bone, the skin of the wrist collapses.
  A hand turn set from an object's axes puts the whole difference into the wrist.
- Feet: GVHMR's depth drifts, so a foot the picture shows planted slides. A sole is down where it barely moves in
  the picture (the camera joints through the intrinsics).
- The video's camera: the rigid fit (Kabsch) of the camera joints onto the world joints on the stable body joints
  (not the wrists) gives world-from-camera; one fit for the clip when the camera stands still. Into the character's
  world it is lowered and scaled as the root is.
- GVHMR's elbow and knee turns often hold a sideways bend (tens of degrees) that a person's cannot make. The robot's
  hinge reaches the same hand or foot with the upper arm or thigh turned instead; clamping each joint on its own
  after the fact moves the hand or foot.
- `contact.joints` measures swing then twist; the robot's three joints turn in sequence. Near a half-turn of twist
  the two read differently, so a robot pose inside its limits can read as past a range there.
- With a video, the person's hand shows how it is held: its turn, where the object lies across the palm and
  fingers, how far each finger closes. An object held still in a hand moves with it (the hand's world transform
  times a grip). Hand readings jitter by millimetres and degrees from frame to frame, so a hand copied frame by frame
  passes in and out of an object it holds still.
- The skin is the bones' weighted blend, and the motion is the bones. When a hand's joint centres are inside an
  object, the bones are wrong: only turning the fingers and wrist or moving the object fixes it. When the joints are
  outside and only the skin dips in by a few millimetres, the skin's thickness and blending cause it. Moving skin
  vertices changes the picture, not the motion.

## What a video cannot tell

- Depth: an arm raised overhead and an arm reaching toward the camera can look the same, and so can a flat
  object tilted toward or away from it. Decide from what the person is doing.
- What the video shows clearly (silhouettes, where the hands are in the image) must stay right. Where it
  cannot tell (depth, covered parts, an object's back), choose what makes sense for the action.
