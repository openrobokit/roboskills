"""Planted feet in the retargeting targets, read off the person.

`level(skin, fps)`: [T] the floor under the person each frame, from its posed skin [T, V, 3] (z-up): the lowest point,
its 1 s rolling minimum, smoothed. GVHMR's world drifts up and down by tens of centimetres over a clip; take it off
every height first, or `down` misses the frames the floor has risen on.
`down(points, fps, seen)`: [T, 4] which soles are planted: each of the person's four sole joints (`points` [T, 4, 3],
SMPL-X's `left_ankle`, `left_foot`, `right_ankle`, `right_foot`) within 3 cm of its own standing height and slower
than 0.2 m/s across the video's view (``seen``, the same joints in its camera: GVHMR's depth drifts), for 3 frames
or more.
`plant(soles, turns, positions, down, fps)`: the targets' positions with each planted foot held where it landed and
its lowest sole point on the floor, no sole point below it, the corrections blended between stances; first the whole
body gives back the drift of the soles that are down.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from roboskills.workflows.motion_gen.character.character import Character


SOLES = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")  # a character's sole bones, in `down`'s order
SOLE_JOINTS = ("left_ankle", "left_foot", "right_ankle", "right_foot")  # SMPL-X's, in the same order
NEAR_M, STILL_MPS, EASE_S = 0.03, 0.2, 0.15  # planted: this near its standing height, this still; a toe eases in


@dataclass(frozen=True)
class Soles:
    """The four soles: ``index`` the bone carrying each, ``points`` its sole skin [P, 3] in that bone's frame from its
    rest joint, ``rest`` [4] the bone's height standing flat on the floor."""

    index: tuple[int, ...]
    points: tuple[np.ndarray, ...]
    rest: np.ndarray


def soles(character: Character) -> Soles:
    """The character's `SOLES`: the skin each bone and the bones below it (a foot's toe has its own) drive most."""
    dominant, index = character.skin_weights.argmax(1), [character.bone(role) for role in SOLES]
    below = {j: {j} for j in index}
    for k, p in enumerate(character.parents):  # parents come first
        for j in index:
            if p in below[j]:
                below[j].add(k)
    under = [list(below[j] - {k for i in index if i != j and i in below[j] for k in below[i]}) for j in index]
    points = tuple(character.verts[np.isin(dominant, u)] - character.rest_joints[j] for j, u in zip(index, under))
    return Soles(tuple(index), points, character.rest_joints[index, 2])


def lowest(soles: Soles, turns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """[T, 4] each sole's lowest point above the floor, for the bones' turns [T, J, 3, 3] and positions [T, J, 3]."""
    out = np.empty((len(turns), len(soles.index)))
    for k, (j, points) in enumerate(zip(soles.index, soles.points)):
        out[:, k] = (np.einsum("tij,pj->tpi", turns[:, j], points)[..., 2] + positions[:, j, None, 2]).min(1)
    return out


def level(skin: np.ndarray, fps: float) -> np.ndarray:
    floor = minimum_filter1d(skin[..., 2].min(1), max(round(fps), 1), mode="nearest")
    return uniform_filter1d(floor, max(round(fps / 4), 1), mode="nearest")  # within the minimum's reach: never above


def _steady(out: np.ndarray) -> np.ndarray:
    for column in range(out.shape[1]):  # blips under 3 frames dropped, then lifts under 3 frames bridged
        for value in (True, False):
            flips = np.flatnonzero(np.diff(np.r_[0, (out[:, column] == value).astype(int), 0]))
            for a, b in zip(flips[::2], flips[1::2]):
                if b - a < 3 and (value or 0 < a and b < len(out)):
                    out[a:b, column] = not value
    return out


def down(points: np.ndarray, fps: float, seen: np.ndarray) -> np.ndarray:
    height = points[..., 2] - np.percentile(points[..., 2], 5, axis=0)
    velocity = np.gradient(seen, axis=0) * fps
    ray = seen / np.linalg.norm(seen, axis=-1, keepdims=True)
    velocity -= (velocity * ray).sum(-1, keepdims=True) * ray  # across the view: less its part along the ray
    return _steady((height < NEAR_M) & (np.linalg.norm(velocity, axis=-1) < STILL_MPS))


def plant(soles: Soles, turns: np.ndarray, positions: np.ndarray, down: np.ndarray, fps: float) -> np.ndarray:
    out, frames, bones = positions.copy(), np.arange(len(positions)), list(soles.index)
    held = down[1:] & down[:-1]  # each frame's mean step of the soles down on it and the one before, given back
    step = (np.diff(out[:, bones, :2], axis=0) * held[..., None]).sum(1) / np.maximum(held.sum(1), 1)[:, None]
    shift = np.r_[np.zeros((1, 2)), -np.cumsum(step, axis=0)]
    out[..., :2] += uniform_filter1d(shift, round(fps / 5), axis=0, mode="nearest")[:, None]  # the drift only
    for side in (0, 2):  # a foot: its heel and toe bones, moved up or down together
        ks, columns = [bones[side], bones[side + 1]], [side, side + 1]
        low = lowest(soles, turns, out)[:, columns].min(1)
        planted, flat = down[:, columns].any(1), down[:, columns].all(1)
        rest = soles.rest[columns]
        offset = np.where(flat, rest[0] - out[:, ks[0], 2], -low)  # heel and toe both down: the foot stands flat
        lift = np.interp(frames, frames[planted], offset[planted]) if planted.any() else np.zeros(len(out))
        out[:, ks, 2] += np.maximum(lift, -low)[:, None]  # and never a sole point below the floor
        if flat.any():  # the toe down to its rest height on flat frames, eased in and out
            near = np.abs(frames[:, None] - frames[flat][None]).min(1) / (EASE_S * fps)
            drop = np.interp(frames, frames[flat], (rest[1] - out[:, ks[1], 2])[flat])
            out[:, ks[1], 2] += drop * np.clip(1.0 - near, 0.0, 1.0)
        for k, column in zip(ks, columns):  # a planted sole keeps where it landed
            sole = down[:, column]
            if not sole.any():
                continue
            starts = np.flatnonzero(sole & ~np.r_[False, sole[:-1]])
            anchor = out[starts[np.searchsorted(starts, frames, side="right") - 1].clip(0), k, :2]
            shift = np.zeros((len(out), 2))
            shift[sole] = (anchor - out[:, k, :2])[sole]
            for axis in range(2):
                shift[:, axis] = np.interp(frames, frames[sole], shift[sole, axis])
            out[:, k, :2] += shift
    return out
