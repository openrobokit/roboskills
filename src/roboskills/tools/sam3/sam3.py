# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers>=5.0", "torch", "torchvision", "pillow", "numpy", "opencv-python-headless"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""SAM 3: `Sam.detect` (text prompts -> scored masks per image, the model kept loaded in a worker process) and `segment_video`
(a video -> one indexed PNG per frame). Run as a script, this file is the model side, in its own environment (`uv run`
of its inline dependencies); nothing of roboskills is imported there."""

import base64
import io
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np


MIN_SCORE = 0.3  # default SAM 3 score a detection must reach, for a find and every later re-find of the same object
PALETTE = [0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255]  # background, 1, 2, ...


UV = ["uv", "run", "--quiet", __file__]  # this file in its own environment, on $CUDA_VISIBLE_DEVICES


def segment_video(video: Path, prompts: list[str], out: Path, frame: int = 0, threshold: float = 0.4,
                  point: tuple[float, float] | None = None, counts: list[int] | None = None, smaller_on_top: bool = False,
                  stride: int = 1):  # fmt: skip
    """Masks for every frame of `video`, from each prompt's best detection at `frame` (its `counts` best, when given),
    tracked both ways: out/00000.png ... (indexed; 0 background, ids 1, 2, ... in prompt order, later on top, or the
    smaller on top per frame with `smaller_on_top`: a lid over its cup) and out/objects.json (id -> prompt, detected
    text, score, box). A prompt not found falls back to its last word ("sugar box" -> "box"), then to any alternates
    it lists after "|" ("white usb drive|white object"), else is left out. `point`, a pixel on the first prompt's
    object at `frame`, picks the detection whose box holds it, else prompts the tracker itself. With `stride` the
    tracker sees every `stride`-th frame (through `frame`) and the frames between take their nearest one's masks.

    python -m roboskills.tools.sam3 --video V --prompts "sugar box" "right hand" --out masks/
    """
    video, out = Path(video).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    at, many = ",".join(map(str, point)) if point else "-", ",".join(map(str, counts or [1] * len(prompts)))
    order = "area" if smaller_on_top else "prompt"
    argv = ["segment", video, out, frame, threshold, at, many, order, stride, *prompts]
    subprocess.run([*UV, *map(str, argv)], check=True)


def check_masks(masks: Path, frame: int, names: list[str]) -> dict:
    """SAM 3's object (label 1) and actor (2) pixels at the reference `frame` (`names`, their prompts); ok with both."""
    from PIL import Image

    labels = np.array(Image.open(Path(masks) / f"{frame:05d}.png"))
    pixels = [int((labels == i).sum()) for i in (1, 2)]
    reason = "; ".join(f"SAM 3 found no {name} at frame {frame}" for name, n in zip(names, pixels) if not n)
    return dict(ok=not reason, reason=reason, object_pixels=pixels[0], hand_pixels=pixels[1])


class Sam:
    """SAM 3 kept loaded in a worker process (`serve`); call close() before reusing its GPU allocation."""

    def __init__(self):
        self._closed = False
        self.p = subprocess.Popen([*UV, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  start_new_session=True)  # fmt: skip
        try:
            self._reply()
        except BaseException:
            self.close()
            raise

    def close(self):
        """Send EOF and reap the launcher and model worker before returning; a stuck worker is killed with its own
        process group only, never the owning launcher."""
        if self._closed:
            return
        with suppress(BrokenPipeError):
            self.p.stdin.close()
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(self.p.pid, signal.SIGKILL)
            self.p.wait(timeout=5)
        self.p.stdout.close()
        self._closed = True

    def _reply(self):
        if (got := next((json.loads(l[5:]) for l in self.p.stdout if l.startswith("SAM3 ")), None)) is None:
            raise RuntimeError("the SAM 3 worker exited (see the server log)")
        return got

    def detect(self, rgb, prompts, threshold=0.3):
        """{prompt: [{score, box_px, area_px, mask (H, W) bool}]}, best first."""
        from PIL import Image

        png = io.BytesIO()
        Image.fromarray(np.asarray(rgb, np.uint8)).save(png, "png")
        print(json.dumps(dict(png=base64.b64encode(png.getvalue()).decode(), prompts=list(prompts), threshold=threshold)),
              file=self.p.stdin, flush=True)  # fmt: skip
        if not (got := self._reply()).get("ok"):
            raise RuntimeError(f"SAM 3 failed: {got.get('error')}")
        m = np.load(io.BytesIO(base64.b64decode(got["masks"])))
        for det in (x for dets in got["detections"].values() for x in dets):
            det["mask"] = m[str(det["id"])]
        return got["detections"]


def serve():
    """The worker: one JSON request per stdin line {png: the image, base64, prompts, threshold}; one reply per stdout
    line, "SAM3 " + JSON {ok, detections: {prompt: [{id, score, box_px, area_px}]}, masks: base64 .npz by id}."""
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    device, dtype = ("cuda", torch.bfloat16) if torch.cuda.is_available() else ("cpu", torch.float32)
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=dtype).to(device).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    print("SAM3 " + json.dumps(dict(ready=True, device=device)), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            rgb = Image.open(io.BytesIO(base64.b64decode(req["png"]))).convert("RGB")
            img = proc(images=rgb, return_tensors="pt").to(device=device, dtype=dtype)
            with torch.no_grad():  # the image is encoded once; each prompt only runs the text and the decoder
                vision = model.get_vision_features(pixel_values=img["pixel_values"])
            results, masks = {}, {}
            for prompt in req["prompts"]:
                txt = proc(text=prompt, return_tensors="pt").to(device=device)
                with torch.no_grad():
                    out = model(vision_embeds=vision, input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
                r = proc.post_process_instance_segmentation(out, threshold=req.get("threshold", 0.3),
                                                            target_sizes=img["original_sizes"].tolist())[0]  # fmt: skip
                results[prompt] = []
                for score, box, mask in sorted(zip(r["scores"].tolist(), r["boxes"].tolist(), r["masks"].cpu().numpy().astype(bool)),
                                               key=lambda t: -t[0]):  # fmt: skip
                    if mask.any():
                        masks[str(len(masks))] = mask
                        results[prompt].append(dict(id=len(masks) - 1, score=round(score, 3), box_px=[round(x) for x in box], area_px=int(mask.sum())))  # fmt: skip
            npz = io.BytesIO()
            np.savez_compressed(npz, **masks)
            print("SAM3 " + json.dumps(dict(ok=True, detections=results, masks=base64.b64encode(npz.getvalue()).decode())), flush=True)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            print("SAM3 " + json.dumps(dict(ok=False, error=repr(e), detections={})), flush=True)


def segment(video, out, frame, threshold, at, many, order, stride, *prompts):
    """`segment_video`'s model side."""
    import cv2
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    capture, frames, frame, out = cv2.VideoCapture(video), [], int(frame), Path(out)
    point = None if at == "-" else [float(x) for x in at.split(",")]
    while (got := capture.read())[0]:
        frames.append(np.ascontiguousarray(got[1][..., ::-1]))
    model, proc = (
        Sam3Model.from_pretrained("facebook/sam3").cuda().eval(),
        Sam3Processor.from_pretrained("facebook/sam3"),
    )
    found, i = {}, 0
    for prompt, count in zip(prompts, map(int, many.split(","))):
        i += 1
        names = prompt.split("|")
        for text in dict.fromkeys([names[0], names[0].split()[-1], *names[1:]]):
            inputs = proc(images=Image.fromarray(frames[frame]), text=text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                r = proc.post_process_instance_segmentation(
                    model(**inputs), threshold=float(threshold), target_sizes=inputs["original_sizes"].tolist()
                )[0]
            boxes, scores = r["boxes"].tolist(), r["scores"].tolist()
            if i == 1 and point:  # only the detection holding the point
                held = [j for j, b in enumerate(boxes) if b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]]
                boxes, scores = [boxes[j] for j in held], [scores[j] for j in held]
            if scores:
                for j, best in enumerate(np.argsort(scores)[::-1][:count]):
                    found[i + j] = dict(prompt=prompt, text=text, score=scores[best], box=boxes[best])
                i += min(count, len(scores)) - 1
                break
        if i == 1 and point and i not in found:  # the point alone prompts the tracker
            found[i] = dict(prompt=prompt, text=None, score=None, box=None, point=point)
    del model
    torch.cuda.empty_cache()
    tracker = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").cuda().eval()
    video = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    seen = list(range(frame % int(stride), len(frames), int(stride)))  # the frames the tracker sees
    state = video.init_video_session(
        video=[frames[f] for f in seen], inference_device="cuda", video_storage_device="cpu"
    )
    labels, start = np.zeros((len(frames), *frames[0].shape[:2]), np.uint8), seen.index(frame)
    with torch.no_grad():
        for i, hit in found.items():
            prompt = (
                dict(input_boxes=[[hit["box"]]])
                if hit["box"]
                else dict(input_points=[[[hit["point"]]]], input_labels=[[[1]]])
            )
            video.add_inputs_to_inference_session(state, frame_idx=start, obj_ids=i, **prompt)
            tracker(inference_session=state, frame_idx=start)  # each object conditioned before tracking
        for reverse in (False, True):
            for step in tracker.propagate_in_video_iterator(
                state, start_frame_idx=start, max_frame_num_to_track=len(seen), reverse=reverse
            ):
                masks = video.post_process_masks(
                    [step.pred_masks], original_sizes=[list(labels.shape[1:])], binarize=True
                )[0]
                area = lambda t: -int(masks[t[0], 0].sum()) if order == "area" else t[1]
                for index, i in sorted(enumerate(state.obj_ids), key=area):
                    labels[seen[step.frame_idx]][masks[index, 0].cpu().numpy()] = i
    for f in range(len(frames)):  # the frames between: their nearest tracked one's
        labels[f] = labels[seen[int(np.abs(np.array(seen) - f).argmin())]]
    for f, label in enumerate(labels):
        png = Image.fromarray(label)
        png.putpalette(PALETTE)
        png.save(out / f"{f:05d}.png")
    (out / "objects.json").write_text(json.dumps(found, indent=2) + "\n")


if __name__ == "__main__":
    serve() if sys.argv[1] == "serve" else segment(*sys.argv[2:])


def mask_centre(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def join_tracks(first: Path, last: Path, out: Path, frame: int = 0, add: tuple[str, ...] = (), quarter: float = 0.25) -> dict[int, int | None]:  # fmt: skip
    """Per-frame masks `out` from two SAM 3 trackings of the same prompts, one detected at `frame` (`first`), one at
    the clip's last frame (`last`: it re-finds an object the first loses on the way, where it is set down): each of
    the first's instances joined with the last's of its prompt it overlaps most (IoU at least 0.5 in a frame where both
    keep a `quarter` of their area at their own detection frame; else, of a prompt with one instance lost by the end
    and one lost back at `frame`, those two). Each frame takes the one detected nearer to it where that one keeps a
    `quarter` of its area, else the other. One of the last's without a partner, of a prompt in `add` (its noun, before
    any "|"), is kept under a new id when `frame` shows it (20 pixels). Returns {first's id or new id: the last's id}."""
    from PIL import Image

    read = lambda d: (json.loads((d / "objects.json").read_text()), [np.array(Image.open(p)) for p in sorted(d.glob("*.png"))])  # fmt: skip
    (a_found, a), (b_found, b) = read(Path(first)), read(Path(last))
    a, b, n = np.stack(a), np.stack(b), len(a)
    area = lambda labels, i: (labels == i).sum((1, 2))
    alive = lambda labels, i, f: area(labels, i) >= quarter * max(area(labels[f : f + 1], i)[0], 1)
    joined, used = {}, set()
    for i, v in a_found.items():
        live_a = alive(a, int(i), frame)
        best = (0.5, None)
        for j, u in b_found.items():
            if u["prompt"] == v["prompt"] and j not in used:
                both = np.flatnonzero(live_a & alive(b, int(j), n - 1))
                iou = max((((a[f] == int(i)) & (b[f] == int(j))).sum() / ((a[f] == int(i)) | (b[f] == int(j))).sum() for f in both), default=0)  # fmt: skip
                best = max(best, (iou, j), key=lambda t: t[0])
        joined[int(i)] = best[1]
        used.add(best[1])
    for prompt in {v["prompt"] for v in a_found.values()}:  # lost both ways: one of each, the same object
        lost_a = [int(i) for i, v in a_found.items() if v["prompt"] == prompt and joined[int(i)] is None and not alive(a, int(i), frame)[-1]]  # fmt: skip
        lost_b = [j for j, u in b_found.items() if u["prompt"] == prompt and j not in used and not alive(b, int(j), n - 1)[frame]]  # fmt: skip
        if len(lost_a) == 1 and len(lost_b) == 1:
            joined[lost_a[0]] = lost_b[0]
            used.add(lost_b[0])
    labels, found, top, new = a.copy(), dict(a_found), max(map(int, a_found), default=0), set()
    for j, u in b_found.items():  # seen at the first frame without a partner: a new instance
        if j not in used and u["prompt"].split("|")[0] in add and (b[frame] == int(j)).sum() >= 20:
            top += 1
            joined[top], found[str(top)] = j, dict(u)
            new.add(top)
    for i, j in joined.items():  # the last's frames: its half where it is seen, the other where the first's is lost
        if j is not None:
            mine = np.zeros(n, bool) if i in new else alive(a, i, frame)
            for f in np.flatnonzero(alive(b, int(j), n - 1) & ((np.arange(n) >= (frame + n - 1) / 2) | ~mine)):
                labels[f][labels[f] == i] = 0
                labels[f][b[f] == int(j)] = i
    Path(out).mkdir(parents=True, exist_ok=True)
    for f, label in enumerate(labels):
        png = Image.fromarray(label)
        png.putpalette(Image.open(sorted(Path(first).glob("*.png"))[0]).getpalette())
        png.save(Path(out) / f"{f:05d}.png")
    (Path(out) / "objects.json").write_text(json.dumps(found, indent=2) + "\n")
    return joined


def keep_moved(masks: Path, keep: dict[str, int], frame: int = 0) -> dict[int, float]:
    """Of each prompt in `keep` (its noun, before any "|") tracked in a per-frame masks folder, the `keep[noun]`
    instances that move farthest from where they are at `frame` (the 90th percentile, over frames where it keeps a
    quarter of its area there, of its mask centre's distance, in object scales as score_rollout's); the rest made
    background in every frame and dropped from objects.json. Returns each instance's distance."""
    from PIL import Image

    found, paths = json.loads((masks / "objects.json").read_text()), sorted(masks.glob("*.png"))
    labels = np.stack([np.array(Image.open(p)) for p in paths])
    far = {}
    for k, v in found.items():
        if v["prompt"].split("|")[0] in keep:
            first = labels[frame] == int(k)
            if first.sum() < 20:
                far[int(k)] = 0.0
                continue
            ys, xs = np.nonzero(first)
            scale = max(0.5 * np.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1), 0.02 * np.hypot(*first.shape))
            seen = [c for m in labels == int(k) if m.sum() >= 0.25 * first.sum() and (c := mask_centre(m)) is not None]
            far[int(k)] = float(np.percentile(np.linalg.norm(np.array(seen) - mask_centre(first), axis=1), 90) / scale)
    drop = [k for noun, n in keep.items() for k in sorted((int(i) for i, v in found.items() if v["prompt"].split("|")[0] == noun), key=far.get, reverse=True)[n:]]  # fmt: skip
    if drop:
        labels[np.isin(labels, drop)] = 0
        for p, label in zip(paths, labels):
            old, png = Image.open(p), Image.fromarray(label)
            png.putpalette(old.getpalette()) if old.mode == "P" else None
            png.save(p)
        (masks / "objects.json").write_text(json.dumps({k: v for k, v in found.items() if int(k) not in drop}, indent=2) + "\n")  # fmt: skip
    return far
