"""Compare detector scores for the old and new tensor input path.

Usage: python3 compare_detector_scores.py <frigate_checkout> <model.tflite> [mode]

mode is "new" (default), "shift" (old path with the region moved by 2px) or
"noise" (old path plus sigma 1 noise). The last two show how much the model
output moves for changes that happen between any two frames anyway.

The models used were pose_detection.tflite (a person detector) and
face_detection_full_range_sparse.tflite from the mediapipe 0.10.14 wheel on
PyPI, the photos are the bundled scikit-image samples, and inference runs
on OpenVINO CPU.
"""

import sys

import cv2
import numpy as np
import openvino as ov
import skimage.data as data

repo, model_path = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "new"
sys.path.insert(0, repo)

from frigate.util.image import (  # noqa: E402
    yuv_region_2_color_downscaled,
    yuv_region_2_rgb,
)

model = ov.Core().compile_model(model_path, "CPU")
size = model.inputs[0].get_shape()[1]
score_output = [o for o in model.outputs if o.get_partial_shape()[2].get_length() == 1][
    0
]
images = ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


diffs: list[float] = []
flips = positives = regions = 0
for name in images:
    rgb = cv2.resize(getattr(data, name)()[..., :3], (1280, 720), cv2.INTER_CUBIC)
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)

    for side in (300, 400, 520, 640, 720):
        for x in range(-40, 1280 - side + 41, 160):
            for y in (-20, 0, 720 - side):
                region = (x, y, x + side, y + side)
                old = cv2.resize(yuv_region_2_rgb(yuv, region), (size, size))

                if mode == "new":
                    new = yuv_region_2_color_downscaled(
                        yuv, region, size, size, cv2.COLOR_YUV2RGB_I420
                    )
                elif mode == "shift":
                    moved = (x + 2, y, x + side + 2, y + side)
                    new = cv2.resize(yuv_region_2_rgb(yuv, moved), (size, size))
                else:
                    rng = np.random.default_rng(abs(x) * 100000 + abs(y) * 10 + side)
                    new = np.clip(old + rng.normal(0, 1, old.shape), 0, 255)
                    new = new.astype(np.uint8)

                a, b = (
                    sigmoid(
                        model((img.astype(np.float32) / 127.5 - 1)[None])[
                            score_output
                        ].ravel()
                    )
                    for img in (old, new)
                )
                candidates = (a > 0.3) | (b > 0.3)
                diffs.extend(np.abs(a - b)[candidates].tolist())
                flips += int(((a > 0.5) != (b > 0.5)).sum())
                positives += int((a > 0.5).sum())
                regions += 1

d = np.array(diffs)
print(
    f"[{mode}] {regions} regions, score change of candidates: mean {d.mean():.3f} "
    f"p99 {np.percentile(d, 99):.3f}; {flips} of {positives} detections "
    "crossed 0.5"
)
