"""Benchmark create_tensor_input for typical regions of a 1280x720 stream.

Usage: python3 bench_tensor.py <frigate_checkout> <model_size>

Prints json with the best of 5x50 calls per region, in microseconds of
thread CPU time.
"""

import json
import os
import sys
import time

repo, size = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, repo)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from frigate.detectors.detector_config import ModelConfig  # noqa: E402
from frigate.util.object import create_tensor_input  # noqa: E402

cv2.setNumThreads(1)

model_config = ModelConfig(
    width=size, height=size, labelmap_path=os.path.join(repo, "labelmap.txt")
)
rng = np.random.default_rng(0)
bgr = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8), (0, 0), 3)
frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

# regions are square, a multiple of 4 and at least the model size
regions = []
for side in (size, int(size * 1.25), int(size * 1.5), int(size * 2), 720):
    side = min(side, 720) // 4 * 4
    for x in (0, 200, 1280 - side):
        regions.append((x, 0, x + side, side))
side = size // 4 * 4
regions.append((-40, -40, side + 40, side + 40))

per_region = {}
for region in regions:
    best = float("inf")
    for _ in range(5):
        start = time.thread_time_ns()
        for _ in range(50):
            create_tensor_input(frame, model_config, region)
        best = min(best, (time.thread_time_ns() - start) / 50 / 1000)
    per_region[f"{region[2] - region[0]}px@{region[0]}"] = round(best)

print(
    json.dumps({"avg": sum(per_region.values()) / len(per_region), "per": per_region})
)
