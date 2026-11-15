"""Benchmark ImprovedMotionDetector.detect on a synthetic 1280x720 stream.

Usage: python3 bench_motion.py <frigate_checkout> <out.json> [--mask]

Runs single threaded, like a camera's tracking process, and measures thread
CPU time so other load on the machine affects the result as little as
possible. The json output also holds every motion box and a hash of the
background model, so two checkouts can be checked for identical results.
"""

import hashlib
import json
import sys
import time

repo = sys.argv[1]
out = sys.argv[2]
use_mask = "--mask" in sys.argv
sys.path.insert(0, repo)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from frigate.config.camera.motion import MotionConfig  # noqa: E402
from frigate.motion.improved_motion import ImprovedMotionDetector  # noqa: E402

cv2.setNumThreads(1)

WIDTH, HEIGHT = 1280, 720
FRAMES = 400
RUNS = 9
rng = np.random.default_rng(42)

# textured background
background = cv2.GaussianBlur(
    rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8), (0, 0), 6
)
background = cv2.normalize(background, None, 30, 220, cv2.NORM_MINMAX)

frames = []
for i in range(FRAMES):
    y = background.astype(np.int16)
    # a person and a car moving through the scene
    person_x = 100 + (i * 6) % (WIDTH - 200)
    cv2.rectangle(y, (person_x, 300), (person_x + 60, 480), 240, -1)
    car_x = WIDTH - 300 - (i * 9) % (WIDTH - 400)
    cv2.rectangle(y, (car_x, 500), (car_x + 220, 620), 15, -1)
    # slow lighting drift and sensor noise
    y = y + int(10 * np.sin(i / 40)) + rng.normal(0, 3, (HEIGHT, WIDTH))
    # an IR switch or lightning flash
    if 250 <= i < 255:
        y = y + 80
    y = np.clip(y, 0, 255).astype(np.uint8)
    uv = np.full((HEIGHT // 2, WIDTH), 128, np.uint8)
    frames.append(np.vstack([y, uv]))

config = MotionConfig()
mask = np.full((HEIGHT, WIDTH), 255, np.uint8)
if use_mask:
    # a timestamp and a tree, like a typical config
    cv2.rectangle(mask, (0, 0), (500, 60), 0, -1)
    cv2.fillPoly(mask, [np.array([[900, 0], [1280, 0], [1280, 300], [1000, 250]])], 0)
object.__setattr__(config, "rasterized_mask", mask)


def run() -> tuple[list, float]:
    detector = ImprovedMotionDetector((HEIGHT, WIDTH), config, fps=5)
    results: list = []
    background_hash = hashlib.sha1()
    elapsed = 0
    for frame in frames:
        start = time.thread_time_ns()
        boxes = detector.detect(frame)
        elapsed += time.thread_time_ns() - start
        results.append(boxes)
        background_hash.update(detector.avg_frame.tobytes())
    results.append(background_hash.hexdigest())
    return results, elapsed / len(frames) / 1000


run()  # warm up
timings = []
for _ in range(RUNS):
    results, us_per_frame = run()
    timings.append(us_per_frame)

json.dump({"us_per_frame": min(timings), "results": results}, open(out, "w"))
print(f"{repo}: {min(timings):.0f} us per frame")
