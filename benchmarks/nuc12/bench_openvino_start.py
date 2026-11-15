"""Time how long OpenVINOModelRunner takes to load a model in a fresh process.

Usage: python3 bench_openvino_start.py <frigate_checkout> <model.xml> [device]

Run it repeatedly: with the model cache the first run compiles and stores
the model, later runs load the compiled blob.
"""

import sys
import time

repo, model = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "CPU"
sys.path.insert(0, repo)

from frigate.detectors.detection_runners import OpenVINOModelRunner  # noqa: E402

start = time.perf_counter()
OpenVINOModelRunner(model, device, "ssd")
print(f"{(time.perf_counter() - start) * 1000:.0f} ms")
