# CPU optimizations for Intel NUC 12 (i7-1260P, Iris Xe)

Each optimization lives on its own branch based on the same commit
(`74a5973`), so it can be reviewed and measured on its own. The branch
`claude/fervent-newton-nm1cvc` combines all of them and holds these scripts.

| Branch | Change | Output |
| --- | --- | --- |
| `claude/fervent-newton-nm1cvc-motion-percentile` | Motion contrast percentiles from a histogram instead of `np.percentile` | bit identical |
| `claude/fervent-newton-nm1cvc-motion-contrast-lut` | Contrast stretch through a 256 entry lookup table | bit identical |
| `claude/fervent-newton-nm1cvc-motion-blur` | scipy `gaussian_filter` replaced by an exact OpenCV/numpy version | bit identical |
| `claude/fervent-newton-nm1cvc-tensor-input` | Detection regions are downscaled in YUV before color conversion | about 1 level per channel |
| `claude/fervent-newton-nm1cvc-openvino-cache` | Compiled OpenVINO models are cached between starts | identical |

## Results

Measured in a 4 core x86 container with Python 3.11, numpy 1.26, OpenCV 4.11,
scipy 1.17 and OpenVINO 2026.4 (the versions in the Docker image), single
threaded like the camera processes, as thread CPU time, best of 3 x 9 runs.
Absolute numbers on the i7-1260P P-cores will differ, the ratios should be
similar.

### Motion detection (`bench_motion.py`, 1280x720, 400 frames)

| Branch | No mask | With mask |
| --- | --- | --- |
| base | 605 us/frame | 591 us/frame |
| motion-percentile | 307 us (1.97x) | 320 us (1.85x) |
| motion-contrast-lut | 550 us (1.10x) | 604 us (within noise) |
| motion-blur | 543 us (1.11x) | 578 us (1.02x) |
| all combined | 224 us (2.70x) | 230 us (2.57x) |

Every branch produced exactly the same motion boxes and background model
(hash of `avg_frame` after every frame) as the base. The LUT and blur
changes save 30 to 70 us each, which is close to the noise of a single
measurement but shows clearly once the percentile cost is gone (307 to 224 us).

### Tensor input (`bench_tensor.py`, average over regions larger than the model)

| Model size | base | tensor-input |
| --- | --- | --- |
| 300x300 (default OpenVINO SSDLite) | 399 us | 149 us (2.7x) |
| 320x320 | 403 us | 164 us (2.5x) |
| 640x640 (YOLO) | 1018 us | 558 us (1.8x) |

Regions the size of the model or smaller use the unchanged path.

Effect on detections (`compare_detector_scores.py`, person detector, 450
regions over 5 photos): the mean score change of candidates is 0.023 with
the new path, 0.026 for sigma 1 sensor noise and 0.045 for moving the region
by 2 pixels. The change is within what happens between two frames anyway.

### OpenVINO startup (`bench_openvino_start.py`, CPU, fresh process)

| Model | base | cache empty | cache filled |
| --- | --- | --- | --- |
| person detector | ~310 ms | ~330 ms | ~195 ms |
| face detector | ~330 ms | ~360 ms | ~180 ms |
| YOLO-like conv stack | ~255 ms | ~255 ms | ~190 ms |

The CPU plugin compiles quickly, so the gain here is small. The cache matters
on the Iris Xe GPU and on NPUs, where every start rebuilds all OpenCL kernels;
no GPU was available to measure that.
