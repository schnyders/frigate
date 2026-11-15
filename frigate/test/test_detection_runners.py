"""Tests for ONNX Runtime session option selection."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import onnxruntime as ort

from frigate.detectors.detection_runners import (
    CudaGraphRunner,
    get_openvino_cache_dir,
    get_ort_session_options,
)
from frigate.detectors.detector_config import ModelTypeEnum
from frigate.embeddings.types import EnrichmentModelTypeEnum


class TestGetOrtSessionOptions(unittest.TestCase):
    def test_jina_v2_uses_extended(self):
        """jina-clip-v2 returns an identical vector for every image on the CUDA
        execution provider at anything below EXTENDED."""
        options = get_ort_session_options(EnrichmentModelTypeEnum.jina_v2.value)

        self.assertIsNotNone(options)
        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        )

    def test_jina_v1_uses_basic(self):
        options = get_ort_session_options(EnrichmentModelTypeEnum.jina_v1.value)

        self.assertIsNotNone(options)
        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        )

    def test_cpu_enrichment_models_disable_spinning(self):
        for model_type in EnrichmentModelTypeEnum:
            with self.subTest(model_type=model_type):
                options = get_ort_session_options(model_type.value, cpu=True)

                self.assertEqual(
                    options.get_session_config_entry("session.intra_op.allow_spinning"),
                    "0",
                )

    def test_gpu_enrichment_models_keep_spinning(self):
        options = get_ort_session_options(EnrichmentModelTypeEnum.arcface.value)

        with self.assertRaises(RuntimeError):
            options.get_session_config_entry("session.intra_op.allow_spinning")

    def test_arcface_keeps_default_optimization(self):
        options = get_ort_session_options(EnrichmentModelTypeEnum.arcface.value)

        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )

    def test_detector_models_use_defaults(self):
        for model_type in [None, ModelTypeEnum.rfdetr.value]:
            with self.subTest(model_type=model_type):
                self.assertIsNone(get_ort_session_options(model_type, cpu=True))


class TestCudaGraphRunner(unittest.TestCase):
    """CUDA graph capture fails if the arena has to allocate during capture, so
    the session is warmed up with capture disabled before the first real run."""

    def setUp(self):
        self.session = MagicMock()
        self.session.get_outputs.return_value = [MagicMock(name="output")]
        self.io_binding = self.session.io_binding.return_value
        self.input = {"images": np.zeros((1, 3, 320, 320), np.float32)}

    def _annotations(self) -> list[str | None]:
        """Graph annotation id passed with each run, None when unset."""
        annotations = []

        for call in self.session.run_with_iobinding.call_args_list:
            try:
                annotations.append(call.args[1].get_run_config_entry("gpu_graph_id"))
            except RuntimeError:
                annotations.append(None)

        return annotations

    def test_first_run_warms_up_with_capture_disabled(self):
        with patch.object(ort.OrtValue, "ortvalue_from_numpy"):
            CudaGraphRunner(self.session, 0).run(self.input)

        self.assertEqual(
            self._annotations(),
            ["-1"] * CudaGraphRunner.GRAPH_FREE_WARMUP_RUNS + [None],
        )

    def test_later_runs_allow_capture(self):
        with patch.object(ort.OrtValue, "ortvalue_from_numpy"):
            runner = CudaGraphRunner(self.session, 0)
            runner.run(self.input)
            self.session.run_with_iobinding.reset_mock()
            runner.run(self.input)

        self.assertEqual(self._annotations(), [None])
        runner._input_ortvalue.update_inplace.assert_called_once()


class TestOpenVINOCacheDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch(
            "frigate.detectors.detection_runners.OPENVINO_CACHE_DIR",
            os.path.join(self.tmp.name, "cache"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.model_path = os.path.join(self.tmp.name, "model.xml")
        self.weights_path = os.path.join(self.tmp.name, "model.bin")
        self._write(self.model_path, b"<net/>", 1)
        self._write(self.weights_path, b"weights", 1)

    def _write(self, path: str, data: bytes, mtime: int) -> None:
        with open(path, "wb") as f:
            f.write(data)
        os.utime(path, (mtime, mtime))

    def _add_cached_blob(self, cache_dir: str) -> str:
        blob = os.path.join(cache_dir, "123.blob")
        self._write(blob, b"compiled", 1)
        return blob

    def test_unchanged_model_keeps_cache(self):
        cache_dir = get_openvino_cache_dir(self.model_path)
        blob = self._add_cached_blob(cache_dir)

        self.assertEqual(get_openvino_cache_dir(self.model_path), cache_dir)
        self.assertTrue(os.path.exists(blob))

    def test_replaced_weights_clear_cache(self):
        """OpenVINO does not notice new weights, so the stale blob must go."""
        cache_dir = get_openvino_cache_dir(self.model_path)
        blob = self._add_cached_blob(cache_dir)
        self._write(self.weights_path, b"new weights", 2)

        self.assertEqual(get_openvino_cache_dir(self.model_path), cache_dir)
        self.assertFalse(os.path.exists(blob))

    def test_models_get_separate_directories(self):
        other_path = os.path.join(self.tmp.name, "other.onnx")
        self._write(other_path, b"onnx", 1)

        self.assertNotEqual(
            get_openvino_cache_dir(self.model_path),
            get_openvino_cache_dir(other_path),
        )

    def test_missing_model_is_not_cached(self):
        self.assertIsNone(
            get_openvino_cache_dir(os.path.join(self.tmp.name, "missing.onnx"))
        )
