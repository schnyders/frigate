from unittest import TestCase, main

import cv2
import numpy as np

from frigate.util.image import yuv_region_2_color_downscaled, yuv_region_2_rgb


class TestYuvRegion2RGB(TestCase):
    def setUp(self):
        self.bgr_frame = np.zeros((100, 200, 3), np.uint8)
        self.bgr_frame[:] = (0, 0, 255)
        self.bgr_frame[5:55, 5:55] = (255, 0, 0)
        # cv2.imwrite(f"bgr_frame.jpg", self.bgr_frame)
        self.yuv_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2YUV_I420)

    def test_crop_yuv(self):
        cropped = yuv_region_2_rgb(self.yuv_frame, (10, 10, 50, 50))
        # ensure the upper left pixel is blue
        assert np.all(cropped[0, 0] == [0, 0, 255])

    def test_crop_yuv_out_of_bounds(self):
        cropped = yuv_region_2_rgb(self.yuv_frame, (0, 0, 200, 200))
        # cv2.imwrite(f"cropped.jpg", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        # ensure the upper left pixel is red
        # the yuv conversion has some noise
        assert np.all(cropped[0, 0] == [255, 1, 0])
        # ensure the bottom right is black
        assert np.all(cropped[199, 199] == [0, 0, 0])

    def test_crop_yuv_portrait(self):
        bgr_frame = np.zeros((1920, 1080, 3), np.uint8)
        bgr_frame[:] = (0, 0, 255)
        bgr_frame[5:55, 5:55] = (255, 0, 0)
        # cv2.imwrite(f"bgr_frame.jpg", self.bgr_frame)
        yuv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)

        yuv_region_2_rgb(yuv_frame, (0, 852, 648, 1500))
        # cv2.imwrite(f"cropped.jpg", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))


class TestYuvRegion2ColorDownscaled(TestCase):
    def setUp(self):
        # smooth synthetic scene with muted colors, like a real camera frame
        rng = np.random.default_rng(0)
        bgr_frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        bgr_frame = cv2.GaussianBlur(bgr_frame, (0, 0), 4)
        self.yuv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)

    def _convert_then_resize(self, region, size):
        rgb = yuv_region_2_rgb(self.yuv_frame, region)
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    def test_matches_convert_then_resize(self):
        """Downscaling in YUV first only differs from the old path by rounding."""
        for region in [
            (200, 48, 720, 568),
            (400, 0, 1120, 720),
            (0, 0, 640, 640),
            (-40, -40, 600, 600),
            (900, 200, 1400, 700),
        ]:
            for size in (320, 300):
                expected = self._convert_then_resize(region, size)
                result = yuv_region_2_color_downscaled(
                    self.yuv_frame, region, size, size, cv2.COLOR_YUV2RGB_I420
                )

                self.assertEqual(result.shape, expected.shape)
                diff = np.abs(result.astype(int) - expected.astype(int))
                self.assertLess(diff.mean(), 1.5, region)
                self.assertLessEqual(np.percentile(diff, 99), 6, region)

    def test_upscale_falls_back(self):
        """Regions smaller than the model are left to the convert first path."""
        self.assertIsNone(
            yuv_region_2_color_downscaled(
                self.yuv_frame, (0, 0, 200, 200), 320, 320, cv2.COLOR_YUV2RGB_I420
            )
        )


if __name__ == "__main__":
    main(verbosity=2)
