import unittest

import build_k230_a4_recognition_test as builder
import k230_a4_recognition_test as entrypoint


class A4RecognitionEntrypointTests(unittest.TestCase):
    def test_corner_and_rejection_formatting(self):
        self.assertEqual(
            entrypoint._format_corners(
                [(1.2, 2.8), (10.0, 20.0)]
            ),
            "1:3|10:20",
        )
        self.assertEqual(
            entrypoint._format_rejections(
                {"rejected": {"side": 2, "area": 1}}
            ),
            "area:1|side:2",
        )

    def test_divider_uses_physical_a4_corner_order(self):
        points = entrypoint._divider_points(
            [(0.0, 0.0), (210.0, 0.0), (210.0, 297.0), (0.0, 297.0)],
            148.5,
        )
        self.assertEqual(points, ((0.0, 148.5), (210.0, 148.5)))

    def test_standalone_contains_only_a4_runtime(self):
        source = builder.build_source()
        compile(source, str(builder.OUTPUT), "exec")
        self.assertIn("class A4BoundaryTracker", source)
        self.assertIn("def detect_a4_boundary", source)
        self.assertIn("def main():", source)
        self.assertNotIn("puzzle_vision", source)
        self.assertNotIn("plan_rectangle_assembly", source)
        self.assertNotIn("import puzzle_config", source)
        self.assertNotIn("from puzzle_a4_boundary", source)
        self.assertNotIn("cfg.", source)


if __name__ == "__main__":
    unittest.main()
