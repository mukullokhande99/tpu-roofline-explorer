import math
import tempfile
import unittest
from pathlib import Path

from roofline import (
    Architecture,
    GemmWorkload,
    estimate_hbm_traffic,
    evaluate,
    mxu_utilization,
    plot_roofline,
    precision_bytes,
    total_operations,
)


class RooflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arch = Architecture("test", 1.0, 100.0, 1.0, 4, 4)
        self.workload = GemmWorkload("gemm", 8, 8, 4, "fp32", 2, 2)

    def test_precision_sizes(self) -> None:
        expected = {
            "posit-(4,1)": 0.5,
            "fp2": 0.25,
            "int4": 0.5,
            "int8": 1.0,
            "bf16": 2.0,
            "fp32": 4.0,
        }
        for precision, byte_count in expected.items():
            self.assertEqual(precision_bytes(precision), byte_count)

    def test_operations(self) -> None:
        workload = GemmWorkload("gemm", 4, 4, 4, "bf16")
        self.assertEqual(total_operations(workload), 128)

    def test_reuse_reduces_reload_traffic_but_not_below_compulsory(self) -> None:
        no_reuse = GemmWorkload("none", 8, 8, 4, "fp32", 1, 1)
        full_reuse = GemmWorkload("full", 8, 8, 4, "fp32", 100, 100)
        no_reuse_traffic = estimate_hbm_traffic(no_reuse, self.arch)
        full_reuse_traffic = estimate_hbm_traffic(full_reuse, self.arch)
        compulsory = (8 * 4 + 4 * 8 + 8 * 8) * 4
        self.assertGreater(no_reuse_traffic.total_bytes, full_reuse_traffic.total_bytes)
        self.assertEqual(full_reuse_traffic.total_bytes, compulsory)

    def test_small_sram_reduces_effective_reuse(self) -> None:
        tiny_arch = Architecture("tiny", 1.0, 100.0, 0.00001, 4, 4)
        traffic = estimate_hbm_traffic(self.workload, tiny_arch)
        self.assertLess(traffic.sram_residency, 1.0)
        self.assertLess(traffic.effective_weight_reuse, 2.0)
        self.assertLess(traffic.effective_activation_reuse, 2.0)

    def test_mxu_utilization_includes_fill_and_drain(self) -> None:
        workload = GemmWorkload("gemm", 4, 4, 4, "bf16")
        expected = 4 / (4 + 4 + 4 - 2)
        self.assertTrue(math.isclose(mxu_utilization(workload, self.arch), expected))

    def test_roofline_latency_is_maximum(self) -> None:
        result = evaluate(self.workload, self.arch)
        self.assertEqual(
            result.estimated_latency_s,
            max(result.compute_bound_latency_s, result.memory_bound_latency_s),
        )

    def test_plot_is_created(self) -> None:
        result = evaluate(self.workload, self.arch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roofline.png"
            plot_roofline([result], path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GemmWorkload("bad", 0, 4, 4)
        with self.assertRaises(ValueError):
            GemmWorkload("bad", 4, 4, 4, "fp16")
        with self.assertRaises(ValueError):
            GemmWorkload("bad", 4, 4, 4, "bf16", 0.5, 1)

    def test_vxu_uses_lane_waves_without_systolic_fill_drain(self) -> None:
        vxu = Architecture("vxu", 0.512, 100.0, 1.0, 1, 256, "vxu", 256)
        mxu = Architecture("mxu", 2.048, 100.0, 1.0, 32, 32)
        workload = GemmWorkload("vector", 1, 1025, 4, "int8")
        self.assertEqual(math.ceil(workload.n / 256), 5)
        self.assertGreater(mxu_utilization(workload, vxu), 0)
        self.assertNotEqual(
            mxu_utilization(workload, vxu), mxu_utilization(workload, mxu)
        )


if __name__ == "__main__":
    unittest.main()
