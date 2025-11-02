import os
import unittest

import yaml

from fluxbind.graph.graph import HwlocTopology

# Assuming your classes are in these files. Adjust the import paths if needed.
from fluxbind.graph.shape import Shape, TopologyResult

here = os.path.dirname(os.path.abspath(__file__))

TEST_XML_FILE = os.path.join(here, "single-node.xml")
TEST_GPU_XML_FILE = os.path.join(here, "corona.xml")


class TestShapeAllocator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This method runs once before any tests. It ensures our test XML file exists.
        """
        if not os.path.exists(TEST_XML_FILE):
            raise FileNotFoundError(
                f"Test XML file not found: {TEST_XML_FILE}. Please generate it first."
            )

    def run_test_case(self, shape, local_rank: int, local_size: int) -> TopologyResult:
        """
        A helper function to run the allocator for a single test case.
        It writes a temporary YAML file and calls the Shape.run() method.
        """
        # Instantiate the Shape class with our temporary file
        shape_allocator = Shape(yaml.load(shape, Loader=yaml.SafeLoader))

        # Call the core run logic directly, simulating a flux task
        result = shape_allocator.run(
            local_size=local_size,
            local_rank=local_rank,
            xml_file=TEST_XML_FILE,
        )
        return result

    def test_01_simple_cores_multi_rank(self):
        """
        Test: Ask for a total pool of 8 cores, to be divided among 2 ranks.
        This tests the core "pool division" logic.
        """
        print("\n--- Testing: Simple Core Pool (Multi-Rank) ---")
        shape_yaml = """
resources:
  - type: core
    count: 8
"""
        # --- Simulate Rank 0 ---
        # We expect this rank to get the first 4 cores (0, 1, 2, 3)
        # Your machine has SMT, so Core 0 = PU 0,1; Core 1 = PU 2,3; etc.
        # The expected mask for binding to the first PU of each core is calculated manually.
        # Core 0 -> PU 0 -> cpuset 0x1
        # Core 1 -> PU 2 -> cpuset 0x4
        # Core 2 -> PU 4 -> cpuset 0x10
        # Core 3 -> PU 6 -> cpuset 0x40
        # The combined mask is 0x1 + 0x4 + 0x10 + 0x40 = 0x55
        result_rank0 = self.run_test_case(shape_yaml, local_rank=0, local_size=2)
        self.assertEqual(result_rank0.mask, "0x00000055")
        self.assertEqual(len(result_rank0.nodes), 4)

        # --- Simulate Rank 1 ---
        # We expect this rank to get the next 4 cores (4, 5, 6, 7)
        # Core 4 -> PU 8  -> 0x100
        # Core 5 -> PU 10 -> 0x400
        # Core 6 -> PU 12 -> 0x1000
        # Core 7 -> PU 14 -> 0x4000
        # The combined mask is 0x100 + 0x400 + 0x1000 + 0x4000 = 0x5500
        result_rank1 = self.run_test_case(shape_yaml, local_rank=1, local_size=2)
        self.assertEqual(result_rank1.mask, "0x00005500")
        self.assertEqual(len(result_rank1.nodes), 4)

    def test_02_explicit_bind_pu_multi_rank(self):
        """
        Test: Ask for a total pool of 4 cores, bind to PUs, divide among 2 ranks.
        This tests Rule 1 (explicit override) and SMT-aware division.
        """
        print("\n--- Testing: Explicit `bind: pu` (Multi-Rank) ---")
        shape_yaml = """
options:
  bind: pu
resources:
  - type: core
    count: 4
"""
        # The total pool of resources will be all PUs on the first 4 cores.
        # This is 8 PUs total (PU 0,1,2,3,4,5,6,7).

        # --- Simulate Rank 0 ---
        # We expect this rank to get the first 4 PUs (0, 1, 2, 3)
        # The mask for these is 0x1 | 0x2 | 0x4 | 0x8 = 0xF
        result_rank0 = self.run_test_case(shape_yaml, local_rank=0, local_size=2)
        self.assertEqual(result_rank0.mask, "0x0000000f")
        self.assertEqual(len(result_rank0.nodes), 4)

        # --- Simulate Rank 1 ---
        # We expect this rank to get the next 4 PUs (4, 5, 6, 7)
        # The mask for these is 0x10 | 0x20 | 0x40 | 0x80 = 0xF0
        result_rank1 = self.run_test_case(shape_yaml, local_rank=1, local_size=2)
        self.assertEqual(result_rank1.mask, "0x000000f0")
        self.assertEqual(len(result_rank1.nodes), 4)

    def test_03_implicit_bind_core(self):
        """
        Test: Ask for 2 cores. Implicitly, this should bind to core.
        """
        print("\n--- Testing: Implicit `bind: core` (Rule 2) ---")
        shape_yaml = """
resources:
  - type: core
    count: 2
"""
        # We expect Core:0 and Core:1.
        # Binding to core means first PU of each: PU:0 (0x1) and PU:2 (0x4).
        # Mask = 0x1 | 0x4 = 0x5
        result = self.run_test_case(shape_yaml, local_rank=0, local_size=1)
        self.assertEqual(result.mask, "0x00000005")

    def test_04_default_bind_core_from_container(self):
        """
        Test: Ask for a container (l3cache) with no binding preference.
        This should fall back to the default 'core' binding.
        """
        print("\n--- Testing: Default `bind: core` from Container (Rule 3) ---")
        shape_yaml = """
resources:
  - type: l3cache
    count: 1
"""
        # We expect all 8 cores from the single L3 cache.
        # Binding to core means the first PU of all 8 cores.
        # Mask = 0x5555 (calculated from previous tests: 0x5500 | 0x55)
        result = self.run_test_case(shape_yaml, local_rank=0, local_size=1)
        self.assertEqual(result.mask, "0x00005555")
        self.assertEqual(len(result.nodes), 8)

    def test_05_bind_none(self):
        """
        Test: Ask for resources but specify no binding.
        """
        print("\n--- Testing: `bind: none` ---")
        shape_yaml = """
options:
  bind: none
resources:
  - type: core
    count: 2
"""
        result = self.run_test_case(shape_yaml, local_rank=0, local_size=1)
        self.assertEqual(result.mask, "UNBOUND")
        self.assertIsNotNone(result.nodes)

    def test_06_pattern_scatter_multi_rank(self):
        """
        Test: Distribute a pool of 8 cores among 4 ranks using a scatter pattern.
        """
        print("\n--- Testing: `pattern: scatter` (Multi-Rank) ---")
        shape_yaml = """
resources:
  - type: core
    count: 8
    pattern: scatter
"""
        # Total pool: [C0, C1, C2, C3, C4, C5, C6, C7]
        # local_size=4, so items_per_rank = 8 // 4 = 2

        # Rank 0 should get strided cores: C0, C4
        # Mask = PU0(0x1) | PU8(0x100) = 0x101
        result_rank0 = self.run_test_case(shape_yaml, local_rank=0, local_size=4)
        self.assertEqual(result_rank0.mask, "0x00000101")
        self.assertEqual(len(result_rank0.nodes), 2)

        # Rank 1 should get strided cores: C1, C5
        # Mask = PU2(0x4) | PU10(0x400) = 0x404
        result_rank1 = self.run_test_case(shape_yaml, local_rank=1, local_size=4)
        self.assertEqual(result_rank1.mask, "0x00000404")
        self.assertEqual(len(result_rank1.nodes), 2)

    def test_07_pattern_reverse_multi_rank(self):
        """
        Test: Distribute a pool of 8 cores among 2 ranks, packed but in reverse order.
        """
        print("\n--- Testing: `reverse: true` (Multi-Rank) ---")
        shape_yaml = """
resources:
  - type: core
    count: 8
    reverse: true
"""
        # Pool is reversed: [C7, C6, C5, C4, C3, C2, C1, C0]
        # local_size=2, so items_per_rank = 8 // 2 = 4

        # Rank 0 gets the first 4 from the reversed list: C7, C6, C5, C4
        # This is the same as rank 1 in the normal packed test. Mask = 0x5500
        result_rank0 = self.run_test_case(shape_yaml, local_rank=0, local_size=2)
        self.assertEqual(result_rank0.mask, "0x00005500")
        self.assertEqual(len(result_rank0.nodes), 4)

        # Rank 1 gets the next 4 from the reversed list: C3, C2, C1, C0
        # This is the same as rank 0 in the normal packed test. Mask = 0x55
        result_rank1 = self.run_test_case(shape_yaml, local_rank=1, local_size=2)
        self.assertEqual(result_rank1.mask, "0x00000055")
        self.assertEqual(len(result_rank1.nodes), 4)


@unittest.skipUnless(
    os.path.exists(TEST_GPU_XML_FILE), f"Skipping GPU tests, {TEST_GPU_XML_FILE} not found."
)
class TestGpuShapeAllocator(unittest.TestCase):
    """
    Tests for GPU-aware allocation, requiring a multi-NUMA topology.
    These tests will use the SERVER_XML_FILE (corona.xml).
    """

    def run_test_case(
        self, shape_yaml_str: str, local_rank: int, local_size: int, **kwargs
    ) -> TopologyResult:
        shape_allocator = Shape(yaml.safe_load(shape_yaml_str))
        result = shape_allocator.run(
            xml_file=TEST_GPU_XML_FILE, local_size=local_size, local_rank=local_rank, **kwargs
        )
        return result

    def test_01_gpu_local_binding(self):
        """
        Test: Assign 1 GPU per task and bind to 2 local cores.
        Verify that ranks targeting NUMA 0 get cores on Package 0, and
        ranks targeting NUMA 1 get cores on Package 1.
        """
        print("\n--- Testing: `bind: gpu-local` (Multi-NUMA) ---")
        shape_yaml = """
options:
  bind: gpu-local
  bind_to: core # This will be the default, but explicit is good for tests
resources:
  # This section defines the CPU resources to find within the GPU-local domain
  - type: core
    count: 2
"""
        # Based on corona.xml, there are 8 GPUs. Ranks 0-3 should get GPUs on NUMA 0.
        # Ranks 4-7 should get GPUs on NUMA 1.

        # --- Simulate Rank 1 (targets a GPU on NUMA 0) ---
        result_rank1 = self.run_test_case(shape_yaml, local_rank=1, local_size=8, gpus_per_task=1)

        # It should get the second GPU, which is on NUMA 0.
        self.assertEqual(result_rank1.gpu_string, "1")
        # It should be assigned 2 cores.
        self.assertEqual(len(result_rank1.nodes), 2)
        # Verify both assigned cores are on Package 0.
        for gp, data in result_rank1.nodes:
            package = result_rank1.topo.get_ancestor_of_type(gp, "Package")
            self.assertIsNotNone(package)
            self.assertEqual(
                package[1].get("os_index"),
                0,
                f"Core {data.get('os_index')} is on the wrong package!",
            )

        # --- Simulate Rank 5 (targets a GPU on NUMA 1) ---
        result_rank5 = self.run_test_case(shape_yaml, local_rank=5, local_size=8, gpus_per_task=1)

        # It should get the sixth GPU, which is on NUMA 1.
        self.assertEqual(result_rank5.gpu_string, "5")
        self.assertEqual(len(result_rank5.nodes), 2)
        # Verify both assigned cores are on Package 1.
        for gp, data in result_rank5.nodes:
            package = result_rank5.topo.get_ancestor_of_type(gp, "Package")
            self.assertIsNotNone(package)
            self.assertEqual(
                package[1].get("os_index"),
                1,
                f"Core {data.get('os_index')} is on the wrong package!",
            )

    def test_02_gpu_remote_binding(self):
        """
        Test: Assign a GPU on NUMA 0, but bind to cores on NUMA 1.
        """
        print("\n--- Testing: `bind: gpu-remote` (Multi-NUMA) ---")
        shape_yaml = """
options:
  bind: gpu-remote
  bind_to: core
resources:
  - type: core
    count: 2
"""
        # --- Simulate Rank 0 (targets a GPU on NUMA 0) ---
        result_rank0 = self.run_test_case(shape_yaml, local_rank=0, local_size=8, gpus_per_task=1)

        # It should get the first GPU.
        self.assertEqual(result_rank0.gpu_string, "0")
        # It should be assigned 2 cores.
        self.assertEqual(len(result_rank0.nodes), 2)
        # Verify both assigned cores are on the REMOTE package (Package 1).
        for gp, data in result_rank0.nodes:
            package = result_rank0.topo.get_ancestor_of_type(gp, "Package")
            self.assertIsNotNone(package)
            self.assertEqual(
                package[1].get("os_index"),
                1,
                f"Core {data.get('os_index')} was bound locally, not remotely!",
            )

    def test_03_gpu_remote_fails_on_single_numa(self):
        """
        Test: Ensure gpu-remote raises an error on a single-NUMA machine.
        """
        print("\n--- Testing: `bind: gpu-remote` (Single-NUMA, Expect Fail) ---")
        shape_yaml = """
options:
  bind: gpu-remote
resources:
  - type: core
    count: 1
"""

        # We need a separate runner for this test that uses the laptop XML.
        def run_on_laptop():
            shape_allocator = Shape(yaml.safe_load(shape_yaml))
            shape_allocator.run(xml_file=TEST_XML_FILE, local_size=1, local_rank=0, gpus_per_task=1)

        with self.assertRaisesRegex(RuntimeError, "Cannot find a remote NUMA node"):
            run_on_laptop()


if __name__ == "__main__":
    unittest.main()
