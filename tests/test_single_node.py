import os
import unittest

import yaml

from fluxbind.graph.graphic import TopologyVisualizer
from fluxbind.graph.shape import Shape, TopologyResult

here = os.path.dirname(os.path.abspath(__file__))

TEST_XML_FILE = os.path.join(here, "single-node.xml")
TEST_GPU_XML_FILE = os.path.join(here, "corona.xml")
GRAPHICS_OUTPUT_DIR = os.path.join(here, "img")


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
        os.makedirs(GRAPHICS_OUTPUT_DIR, exist_ok=True)

    def run_test_case(
        self,
        shape,
        local_rank: int,
        local_size: int,
        output_filename: str = None,
        title: str = None,
        gpus_per_task=None,
        **kwargs,
    ) -> TopologyResult:
        """
        A helper function that runs an allocation and optionally generates a graphic.
        """
        shape_allocator = Shape(yaml.safe_load(shape))

        # Use the correct XML file based on kwargs
        xml_file = kwargs.get("xml_file", TEST_XML_FILE)
        result = shape_allocator.run(
            xml_file=xml_file,
            local_size=local_size,
            local_rank=local_rank,
            gpus_per_task=gpus_per_task,
        )

        # If a filename is provided, generate the visualization
        if output_filename and result.nodes and result.topo:
            filepath = os.path.join(GRAPHICS_OUTPUT_DIR, output_filename)
            visualizer = TopologyVisualizer(
                result.topo, result.nodes, affinity_target=result.topo.last_affinity_target
            )
            # We can customize the title for the plot
            visualizer.title = title or f"Allocation for local_rank {local_rank}"
            visualizer.draw(filepath)

        return result

    def test_01_simple_cores_multi_rank(self):
        print("\n--- Testing: Simple Core Pool (Multi-Rank) ---")
        shape_yaml = """
resources:
  - type: core
    count: 8
"""
        result_rank0 = self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=2,
            output_filename="01_simple_cores_rank0.png",
            title="Simple Cores: Rank 0 of 2",
        )
        self.assertEqual(result_rank0.mask, "0x00000055")

        result_rank1 = self.run_test_case(
            shape_yaml,
            local_rank=1,
            local_size=2,
            output_filename="01_simple_cores_rank1.png",
            title="Simple Cores: Rank 1 of 2",
        )
        self.assertEqual(result_rank1.mask, "0x00005500")

    def test_02_explicit_bind_pu_multi_rank(self):
        print("\n--- Testing: Explicit `bind: pu` (Multi-Rank) ---")
        shape_yaml = """
options:
  bind: pu
resources:
  - type: core
    count: 4
"""
        result_rank0 = self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=2,
            output_filename="02_explicit_pu_rank0.png",
            title="Explicit PU Binding: Rank 0 of 2",
        )
        self.assertEqual(result_rank0.mask, "0x0000000f")

        result_rank1 = self.run_test_case(
            shape_yaml,
            local_rank=1,
            local_size=2,
            output_filename="02_explicit_pu_rank1.png",
            title="Explicit PU Binding: Rank 1 of 2",
        )
        self.assertEqual(result_rank1.mask, "0x000000f0")

    def test_03_implicit_bind_core(self):
        print("\n--- Testing: Implicit `bind: core` (Rule 2) ---")
        shape_yaml = "resources:\n  - type: core\n    count: 2"
        self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=1,
            output_filename="03_implicit_core.png",
            title="Implicit Core Binding",
        )

    def test_04_default_bind_core_from_container(self):
        print("\n--- Testing: Default `bind: core` from Container (Rule 3) ---")
        shape_yaml = "resources:\n  - type: l3cache\n    count: 1"
        self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=1,
            output_filename="04_default_core_container.png",
            title="Default Core Binding from L3Cache",
        )

    def test_05_bind_none(self):
        print("\n--- Testing: `bind: none` ---")
        shape_yaml = "options:\n  bind: none\nresources:\n  - type: core\n    count: 2"
        result = self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=1,
            output_filename="05_bind_none.png",
            title="Bind None (shows found resources)",
        )
        self.assertEqual(result.mask, "UNBOUND")

    def test_06_pattern_scatter_multi_rank(self):
        print("\n--- Testing: `pattern: scatter` (Multi-Rank) ---")
        shape_yaml = "resources:\n  - type: core\n    count: 8\n    pattern: scatter"
        self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=4,
            output_filename="06_scatter_rank0.png",
            title="Scatter Pattern: Rank 0 of 4",
        )
        self.run_test_case(
            shape_yaml,
            local_rank=1,
            local_size=4,
            output_filename="06_scatter_rank1.png",
            title="Scatter Pattern: Rank 1 of 4",
        )

    def test_07_pattern_reverse_multi_rank(self):
        print("\n--- Testing: `reverse: true` (Multi-Rank) ---")
        shape_yaml = "resources:\n  - type: core\n    count: 8\n    reverse: true"
        self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=2,
            output_filename="07_reverse_rank0.png",
            title="Reverse Pattern: Rank 0 of 2",
        )
        self.run_test_case(
            shape_yaml,
            local_rank=1,
            local_size=2,
            output_filename="07_reverse_rank1.png",
            title="Reverse Pattern: Rank 1 of 2",
        )

    def test_08_explicit_core_binding_multi_rank(self):
        """
        Test: A multi-rank job where the shape explicitly asks for cores.
        This validates Rule 2 (Implicit Intent) in a distribution scenario.
        """
        print("\n--- Testing: Explicit `bind: core` (Multi-Rank, Rule 2) ---")
        shape_yaml = """
# No options block is provided. The script should infer the binding
# preference from the resource type requested.
resources:
  - type: core
    count: 4
"""
        # We are asking for a total pool of 4 cores, to be divided among 2 ranks.
        # Each rank should get 2 cores.

        # We expect this rank to get the first 2 cores: Core:0, Core:1
        # The binding is implicitly core so we bind to the first PU of each.
        # Mask = PU:0 (0x1) | PU:2 (0x4) = 0x5
        result_rank0 = self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=2,
            output_filename="08_explicit_core_rank0.png",
            title="Explicit Core Binding (2 per rank): Rank 0 of 2",
        )
        self.assertEqual(result_rank0.mask, "0x00000005")
        self.assertEqual(len(result_rank0.nodes), 2)

        # We expect this rank to get the next 2 cores: Core:2, Core:3
        # Binding to the first PU of each.
        # Mask = PU:4 (0x10) | PU:6 (0x40) = 0x50
        result_rank1 = self.run_test_case(
            shape_yaml,
            local_rank=1,
            local_size=2,
            output_filename="08_explicit_core_rank1.png",
            title="Explicit Core Binding (2 per rank): Rank 1 of 2",
        )
        self.assertEqual(result_rank1.mask, "0x00000050")
        self.assertEqual(len(result_rank1.nodes), 2)


@unittest.skipUnless(
    os.path.exists(TEST_GPU_XML_FILE), f"Skipping GPU tests, {TEST_GPU_XML_FILE} not found."
)
class TestGpuShapeAllocator(TestShapeAllocator):  # Inherit the helper
    """
    Tests for GPU-aware allocation, requiring a multi-NUMA topology.
    """

    def test_01_gpu_local_binding(self):
        print("\n--- Testing: `bind: gpu-local` (Multi-NUMA) ---")
        shape_yaml = (
            "options:\n  bind: gpu-local\n  bind_to: core\nresources:\n  - type: core\n    count: 2"
        )

        self.run_test_case(
            shape_yaml,
            local_rank=1,
            local_size=8,
            gpus_per_task=1,
            xml_file=TEST_GPU_XML_FILE,
            output_filename="08_gpu_local_rank1_numa0.png",
            title="GPU Local: Rank 1 (GPU on NUMA 0)",
        )
        self.run_test_case(
            shape_yaml,
            local_rank=5,
            local_size=8,
            gpus_per_task=1,
            xml_file=TEST_GPU_XML_FILE,
            output_filename="08_gpu_local_rank5_numa1.png",
            title="GPU Local: Rank 5 (GPU on NUMA 1)",
        )

    def test_02_gpu_remote_binding(self):
        print("\n--- Testing: `bind: gpu-remote` (Multi-NUMA) ---")
        shape_yaml = "options:\n  bind: gpu-remote\n  bind_to: core\nresources:\n  - type: core\n    count: 2"

        self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=8,
            gpus_per_task=1,
            xml_file=TEST_GPU_XML_FILE,
            output_filename="09_gpu_remote_rank0.png",
            title="GPU Remote: Rank 0 (GPU on NUMA 0, CPU on NUMA 1)",
        )

    def test_03_gpu_remote_fails_on_single_numa(self):
        print("\n--- Testing: `bind: gpu-remote` (Single-NUMA, Expect Fail) ---")
        shape_yaml = "options:\n  bind: gpu-remote\nresources:\n  - type: core\n    count: 1"

        # This test is expected to raise an error, so no graphic will be generated.
        with self.assertRaisesRegex(RuntimeError, "Cannot find a remote NUMA node"):
            self.run_test_case(
                shape_yaml, local_rank=0, local_size=1, gpus_per_task=1, xml_file=TEST_XML_FILE
            )


if __name__ == "__main__":
    unittest.main()
