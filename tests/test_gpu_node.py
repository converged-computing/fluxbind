import os
import unittest

import yaml

from fluxbind.graph.graphic import TopologyVisualizer
from fluxbind.graph.shape import Shape, TopologyResult

here = os.path.dirname(os.path.abspath(__file__))

TEST_GPU_XML_FILE = os.path.join(here, "corona.xml")
GRAPHICS_OUTPUT_DIR = os.path.join(here, "img", "gpu")


@unittest.skipUnless(
    os.path.exists(TEST_GPU_XML_FILE), f"Skipping GPU tests, {TEST_GPU_XML_FILE} not found."
)
class TestGpuShapeAllocator(unittest.TestCase):
    """
    Tests for GPU-aware allocation, requiring a multi-NUMA topology (corona.xml).
    """

    @classmethod
    def setUpClass(cls):
        os.makedirs(GRAPHICS_OUTPUT_DIR, exist_ok=True)

    def run_test_case(
        self,
        shape_yaml_str: str,
        local_rank: int,
        local_size: int,
        title: str,
        output_filename: str,
        width=30,
        height=24,
        **kwargs,
    ) -> TopologyResult:
        """
        A helper function to run a single test case and generate a graphic.
        Defaults to a smaller image size, but can be overridden.
        """
        shape_allocator = Shape(yaml.safe_load(shape_yaml_str))
        result = shape_allocator.run(
            xml_file=TEST_GPU_XML_FILE, local_size=local_size, local_rank=local_rank, **kwargs
        )

        if output_filename and result.nodes:
            filepath = os.path.join(GRAPHICS_OUTPUT_DIR, output_filename)
            visualizer = TopologyVisualizer(
                result.topo,
                result.nodes,  # This single list contains all assigned resources
                affinity_target=result.topo.last_affinity_target,
            )
            visualizer.title = title
            # Pass the width and height to the draw method
            visualizer.draw(filepath, width=width, height=height)

        return result

    def test_01_gpu_local_binding(self):
        """
        Test: Assign 1 GPU per task and bind to 2 local cores.
        This test now generates a graphic for all 8 ranks.
        """
        print("\n--- Testing: `bind: gpu-local` (Multi-NUMA) ---")
        shape_yaml = """
options:
  bind: gpu-local
resources:
  - type: core
    count: 2
"""
        local_size = 8
        gpus_per_task = 1

        for i in range(local_size):
            title = f"GPU Local: Rank {i} of {local_size}"
            output_filename = f"01_gpu_local_rank{i}.png"
            print(f"  -> Testing and generating graphic for local_rank {i}...")

            result = self.run_test_case(
                shape_yaml,
                local_rank=i,
                local_size=local_size,
                gpus_per_task=gpus_per_task,
                title=title,
                output_filename=output_filename,
                width=36,
                height=16,
            )

            self.assertEqual(result.gpu_string, str(i))
            expected_package = 0 if i < 4 else 1

            self.assertTrue(result.nodes, f"Rank {i} was not assigned any CPU nodes.")
            # We only check the CPU nodes for package affinity
            cpu_nodes = [node for node in result.nodes if node[1].get("type") in ["Core", "PU"]]
            for gp, _ in cpu_nodes:
                package = result.topo.get_ancestor_of_type(gp, "Package")
                self.assertIsNotNone(
                    package, f"Could not find parent package for core on rank {i}."
                )
                self.assertEqual(
                    package[1].get("os_index"),
                    expected_package,
                    f"Rank {i} bound to wrong package.",
                )

    def test_02_gpu_remote_binding(self):
        """
        Test: Assign a GPU on one NUMA node, but bind to cores on the other.
        This test now generates a graphic for all 8 ranks.
        """
        print("\n--- Testing: `bind: gpu-remote` (Multi-NUMA) ---")
        shape_yaml = """
options:
  bind: gpu-remote
resources:
  - type: core
    count: 4
"""
        local_size = 8
        gpus_per_task = 1

        for i in range(local_size):
            title = f"GPU Remote: Rank {i} of {local_size}"
            output_filename = f"02_gpu_remote_rank{i}.png"
            print(f"  -> Testing and generating graphic for local_rank {i}...")

            result = self.run_test_case(
                shape_yaml,
                local_rank=i,
                local_size=local_size,
                gpus_per_task=gpus_per_task,
                title=title,
                output_filename=output_filename,
                width=36,
                height=16,
            )
            self.assertEqual(result.gpu_string, str(i))

            expected_package = 1 if i < 4 else 0

            cpu_nodes = [node for node in result.nodes if node[1].get("type") in ["Core", "PU"]]
            self.assertTrue(cpu_nodes, f"Rank {i} was not assigned any CPU nodes.")
            for gp, _ in cpu_nodes:
                package = result.topo.get_ancestor_of_type(gp, "Package")
                self.assertIsNotNone(
                    package, f"Could not find parent package for core on rank {i}."
                )
                self.assertEqual(
                    package[1].get("os_index"),
                    expected_package,
                    f"Rank {i} bound to wrong package.",
                )

    def test_03_contextual_affinity_to_gpu(self):
        """
        Test: Find a GPU, then find cores with affinity to *that specific* GPU.
        """
        print("\n--- Testing: Contextual Affinity to specific GPU ---")
        shape_yaml = """
resources:
  - type: numanode
    count: 1
    with:
      - type: gpu
        count: 1
        with:
          - type: core
            count: 2
            affinity:
              type: gpu
"""
        result = self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=1,
            title="Contextual Affinity",
            output_filename="03_contextual_affinity.png",
            width=36,
            height=16,
        )
        self.assertIsNotNone(result.topo.last_affinity_target, "Affinity target was not set")
        target_pkg = result.topo.get_ancestor_of_type(
            result.topo.last_affinity_target[0], "Package"
        )
        self.assertIsNotNone(target_pkg, "Could not find parent package for affinity target.")

        cpu_nodes = [node for node in result.nodes if node[1].get("type") in ["Core", "PU"]]
        for gp, _ in cpu_nodes:
            package = result.topo.get_ancestor_of_type(gp, "Package")
            self.assertIsNotNone(package, "Could not find parent package for allocated core.")
            self.assertEqual(package[1].get("os_index"), target_pkg[1].get("os_index"))

    def test_04_multi_resource_gpu_and_nic(self):
        """
        Test: Find a NUMA node that contains BOTH a GPU and a NIC.
        """
        print("\n--- Testing: Find NUMA node with GPU and NIC ---")
        shape_yaml = """
resources:
  - type: numanode
    count: 1
    with:
      - type: nic
        count: 1
      - type: gpu
        count: 1
"""
        result = self.run_test_case(
            shape_yaml,
            local_rank=0,
            local_size=1,
            title="Find NUMA with GPU+NIC",
            output_filename="04_gpu_and_nic.png",
            width=36,
            height=16,
        )
        # IMPORTANT: we have 24 actual result, and 1 GPU and 1 NIC here.
        self.assertEqual(len(result.nodes), 26)
        for gp, _ in result.nodes:
            package = result.topo.get_ancestor_of_type(gp, "Package")
            self.assertIsNotNone(package, "Could not find parent package for allocated core.")
            self.assertEqual(package[1].get("os_index"), 1)

    def test_05_multi_gpu_task(self):
        """
        Test: Assign 2 GPUs per task and bind locally.
        """
        print("\n--- Testing: Multi-GPU Task (2 GPUs per rank) ---")
        shape_yaml = """
options:
  bind: gpu-local
resources:
  - type: core
    count: 4
"""
        local_size = 4
        gpus_per_task = 2

        for i in range(local_size):
            title = f"Multi-GPU: Rank {i} of {local_size}"
            output_filename = f"05_multi_gpu_rank{i}.png"
            print(f"  -> Testing and generating graphic for local_rank {i}...")

            result = self.run_test_case(
                shape_yaml,
                local_rank=i,
                local_size=local_size,
                gpus_per_task=gpus_per_task,
                title=title,
                output_filename=output_filename,
                width=36,
                height=16,
            )

            gpu_start = i * gpus_per_task
            gpu_end = gpu_start + gpus_per_task
            expected_gpus = ",".join(map(str, range(gpu_start, gpu_end)))
            self.assertEqual(result.gpu_string, expected_gpus)

            expected_package = 0 if i < 2 else 1

            cpu_nodes = [node for node in result.nodes if node[1].get("type") in ["Core", "PU"]]
            self.assertTrue(cpu_nodes, f"Rank {i} was not assigned any CPU nodes.")
            for gp, _ in cpu_nodes:
                package = result.topo.get_ancestor_of_type(gp, "Package")
                self.assertIsNotNone(
                    package, f"Could not find parent package for core on rank {i}."
                )
                self.assertEqual(
                    package[1].get("os_index"),
                    expected_package,
                    f"Rank {i} bound to wrong package.",
                )


if __name__ == "__main__":
    unittest.main()
