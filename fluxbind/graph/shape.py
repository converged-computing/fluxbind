import logging
import sys
from dataclasses import dataclass

import fluxbind.shape.commands as commands
import fluxbind.utils as utils
from fluxbind.graph.graph import HwlocTopology
from fluxbind.graph.graphic import TopologyVisualizer
from fluxbind.shape.gpu import GPUAssignment

log = logging.getLogger(__name__)


@dataclass
class TopologyResult:
    """
    A simple dataclass to hold the results of an allocation run.
    """

    topo: HwlocTopology = None
    nodes: list = None
    mask: str = None
    gpu_string: str = "NONE"


class Shape:
    """
    Finds a hardware binding for a single task by interpreting a shapefile
    according to a hierarchy of rules. This class implements the "pool division"
    model, where a total set of resources on a node is divided among the
    local tasks.
    """

    valid_bind_modes = ["core", "pu", "process", "none", "gpu-local", "gpu-remote"]
    bind_default = "core"

    def __init__(self, jobspec, debug=False):
        """
        Initializes the Shape object, parsing the jobspec and any binding options.
        """
        self._setup_logging(debug)
        self.load(jobspec)
        self.hwloc_calc = commands.HwlocCalcCommand()
        self.set_bind_preference()

    def load(self, jobspec):
        """
        Load the jobspec, or if already loaded, just set.
        """
        if isinstance(jobspec, dict):
            self.jobspec = jobspec
        else:
            self.jobspec = utils.read_yaml(jobspec)

    def set_bind_preference(self):
        """
        Get the binding preference.
        """
        options = self.jobspec.get("options", {})
        bind_mode = options.get("bind")
        if bind_mode:
            bind_mode = bind_mode.lower()
            if bind_mode not in self.valid_bind_modes:
                raise ValueError(
                    f"Invalid 'bind' option: {bind_mode}. Must be one of {self.valid_bind_modes}."
                )
            if bind_mode == "process":
                bind_mode = "pu"
        self.bind_mode = bind_mode

    def _setup_logging(self, debug=False):
        """
        Setup logging, honoring debug if user provides from client.
        """
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="[%(levelname)s] %(message)s",
            force=True,
        )

    def get_binding_for_rank(
        self,
        rank,
        local_rank,
        local_size,
        gpus_per_task=None,
        xml_file=None,
        graphic=None,
        **kwargs,
    ):
        """
        Main entrypoint. Calculates a binding for a rank given the job geometry.
        Unused arguments like 'node_id' are captured by kwargs and discarded.
        TODO: we should make everything lower() from the getgo.
        """
        log.info(
            f"Processing request for global_rank={rank}, with local_rank={local_rank} of {local_size}"
        )
        if local_rank >= local_size:
            raise ValueError(f"local-rank ({local_rank}) must be < local_size ({local_size}).")

        mapping = self.run(xml_file, local_size, local_rank, gpus_per_task)

        if graphic and mapping.nodes:
            visualizer = TopologyVisualizer(
                mapping.topo, mapping.nodes, affinity_target=mapping.topo.last_affinity_target
            )
            visualizer.draw(graphic)

        final_output = f"{mapping.mask};{mapping.gpu_string}"
        log.info(
            f"\nSUCCESS! The final cpuset binding mask for rank {local_rank} is: {final_output}"
        )

        if mapping.mask:
            print(final_output)
        sys.exit(0 if mapping.mask else 1)

    def get_bind_type(self, total_allocation):
        """
        Finds a binding by interpreting a shapefile according to a hierarchy of rules:
          1. Explicit options.bind in the shapefile.
          2. Implicit intent from the most granular resource type requested.
          3. A default of core for HPC.
        """
        # See https://gist.github.com/vsoch/be2d1ec712e33ec157bab2dc9a36b10a
        # User explicit preference takes priority. We check here because the gpu-* types
        # cannot trigger here.
        if self.bind_mode in ["core", "pu", "none"]:
            log.info(
                f"Using explicit binding preference from shapefile options: '{self.bind_mode}'."
            )
            return self.bind_mode

        # Try to infer implicit intent
        if total_allocation:
            most_granular_type = (
                max(total_allocation, key=lambda item: item[1].get("depth", -1))[1]
                .get("type")
                .lower()
            )
            if most_granular_type in ["core", "pu"]:
                log.info(
                    f"Using implicit binding preference from resource request: '{most_granular_type}'."
                )
                return most_granular_type

        # Fall back to a safe default for HPC.
        log.info("No explicit or implicit preference. Using safe HPC default: 'core'.")
        return self.bind_default

    def get_gpu_binding(self, topology, local_rank, gpus_per_task):
        """
        Handles GPU-specific binding logic.

        This method determines which GPUs are assigned to the current rank and then
        calculates the appropriate CPU search domain (a set of parent Packages)
        based on the GPU's NUMA locality.

        Returns:
            A tuple containing the GPUAssignment object and a set of graph pointers
            to the Package(s) that should be used for the CPU search.
        """
        gpus_per_task = gpus_per_task or 0
        if gpus_per_task <= 0:
            raise ValueError(f"'bind: {self.bind_mode}' requires --gpus-per-task to be > 0.")

        # This uses the dataclass you provided to get our rank's slice of GPUs.
        gpu_assignment = GPUAssignment.for_rank(local_rank, gpus_per_task, topology.ordered_gpus)

        # Determine the target NUMA domains based on the bind mode.
        if self.bind_mode == "gpu-local":
            log.info(
                f"Binding to CPUs local to assigned GPUs (NUMA domains: {list(gpu_assignment.numa_indices)})."
            )
            target_numa_indices = gpu_assignment.numa_indices

        else:  # gpu-remote
            all_numa_indices = {
                data.get("os_index") for _, data in topology.find_objects(type="NUMANode")
            }
            remote_numa_indices = all_numa_indices - gpu_assignment.numa_indices

            if not remote_numa_indices:
                raise RuntimeError(
                    f"Cannot find a remote NUMA node for rank {local_rank}; assigned GPUs span all NUMA domains."
                )

            # For simplicity, we target the first available remote NUMA domain.
            target_numa_indices = {sorted(list(remote_numa_indices))[0]}
            log.info(
                f"Binding to CPUs remote to assigned GPUs (target NUMA domains: {list(target_numa_indices)})."
            )

        # Find the graph pointers for the NUMA objects corresponding to our target indices.
        domain_numa_gps = {
            gp
            for gp, data in topology.find_objects(type="NUMANode")
            if data.get("os_index") in target_numa_indices
        }

        # Now, find the parent Packages of these NUMA domains to define the final CPU search space.
        cpu_binding_domain_gps = set()
        for numa_gp in domain_numa_gps:
            package = topology.get_ancestor_of_type(numa_gp, "Package")
            if package:
                cpu_binding_domain_gps.add(package[0])

        if not cpu_binding_domain_gps:
            raise RuntimeError(f"Could not find a parent Package for the target NUMA domains.")

        return gpu_assignment, cpu_binding_domain_gps

    def run(self, xml_file, local_size, local_rank, gpus_per_task=None):
        """
        Finds a binding by applying the hierarchy of rules to the shapefile and topology.
        This is the definitive version, returning a single, complete list of assigned nodes.
        """
        topology = HwlocTopology(xml_file)
        gpu_assignment = None
        total_allocation = None

        if self.bind_mode in ["gpu-local", "gpu-remote"]:
            gpu_assignment, cpu_domain_gps = self.get_gpu_binding(
                topology, local_rank, gpus_per_task
            )
            total_allocation = [(gp, topology.graph.nodes[gp]) for gp in cpu_domain_gps]
        else:
            log.info(f"Finding the total resource pool for all {local_size} ranks on this node...")
            total_allocation = topology.match_resources(self.jobspec)

        if total_allocation is None:
            raise RuntimeError(
                "Failed to find any resources on the node that match the requested shape."
            )

        # Determine the target binding level. No binding? Then we probably just wanted GPUs.
        bind_level = self.get_bind_type(total_allocation)
        if bind_level == "none":
            return TopologyResult(
                nodes=total_allocation,
                mask="UNBOUND",
                gpu_string=gpu_assignment.cuda_devices if gpu_assignment else "NONE",
            )

        # Change the allocation into a list of bindable nodes.
        log.info(f"Deriving bindable resources with final preference: '{bind_level}'.")
        leaf_nodes = topology.find_bindable_leaves(total_allocation, bind_level)
        if not leaf_nodes:
            raise RuntimeError(
                f"Could not find any bindable resources of type '{bind_level}' for the allocation."
            )

        # Apply a pattern of distribution (e.g., packed/scatter).
        final_nodes = self.apply_binding_pattern(leaf_nodes, local_size, local_rank)
        log.info(
            f"\nAssigning {len(final_nodes)} '{bind_level}' resources for local rank {local_rank}:"
        )
        if not final_nodes:
            return TopologyResult(nodes=[], mask="0x0", topo=topology)

        topology.summarize(final_nodes)

        # Now we need the actual cpusets which is what does the binding.
        cpusets = []
        if bind_level == "pu":
            cpusets = [d["cpuset"] for _, d in final_nodes if "cpuset" in d]
        elif bind_level == "core":
            for core_gp, _ in final_nodes:
                pus = sorted(
                    topology.get_descendants(core_gp, type=topology.translate_type("pu")),
                    key=lambda x: x[1].get("os_index", 0),
                )
                if pus:
                    first_pu_cpuset = pus[0][1].get("cpuset")
                    if first_pu_cpuset:
                        cpusets.append(first_pu_cpuset)
        raw_mask = self.hwloc_calc.get_cpuset(" ".join(cpusets)) if cpusets else "0x0"
        mask = raw_mask.replace(",,", ",")

        # We need to add devices (GPU,NIC) to the final nodes.
        # This is mostly for the graphic visualization
        assigned_devices = []
        if self.bind_mode in ["gpu-local", "gpu-remote"]:

            # For GPU modes, the assigned GPUs are in the `gpu_assignment` object.
            if gpu_assignment and gpu_assignment.pci_ids:
                for pci_id in gpu_assignment.pci_ids:
                    matches = topology.find_objects(type="PCIDev", pci_busid=pci_id.lower())
                    if matches:
                        assigned_devices.extend(matches)
        else:
            # For CPU-driven jobs, the assigned devices are any PCIDevs part of the allocation.
            assigned_devices = [
                node for node in total_allocation if node[1].get("type") == "PCIDev"
            ]

        # THE FINAL LIST dun dun dun
        all_assigned_nodes = final_nodes + assigned_devices

        return TopologyResult(
            # This contains both CPUs and Devices
            nodes=all_assigned_nodes,
            mask=mask,
            topo=topology,
            gpu_string=gpu_assignment.cuda_devices if gpu_assignment else "NONE",
        )

    def apply_binding_pattern(self, leaf_nodes, local_size, local_rank):
        """
        Given a set of chosen leaf nodes (typicall Core/PU) apply a binding pattern.

        The binding pattern is an option in the jobspec.
        """
        main_request = self.jobspec.get("resources", [{}])[0]
        pattern = main_request.get("pattern", "packed").lower()
        reverse = main_request.get("reverse", False)
        log.info(f"Applying distribution pattern: '{pattern}' (reverse={reverse}).")
        if reverse:
            leaf_nodes.reverse()

        # This block is correct. It divides the pool and finds the CPU slice for this rank.
        items_per_rank = len(leaf_nodes) // local_size

        # This will hold the assigned CPUs.
        final_nodes = []
        if items_per_rank == 0 and local_size > 0:
            if local_rank < len(leaf_nodes):
                final_nodes = [leaf_nodes[local_rank]]

        # Pack em up! Like little weiner hotdogs in plastic!
        elif pattern == "packed":
            start_index = local_rank * items_per_rank
            end_index = start_index + items_per_rank
            final_nodes = leaf_nodes[start_index:end_index]

        # I think interleaved is a little different, but I feel lazy right
        # now and don't want to think about it.
        elif pattern in ["scatter", "spread", "interleaved"]:
            strided_slice = leaf_nodes[local_rank::local_size]
            final_nodes = strided_slice[:items_per_rank]
        else:
            raise ValueError(f"Unknown pattern '{pattern}'.")
        return final_nodes
