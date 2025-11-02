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

    def __init__(self, jobspec, debug=False):
        """
        Initializes the Shape object, parsing the jobspec and any binding options.
        """
        self._setup_logging(debug)
        self.load(jobspec)
        self.hwloc_calc = commands.HwlocCalcCommand()

    def load(self, jobspec):
        """
        Load the jobspec, or if already loaded, just set.
        """
        if isinstance(jobspec, dict):
            self.jobspec = jobspec
        else:
            self.jobspec = utils.read_yaml(jobspec)

    @property
    def bind_preference(self):
        """
        Get the binding preference.
        """
        # Rule 1 (pre-check): Parse the explicit user override for binding.
        options = self.jobspec.get("options", {})
        bind_preference = options.get("bind", None)
        if not bind_preference:
            return

        # Map the preference to pu if they provided process
        bind_preference = bind_preference.lower()
        if bind_preference == "process":
            bind_preference = "pu"

        valid_options = ["core", "pu", "none"]
        if bind_preference not in valid_options:
            raise ValueError(
                f"Invalid 'bind' option in shapefile: '{self.bind_preference}'. Must be one of {valid_options}."
            )
        return bind_preference

    def _setup_logging(self, debug=False):
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
        bind_level = self.bind_preference

        # No explicit preference (first priority)
        if bind_level is None:

            # If the most granular type is PU or CPU, assume that is the preference
            most_granular_type = (
                max(total_allocation, key=lambda item: item[1].get("depth", -1))[1]
                .get("type")
                .lower()
            )
            if most_granular_type in ["core", "pu"]:
                bind_level = most_granular_type
                log.info(f"Using implicit binding preference from resource request: '{bind_level}.")

        # Otherwise, fall back to what we expect HPC to want, Core.
        if bind_level is None:
            bind_level = "core"
            log.info(f"No explicit or implicit preference. Using safe HPC default: '{bind_level}'.")

        # Note that bind_level can also be none (unbound)
        return bind_level

    def get_gpu_binding(self, topology, local_rank, gpus_per_task, bind_mode):
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
            raise ValueError(f"'bind: {bind_mode}' requires --gpus-per-task to be > 0.")

        # This uses the dataclass you provided to get our rank's slice of GPUs.
        gpu_assignment = GPUAssignment.for_rank(local_rank, gpus_per_task, topology.ordered_gpus)

        # Determine the target NUMA domains based on the bind mode.
        if bind_mode == "gpu-local":
            log.info(
                f"Binding to CPUs local to assigned GPUs (NUMA domains: {list(gpu_assignment.numa_indices)})."
            )
            target_numa_indices = gpu_assignment.numa_indices

        else:  # gpu-remote
            all_numa_indices = {
                data.get("os_index") for _, data in topology.find_objects(type="numanode")
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
            for gp, data in topology.find_objects(type="numanode")
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
        """
        topology = HwlocTopology(xml_file)
        options = self.jobspec.get("options", {})
        bind_mode = options.get("bind", None)

        gpu_assignment = None
        total_allocation = None

        # Determine the set of resources we need to find CPUs in.
        if bind_mode in ["gpu-local", "gpu-remote"]:
            # If it's a GPU-driven binding, call the helper to get the GPU assignment
            # and the correct CPU search domain.
            gpu_assignment, cpu_domain_gps = self.get_gpu_binding(
                topology, local_rank, gpus_per_task, bind_mode
            )
            # The allocation for the CPU search is now the set of packages determined by the helper.
            total_allocation = [(gp, topology.graph.nodes[gp]) for gp in cpu_domain_gps]
        else:
            # For all other cases, perform the standard resource search based on the shapefile.
            log.info(f"Finding the total resource pool for all {local_size} ranks on this node...")
            total_allocation = topology.match_resources(self.jobspec)

        # Honor a user request for binding level or use a default
        bind_level = self.get_bind_type(total_allocation)

        # Special case of no binding, but we likely want affinity to devices, etc.
        if bind_level == "none":
            log.info("bind: none selected. Skipping cpuset generation.")
            summary_nodes = [
                node
                for node in total_allocation
                if node[1].get("type") not in ["Machine", "Package", "NUMANode"]
            ]
            log.info(
                f"\nFound {len(summary_nodes)} un-bound resources for local rank {local_rank}:"
            )
            topology.summarize(summary_nodes)
            return TopologyResult(
                nodes=total_allocation,
                mask="UNBOUND",
                gpu_string=gpu_assignment.cuda_devices if gpu_assignment else "NONE",
            )

        log.info(f"Deriving bindable resources with final preference: '{bind_level}'.")
        leaf_nodes = []
        explicit_nodes = [
            node for node in total_allocation if node[1].get("type").lower() == bind_level
        ]
        if len(explicit_nodes) == len(total_allocation):
            leaf_nodes = total_allocation
        else:
            for container_gp, _ in total_allocation:
                descendants = topology.get_descendants(
                    container_gp, type=topology.translate_type(bind_level)
                )
                leaf_nodes.extend(descendants)

        leaf_nodes = list({gp: (gp, data) for gp, data in leaf_nodes}.values())
        leaf_nodes.sort(key=topology.get_sort_key_for_node)

        if not leaf_nodes:
            raise RuntimeError(
                f"Could not find any bindable resources of type '{bind_level}' for the allocation."
            )

        # Get pattern and reverse options from the first resource request.
        # This simplification assumes one pattern applies to the whole allocation.
        main_request = self.jobspec.get("resources", [{}])[0]
        pattern = main_request.get("pattern", "packed").lower()
        reverse = main_request.get("reverse", False)

        log.info(f"Applying distribution pattern: '{pattern}' (reverse={reverse}).")

        # Apply the 'reverse' modifier to the canonical list of resources.
        if reverse:
            leaf_nodes.reverse()

        # Calculate the shape ACROSS the node, and then break apart by rank.
        # Remember that:
        #  1. The local size is the number of tasks on a node.
        #  2. The shape is describing the slot we want on a node, and we want some number
        # So we are calculating the total number of resources (len(leaf_nodes)) based on the shape
        # and dividing that number evenly among all tasks on the node. Integer division discards remainders.
        # This makes the assumption we are doing uniform, contiguous chunking.
        items_per_rank = len(leaf_nodes) // local_size
        final_nodes = []

        if items_per_rank == 0 and local_size > 0:
            log.warning(f"Oversubscription detected. Distributing available resources one by one.")
            if local_rank < len(leaf_nodes):
                final_nodes = [leaf_nodes[local_rank]]
        elif pattern == "packed":
            log.debug("Using 'packed' pattern: assigning a contiguous block of resources.")
            start_index = local_rank * items_per_rank
            end_index = start_index + items_per_rank
            final_nodes = leaf_nodes[start_index:end_index]
        elif pattern in ["scatter", "spread", "interleaved"]:
            log.debug("Using 'scatter' pattern: dealing resources like cards.")

            # This slice syntax means "start at local_rank and take every local_size-th item"
            strided_slice = leaf_nodes[local_rank::local_size]

            # Ensure we don't take more than our fair share if division is not perfect.
            final_nodes = strided_slice[:items_per_rank]
        else:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Must be 'packed' or 'scatter'/'spread'/'interleaved'."
            )

        log.info(
            f"\nAssigning {len(final_nodes)} '{bind_level}' resources for local rank {local_rank}:"
        )
        if not final_nodes:
            return TopologyResult(nodes=[], mask="0x0", topo=topology)

        topology.summarize(final_nodes)

        cpusets = []
        if bind_level == "pu":
            cpusets = [d["cpuset"] for _, d in final_nodes if "cpuset" in d]
        elif bind_level == "core":
            log.debug("bind='core' selected. Generating cpuset from the first PU of each Core.")
            for core_gp, _ in final_nodes:
                # Use topology's translate_type for consistency
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
        return TopologyResult(
            nodes=final_nodes,
            mask=mask,
            topo=topology,
            gpu_string=gpu_assignment.cuda_devices if gpu_assignment else "NONE",
        )
