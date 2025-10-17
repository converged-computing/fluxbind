import subprocess
import sys

import fluxbind.shape.commands as commands
import fluxbind.shape.gpu as gpus
import fluxbind.utils as utils


class Shape:
    """
    Parses a YAML shape file and determines the hwloc binding for a given process.

    We get the binding string based on a process's rank and node context.
    """

    def __init__(self, filepath, machine="machine:0"):
        """
        Loads and parses the YAML shape file upon instantiation.
        """
        self.machine = machine
        self.machine_cpuset = commands.hwloc_calc.get_cpuset(self.machine)
        self.data = self.load_file(filepath) or []
        # This discovers and cache hardware properties on init
        # The expectation is that this is running from the node (task)
        self.num_cores = commands.hwloc_calc.count("core", within=self.machine)
        self.num_pus = commands.hwloc_calc.count("pu", within=self.machine)
        self.numa_node_cpusets = commands.hwloc_calc.list_cpusets("numa", within=self.machine)
        self.pus_per_core = self.num_pus // self.num_cores if self.num_cores > 0 else 0
        # For GPU topology, we care about NUMA nodes.
        self.gpus_by_numa = self.discover_gpus()

    def discover_gpus(self):
        """
        Discovers available GPU PCI bus IDs.
        """
        all_pci_ids = []

        # Try for nvidia and then rocm
        for command in [commands.nvidia_smi.get_pci_bus_ids, commands.rocm_smi.get_pci_bus_ids]:
            try:
                # This is pci addresses ACROSS numa nodes
                all_pci_ids = command()
            except Exception:
                pass

        gpus_by_numa = {}

        # For each GPU, find out which NUMA node it belongs to
        for pci_id in all_pci_ids:
            # Ask hwloc for the cpuset where the pci lives.
            # I'm not sure if this will work for nvidia if doens't show in lstopo
            gpu_cpuset = commands.hwloc_calc.get_cpuset(f"pci={pci_id}")
            found_numa = False
            for i, numa_cpuset in enumerate(self.numa_node_cpusets):
                # Check if the GPU's cpuset is a subset of this NUMA node's cpuset
                intersection = commands.hwloc_calc.get_cpuset(f"'{gpu_cpuset}' x '{numa_cpuset}'")

                if intersection == gpu_cpuset:
                    if i not in gpus_by_numa:
                        gpus_by_numa[i] = []
                    gpus_by_numa[i].append(pci_id)
                    found_numa = True
                    break

            # Raise an error - I want to know about this case.
            if not found_numa:
                raise ValueError(f"Warning: Could not determine NUMA locality for GPU {pci_id}")

        # Make an ordered set just for easy list access
        self.ordered_gpus = []
        for numa_idx in sorted(gpus_by_numa.keys()):
            for pci_id in gpus_by_numa[numa_idx]:
                self.ordered_gpus.append({"pci_id": pci_id, "numa_index": numa_idx})

        return gpus_by_numa

    def load_file(self, filepath=None):
        """
        Loads and parses the YAML shape file.
        """
        if filepath is not None:
            return utils.read_yaml(filepath)

    @staticmethod
    def parse_range(range_str: str) -> set:
        """
        Parse a string like '0-7,12,15' into a set of integers.
        """
        indices = set()
        for part in str(range_str).split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
        return indices

    @staticmethod
    def evaluate_formula(formula_template: str, local_rank: int) -> int:
        """
        Evaluate a shell arithmetic formula by substituting $local_rank.

        This assumes running on the rank where the binding is asked for.
        """
        formula = str(formula_template).replace("$local_rank", str(local_rank))
        command = f'echo "{formula}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())

    def find_matching_rule(self, rank: int, node_id: int) -> dict:
        """
        Finds the first rule in the shape data that matches the given rank or node_id.
        """
        rules = self.data if isinstance(self.data, list) else []

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            matches = False
            if "ranks" in rule and rank in self.parse_range(rule["ranks"]):
                matches = True
            if "nodes" in rule and node_id in self.parse_range(rule["nodes"]):
                matches = True
            if matches:
                return rule

        # Last resort - look for default
        if isinstance(self.data, dict) and "default" in self.data:
            return self.data["default"]
        for item in rules:
            if isinstance(item, dict) and "default" in item:
                return item["default"]
        return None

    def get_gpu_local_binding(self, rule: dict, local_rank: int, gpus_per_task: int) -> str:
        """
        Calculate a 'gpu-local' binding using the topology-aware ordered GPU list.
        """
        assignment = gpus.GPUAssignment.for_rank(local_rank, gpus_per_task, self.ordered_gpus)

        # The CPU domain is the union of NUMA nodes for the assigned GPUs.
        domain_locations = [f"numa:{i}" for i in assignment.numa_indices]
        domain = " ".join(domain_locations)
        cpu_binding_string = self.get_binding_in_gpu_domain(rule, local_rank, gpus_per_task, domain)
        return f"{cpu_binding_string};{assignment.cuda_devices}"

    def get_gpu_remote_binding(self, rule: dict, local_rank: int, gpus_per_task: int) -> str:
        """
        Calculates a 'gpu-remote' binding using the topology-aware ordered GPU list.
        """
        if len(self.numa_node_cpusets) < 2:
            raise RuntimeError("'bind: gpu-remote' is invalid on a single-NUMA system.")
        assignment = gpus.GPUAssignment.for_rank(local_rank, gpus_per_task, self.ordered_gpus)

        # Find all remote NUMA domains relative to the set of local domains.
        all_numa_indices = set(range(len(self.numa_node_cpusets)))
        remote_numa_indices = sorted(list(all_numa_indices - assignment.numa_indices))

        if not remote_numa_indices:
            raise RuntimeError(
                f"Cannot find a remote NUMA node for rank {local_rank}; its GPUs span all NUMA domains."
            )

        offset = rule.get("offset", 0)
        if offset >= len(remote_numa_indices):
            raise ValueError(f"Offset {offset} is out of range for remote NUMA domains.")

        target_remote_numa_idx = remote_numa_indices[offset]
        domain = f"numa:{target_remote_numa_idx}"

        cpu_binding_string = self.get_binding_in_gpu_domain(rule, local_rank, gpus_per_task, domain)
        return f"{cpu_binding_string};{assignment.cuda_devices}"

    def get_binding_in_gpu_domain(
        self, rule: dict, local_rank: int, gpus_per_task: int, domain: str
    ):
        """
        A dedicated binding engine for GPU jobs. It applies user preferences within a calculated domain
        (e.g., "numa:0" or "numa:0 numa:1").
        """
        hwloc_type = rule.get("type")
        if not hwloc_type:
            raise ValueError("Rule with GPU binding must have a 'type'.")

        if hwloc_type in ["numa", "package", "machine"]:
            # If a broad type is requested, the binding is the domain itself.
            return domain

        elif hwloc_type in ["core", "pu", "l2cache", "l3cache"]:

            # Get the number of objects to select, defaulting to 1.
            count = rule.get("count", 1)

            all_indices_in_domain = commands.hwloc_calc.get_object_in_set(
                domain, hwloc_type, "all"
            ).split(",")
            if not all_indices_in_domain or not all_indices_in_domain[0]:
                raise RuntimeError(f"No objects of type '{hwloc_type}' found in domain '{domain}'.")

            if "prefer" in rule:
                if count > 1:
                    raise ValueError("'prefer' and 'count > 1' cannot be used together.")
                try:
                    requested_index = str(int(rule["prefer"]))
                    if requested_index in all_indices_in_domain:
                        return f"{hwloc_type}:{requested_index}"
                    else:
                        print(
                            f"Warning: Preferred index '{requested_index}' not available in domain '{domain}'. Falling back.",
                            file=sys.stderr,
                        )
                except (ValueError, TypeError):
                    raise ValueError(
                        f"The 'prefer' key must be a simple integer, but got: {rule['prefer']}"
                    )

            # Default assignment: Calculate the slice of objects for this rank.
            # We need to know this rank's turn on the current domain.
            num_domains = len(domain.split())
            rank_turn_in_domain = local_rank // num_domains

            start_index = rank_turn_in_domain * count
            end_index = start_index + count

            if end_index > len(all_indices_in_domain):
                raise ValueError(
                    f"Not enough '{hwloc_type}' objects in domain '{domain}' to satisfy request "
                    f"for {count} objects for rank {local_rank} (needs up to index {end_index-1}, "
                    f"only {len(all_indices_in_domain)} available)."
                )

            # Get the slice of object indices.
            target_indices_slice = all_indices_in_domain[start_index:end_index]

            # Construct a space-separated list of location objects.
            # e.g., "core:0 core:1 core:2 core:3 core:4 core:5"
            binding_locations = [f"{hwloc_type}:{i}" for i in target_indices_slice]
            return " ".join(binding_locations)
        else:
            raise ValueError(f"Unsupported type '{hwloc_type}' for GPU binding.")

    def get_binding_for_rank(self, rank, node_id, local_rank, gpus_per_task=None) -> str:
        """
        The main method to get the final hwloc binding string for a process.

        Args:
            rank: The global rank of the process.
            node_id: The logical ID of the node in the allocation.
            local_rank: The rank of the process on the local node.

        Returns:
            The hwloc location string (e.g., "core:5") or a keyword "UNBOUND".
        """
        rule = self.find_matching_rule(rank, node_id)
        if rule is None:
            raise ValueError(
                f"No matching rule or default found for rank {rank} on node {node_id}."
            )

        hwloc_type = rule.get("type")
        if hwloc_type is None:
            raise ValueError(f"Matching rule has no 'type' defined: {rule}")

        # Are we doing something with GPU?
        if rule.get("bind") == "gpu-local":
            return self.get_gpu_local_binding(rule, local_rank, gpus_per_task)
        if rule.get("bind") == "gpu-remote":
            return self.get_gpu_remote_binding(rule, local_rank, gpus_per_task)

        cpu_binding_string = self.get_cpu_binding(hwloc_type, rule, local_rank)

        # Return the CPU binding and the "NONE" sentinel for the device.
        return f"{cpu_binding_string};NONE"

    def get_cpu_binding(self, hwloc_type, rule, local_rank):
        """
        Get CPU binding for a rank.
        """
        if hwloc_type.lower() == "unbound":
            return "UNBOUND"

        # Simple pattern names!
        if "pattern" in rule:
            pattern = rule.get("pattern", "packed").lower()
            reverse = rule.get("reverse", False)

            if pattern == "packed":
                # packed we need to know the total number of the target object type.
                total_objects = commands.hwloc_calc.count(hwloc_type, within=self.machine)
                target_index = local_rank
                if reverse:
                    target_index = total_objects - 1 - local_rank
                return f"{hwloc_type}:{target_index}"

            elif pattern == "interleave":
                # I think interleave could work well for SMT-aware binding of PUs.
                if hwloc_type != "pu":
                    raise ValueError(
                        "The 'interleave' pattern requires 'type: pu' for SMT-aware binding."
                    )

                # These values were discovered and cached in __init__
                core_index = local_rank % self.num_cores
                pu_on_core_index = local_rank // self.num_cores
                if reverse:
                    core_index = self.num_cores - 1 - core_index

                # Return the hierarchical binding string
                return f"core:{core_index}.pu:{pu_on_core_index}"

            else:
                raise ValueError(f"Unknown pattern '{pattern}'. Use 'packed' or 'interleave'.")

        # More complex, custom user formula
        elif "formula" in rule:
            formula_template = rule.get("formula")
            if formula_template is None:
                raise ValueError(f"Matching rule has no 'formula' defined: {rule}")

            index = self.evaluate_formula(formula_template, local_rank)
            if index is None:
                raise ValueError("Formula evaluation failed.")

            return f"{hwloc_type}:{index}"

        else:
            raise ValueError(f"Rule must contain either a 'pattern' or a 'formula': {rule}")
