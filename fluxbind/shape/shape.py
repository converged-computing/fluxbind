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
        self.data = self.load_file(filepath)
        # This discovers and cache hardware properties on init
        # The expectation is that this is running from the node (task)
        self.num_cores = commands.hwloc_calc.count("core", within=self.machine)
        self.num_pus = commands.hwloc_calc.count("pu", within=self.machine)
        self.numa_node_cpusets = commands.hwloc_calc.list_cpusets("numa", within=self.machine)
        self.pus_per_core = self.num_pus // self.num_cores if self.num_cores > 0 else 0
        self.gpu_pci_ids = self.discover_gpus()

    def discover_gpus(self) -> list:
        """
        Discovers available GPU PCI bus IDs.
        """
        return commands.nvidia_smi.get_pci_bus_ids()

    def load_file(self, filepath):
        """
        Loads and parses the YAML shape file.
        """
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
        assignment = gpus.GPUAssignment.for_rank(local_rank, gpus_per_task, self.gpu_pci_ids)
        pci_locations = [f"pci={pci_id}" for pci_id in assignment.pci_ids]
        domain_cpuset = commands.hwloc_calc.union_of_locations(pci_locations)
        cpu_binding_string = self.get_gpu_cpu_binding(
            rule, local_rank, gpus_per_task, domain_cpuset
        )
        return f"{cpu_binding_string};{assignment.cuda_devices}"

    def get_gpu_remote_binding(self, rule: dict, local_rank: int, gpus_per_task: int) -> str:
        if len(self.numa_node_cpusets) < 2:
            raise RuntimeError("'bind: gpu-remote' is invalid on a single-NUMA system.")
        assignment = gpus.GPUAssignment.for_rank(local_rank, gpus_per_task, self.gpu_pci_ids)
        primary_gpu_pci_id = assignment.pci_ids[0]
        primary_gpu_numa_cpuset = commands.hwloc_calc.get_cpuset(f"numaof:pci={primary_gpu_pci_id}")
        remote_numa_cpusets = [cs for cs in self.numa_node_cpusets if cs != primary_gpu_numa_cpuset]
        if not remote_numa_cpusets:
            raise RuntimeError(
                f"Could not find a NUMA node remote from GPU {assignment.indices[0]}."
            )
        offset = rule.get("offset", 0)
        domain = remote_numa_cpusets[offset]
        cpu_binding_string = self.get_gpu_cpu_binding(rule, local_rank, gpus_per_task, domain)
        return f"{cpu_binding_string};{assignment.cuda_devices}"

    def get_gpu_cpu_binding(
        self, rule: dict, local_rank: int, gpus_per_task: int, domain: str
    ) -> str:
        hwloc_type = rule.get("type")
        if not hwloc_type:
            raise ValueError("Rule with GPU binding must have a 'type'.")

        if hwloc_type in ["numa", "package", "l3cache", "machine"]:
            # The user wants to bind to the entire domain, try to get location for it
            return commands.hwloc_calc.get_cpuset(f"'{domain}'")

        elif hwloc_type in ["core", "pu", "l2cache"]:
            # This logic returns a simple name like "core:5", which is correct.
            if "prefer" in rule:
                try:
                    requested_index = int(rule["prefer"])
                    return commands.hwloc_calc.get_object_in_set(
                        domain, hwloc_type, requested_index
                    )
                except (ValueError, RuntimeError, TypeError):
                    print(
                        f"Warning: Preferred index '{rule['prefer']}' invalid/not in domain '{domain}'. Falling back.",
                        file=sys.stderr,
                    )

            rank_index_in_gpu_group = local_rank // gpus_per_task
            return commands.hwloc_calc.get_object_in_set(
                domain, hwloc_type, rank_index_in_gpu_group
            )
        else:
            raise ValueError(f"Unsupported type '{hwloc_type}' for GPU locality binding.")

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
