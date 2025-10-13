import re
import subprocess
import sys

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
        self.data = self.load_file(filepath)
        # This discovers and cache hardware properties on init
        # The expectation is that this is running from the node (task)
        self.num_cores = self.discover_hardware("core")
        self.num_pus = self.discover_hardware("pu")
        self.pus_per_core = self.num_pus // self.num_cores if self.num_cores > 0 else 0
        self.gpu_objects = self.discover_gpus()

    def discover_gpus(self) -> list:
        """
        Discovers available GPU objects using hwloc.
        """
        try:
            cmd = f"hwloc-calc --whole-system {self.machine} -I os --filter-by-type-name CUDA"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip().split()
        except subprocess.CalledProcessError:
            print("Warning: Could not discover any CUDA GPUs.", file=sys.stderr)
            return []

    def load_file(self, filepath):
        """
        Loads and parses the YAML shape file.
        """
        return utils.read_yaml(filepath)

    def discover_hardware(self, hw_type: str) -> int:
        """
        Runs hwloc-calc to get the number of a specific hardware object.
        """
        try:
            command = f"hwloc-calc --number-of {hw_type} {self.machine}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            # Better to raise an error than to silently return a default
            raise RuntimeError(f"Failed to determine number of '{hw_type}': {e}")

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
        processed_formula = str(formula_template)
        substitutions = re.findall(r"\{\{([^}]+)\}\}", processed_formula)
        for command_to_run in set(substitutions):
            try:
                result = subprocess.run(
                    command_to_run, shell=True, capture_output=True, text=True, check=True
                )
                placeholder = f"{{{{{command_to_run}}}}}"
                processed_formula = processed_formula.replace(placeholder, result.stdout.strip())
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error executing sub-command '{command_to_run}': {e}")

        # Substitute local_rank and evaluate final expression
        final_expression = processed_formula.replace("$local_rank", str(local_rank))
        command = f'echo "{final_expression}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()

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

    def get_gpu_binding_for_rank(self, on_domain, hwloc_type, local_rank):
        """
        Get a GPU binding for a rank. Local means a numa node close by, remote not.
        """
        if not self.gpu_objects:
            raise RuntimeError("Shape is GPU-aware, but no GPUs were discovered.")
        if local_rank >= len(self.gpu_objects):
            raise IndexError(
                f"local_rank {local_rank} is out of range for {len(self.gpu_objects)} GPUs."
            )

        my_gpu_object = self.gpu_objects[local_rank]

        pci_bus_id_cmd = f"hwloc-pci-lookup {my_gpu_object}"
        cuda_devices = subprocess.run(
            pci_bus_id_cmd, shell=True, capture_output=True, text=True, check=True
        ).stdout.strip()

        local_numa_cmd = f"hwloc-calc {my_gpu_object} --ancestor numa -I"
        local_numa_id = int(
            subprocess.run(
                local_numa_cmd, shell=True, capture_output=True, text=True, check=True
            ).stdout.strip()
        )

        target_numa_location = ""
        if on_domain == "gpu-local":
            target_numa_location = f"numa:{local_numa_id}"
        else:  # gpu-remote
            remote_numa_id = (local_numa_id + 1) % self.num_numa
            target_numa_location = f"numa:{remote_numa_id}"

        # If the requested type is just 'numa', we're done.
        if hwloc_type == "numa":
            return f"{target_numa_location},{cuda_devices}"

        # Otherwise, find the first object of the requested type WITHIN that NUMA domain.
        # This is a powerful composition of the two concepts.
        # E.g., find the first 'core' on the 'gpu-local' NUMA domain.
        cmd = f"hwloc-calc {target_numa_location} --intersect {hwloc_type} --first"
        binding_string = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{binding_string},{cuda_devices}"

    def get_binding_for_rank(self, rank: int, node_id: int, local_rank: int) -> str:
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

        on_domain = rule.get("on")
        if on_domain in ["gpu-local", "gpu-remote"]:
            return self.get_gpu_binding_for_rank(on_domain, hwloc_type, local_rank)

        cpu_binding_string = self.get_cpu_binding(hwloc_type, rule, local_rank)

        # Return the CPU binding and the "NONE" sentinel for the device.
        return f"{cpu_binding_string},NONE"

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
                total_objects = self.discover_hardware(hwloc_type)
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
