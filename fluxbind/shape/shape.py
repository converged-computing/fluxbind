import subprocess

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
