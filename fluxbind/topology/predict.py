import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml


def read_yaml_file(filename):
    """
    Read YAML from a filename
    """
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def parse_indices(index_str):
    """
    This can parse a number or range into actual indices.
    E.g., 0, 0-3, etc.
    """
    indices = set()
    for part in index_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(list(indices))


class TopologyParser:
    """
    A class to fetch, parse, and analyze hwloc topology data.
    """

    def __init__(self):
        self.xml_content = None
        self.core_map = {}
        self.numa_map = {}

    def load_from_file(self, file_path):
        """
        Load from XML file.
        """
        print(f"Reading topology from provided file: {file_path}")
        with open(file_path, "r") as f:
            self.xml_content = f.read()
        self.parse()

    def load_from_lstopo(self):
        """
        Use lstopo to derive topology of running system.
        """
        print("No topology file provided. Generating live topology using 'lstopo'...")
        command = ["lstopo", "-p", "--no-io", "--output-format", "xml"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.xml_content = result.stdout
        self.parse()

    def parse(self):
        """
        Parse the xml topology into cores, pus, etc.
        """
        if not self.xml_content:
            raise RuntimeError("Cannot parse, no XML content loaded.")

        root = ET.fromstring(self.xml_content)
        self.core_map = {}
        self.numa_map = {}

        # Each core has some number of PU (processing units) under it.
        # I think for affinity we need to care about both, although notably
        # flux only is looking at the level of a core.
        for element in root.findall('.//object[@type="Core"]'):
            try:
                core_id = int(element.get("os_index"))
                pu_ids = [int(pu.get("os_index")) for pu in element.findall('./object[@type="PU"]')]
                if pu_ids:
                    self.core_map[core_id] = sorted(pu_ids)
            except Exception as e:
                print(f"Warning: issue with {element}: {e}")

        # The numa nodes contain the Core -> PU units
        for element in root.findall('.//object[@type="NUMANode"]'):
            try:
                numa_id = int(element.get("os_index"))
                cpuset = element.get("cpuset")
                if cpuset:
                    self.numa_map[numa_id] = cpuset
            except Exception as e:
                print(f"Warning: issue with {element}: {e}")

    # --- NEW HELPER 1: Handles comma-separated hex strings ---
    def _parse_cpuset_str_to_list(self, cpuset_str: str) -> list[int]:
        """Converts a potentially comma-separated hex string into a list of integers."""
        return [int(chunk, 16) for chunk in cpuset_str.strip().split(",")]

    # --- NEW HELPER 2: Performs bitwise operations on lists of integers ---
    def _operate_on_cpuset_lists(
        self, list_a: list[int], list_b: list[int], operator: str
    ) -> list[int]:
        """Performs a bitwise operation on two lists of cpuset integers, padding with 0."""
        max_len = max(len(list_a), len(list_b))
        result_list = []
        for i in range(max_len):
            val_a = list_a[i] if i < len(list_a) else 0
            val_b = list_b[i] if i < len(list_b) else 0

            if operator == "+":
                result_list.append(val_a | val_b)
            elif operator == "x":
                result_list.append(val_a & val_b)
            elif operator == "^":
                result_list.append(val_a ^ val_b)
            elif operator == "~":
                result_list.append(val_a & ~val_b)
        return result_list

    def evaluate_expression(self, expression_list):
        """
        Calculates a list of cpusets by splitting the expression into groups
        using 'AND' as a separator. Can also handle one group, obviously.
        """
        if not expression_list:
            return []

        groups = []
        current_group = []
        for item in expression_list:
            # numa:0 x core:0-1 AND numa:0 + core:2-3
            if item.upper() == "AND":
                if not current_group:
                    raise ValueError("Found 'AND' separator without a preceding group expression.")
                groups.append(current_group)
                # Reset for next group
                current_group = []
            else:
                current_group.append(item)

        # Append the final group after the loop finishes
        if not current_group:
            raise ValueError("Expression cannot end with a dangling 'AND' separator.")
        groups.append(current_group)

        # Evaluate each group expression to get its integer mask
        # A group can share the same mask (they have access to same cpuset)
        # so we need to include the order (we assume to be rank)
        final_masks = {}
        for i, group in enumerate(groups):
            mask_list = self.evaluate_group(group)
            expression = " ".join(group)
            # Format the list back into a canonical, 64-bit padded string
            formatted_mask = ",".join([f"0x{chunk:016x}" for chunk in mask_list])
            final_masks[f"{i} {expression}"] = formatted_mask
        return final_masks

    def evaluate_group(self, expression_list):
        """
        Calculate a final cpuset by evaluating an expression of locations and operators.
        E.g., an expression list will be like ["numa:0", "x", "core:0-7"]. See the comments
        below for examples.
        """
        if not expression_list:
            return [0]

        # Start with the mask of the very first location.
        current_mask_list = self.calculate_cpuset_for_location(expression_list[0])

        # Create an iterator for the rest of the list, which should be operator-location pairs.
        it = iter(expression_list[1:])

        for operator in it:
            try:
                # After an operator, there MUST be another location.
                location = next(it)
            except StopIteration:
                raise ValueError(f"Error: Operator '{operator}' must be followed by a location.")

            # Note: this is also a list of integers
            location_mask_list = self.calculate_cpuset_for_location(location)

            # All bitwise operations are replaced by a call to the new helper
            current_mask_list = self._operate_on_cpuset_lists(
                current_mask_list, location_mask_list, operator
            )

        # Return the raw list of integers. Formatting is handled by the caller.
        return current_mask_list

    def calculate_cpuset_for_location(self, location_str):
        """
        Private helper to calculate the cpuset for a single location string.
        Returns the mask as a LIST of integers for easy combination.
        I discovered with corona.xml we can have lists of things :)
        """
        if location_str.lower().startswith("0x"):
            return self._parse_cpuset_str_to_list(location_str)

        # Give the user feedback the input provided sucks
        try:
            obj_type, obj_index_str = location_str.lower().split(":", 1)
        except Exception as e:
            raise e

        try:
            indices = parse_indices(obj_index_str)
        except ValueError:
            print(f"Error: Invalid index format in '{location_str}'.", file=sys.stderr)
            return None

        # We expect to get a pu, numa, or core. We can eventually add gpus to this.
        # But I'm not sure how they show up in the topology / if the logic is the same.
        if obj_type == "numa":
            # Logic to handle union of multi-chunk cpusets
            total_mask_list = [0]
            for index in indices:
                if index not in self.numa_map:
                    raise ValueError(f"Error: NUMA node index {index} not found.")
                numa_mask_list = self._parse_cpuset_str_to_list(self.numa_map[index])
                total_mask_list = self._operate_on_cpuset_lists(
                    total_mask_list, numa_mask_list, "+"
                )
            return total_mask_list

        target_pus = set()
        if obj_type == "core":
            for index in indices:
                if index not in self.core_map:
                    raise ValueError(f"Error: Core index {index} not found.")
                target_pus.update(self.core_map[index])

        elif obj_type == "pu":
            max_pu = self.num_pus - 1
            for index in indices:
                if not (0 <= index <= max_pu):
                    raise ValueError(f"Error: PU index {index} is out of range (0-{max_pu}).")
            target_pus.update(indices)
        else:
            raise ValueError(f"Error: Unsupported type '{obj_type}'. Use 'numa', 'core', or 'pu'.")

        # Build a list of integer chunks from the set of PUs
        if not target_pus:
            return [0]

        max_pu_id = max(target_pus)
        num_chunks = (max_pu_id // 64) + 1
        final_mask_list = [0] * num_chunks

        for pu_id in target_pus:
            chunk_index = pu_id // 64
            bit_in_chunk = pu_id % 64
            final_mask_list[chunk_index] |= 1 << bit_in_chunk
        return final_mask_list

    @property
    def num_cores(self):
        """
        Number of physical cores found in the topology.
        """
        return len(self.core_map)

    @property
    def num_pus(self):
        """
        Total number of Processing Units (PUs) found.
        """
        return sum(len(pus) for pus in self.core_map.values())

    def predict_binding_outputs(self, cpuset_mask_str):
        """
        Predicts the logical PU list and the fully-contained core list for a
        given cpuset mask.
        """
        # Guard clause: if the topology hasn't been parsed, we can't make predictions.
        if not self.core_map:
            raise RuntimeError("Topology not parsed. Cannot make predictions.")

        # Important: the cpuset can be a comma-separated list. Parse it correctly.
        mask_list = self._parse_cpuset_str_to_list(cpuset_mask_str)
        bound_pus_set = set()

        # Iterate through each 64-bit chunk to build the complete set of PUs
        for chunk_index, chunk_val in enumerate(mask_list):
            base_pu_id = chunk_index * 64
            for bit_index in range(64):
                # Check if the bit at this position is set
                if (chunk_val >> bit_index) & 1:
                    bound_pus_set.add(base_pu_id + bit_index)

        # First prediction: the 'logical_cpu_list' (like 'taskset -c')
        # To create the human-readable string of PUs (like '0,1,2,3'):
        # 1. Convert the set of bound PUs into a sorted list
        logical_cpu_list = ",".join(map(str, sorted(list(bound_pus_set))))

        # Prediction 2: The 'cores' list (like 'hwloc-calc --intersect core')
        fully_contained_cores = []

        # Iterate through each physical core that was found in the topology.
        # We sort by core_id to ensure the final output list is also sorted.
        # I think this is OK to do.
        for core_id, pus_in_core in sorted(self.core_map.items()):
            # Check if a core is "fully contained" meaning all PUs are in the subset of
            # the known PUs for the core (allowed by the PU set mask)
            # E.g., if core 0 requires PUs {0, 1} and the mask allows {0, 1, 2, 3},
            # then {0, 1} is a subset of {0, 1, 2, 3}, and the condition is true.
            if set(pus_in_core).issubset(bound_pus_set):
                # If the core is fully contained, add its ID to our results list.
                fully_contained_cores.append(core_id)

        # Let's consider both.
        cores_list = ",".join(map(str, fully_contained_cores))
        return logical_cpu_list, cores_list

    def __str__(self):
        return f"Topology with {len(self.numa_map)} NUMA nodes, {self.num_cores} cores, and {self.num_pus} PUs."
