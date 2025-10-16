import json
import subprocess
import sys


class Command:
    """
    Abstract base class for a controlled command.
    """

    def run(self, command, shell: bool = False):
        """
        Private helper to run a subprocess command.
        """
        try:
            result = subprocess.run(
                command, shell=shell, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            cmd_str = command if shell else " ".join(command)
            raise RuntimeError("Command execution failed.") from e
        except FileNotFoundError as e:
            cmd_str = command[0] if isinstance(command, list) else command.split()[0]
            raise RuntimeError(f"Command not found: '{cmd_str}'") from e


class HwlocCalcCommand(Command):
    name = "hwloc-calc"

    def parse_cpuset_to_list(self, cpuset_str: str) -> list[int]:
        """
        Convert a potentially comma-separated hex string into a list of integers.
        """
        if not cpuset_str or cpuset_str.lower() in ["0x0", "0"]:
            return [0]
        return [int(chunk, 16) for chunk in cpuset_str.strip().split(",")]

    def operate_on_lists(self, list_a: list[int], list_b: list[int], operator: str) -> list[int]:
        """
        Perform a bitwise operation on two lists of cpuset integers.
        """
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
            else:
                raise ValueError(f"Unsupported operator '{operator}'")
        return result_list

    def count(self, hw_type: str, within: str = "machine:0") -> int:
        """
        Returns the total number of a specific hardware object.
        """
        try:
            args = ["--number-of", hw_type, within]
            result_stdout = self.run([self.name] + args, shell=False)
            return int(result_stdout)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to count number of '{hw_type}': {e}")

    def list_cpusets(self, hw_type: str, within: str = "machine:0") -> list[str]:
        """
        Returns a list of cpuset strings for each object of a given type.
        """
        try:
            args_intersect = ["--intersect", hw_type, within]
            indices_str = self.run([self.name] + args_intersect, shell=False)
            indices = indices_str.split(",")
            if not indices or not indices[0]:
                return []
            return [self.run([self.name, f"{hw_type}:{i}"], shell=False) for i in indices]
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to list cpusets for '{hw_type}': {e}")

    def get_cpuset(self, location: str) -> str:
        """
        Gets the cpuset for one or more space/operator-separated location strings.
        """
        return self.run(f"{self.name} {location}", shell=True)

    def get_object_in_set(self, cpuset: str, obj_type: str, index: int) -> str:
        """
        Gets the Nth object of a type within a given cpuset.
        """
        list_cmd = f"{self.name} '{cpuset}' --intersect {obj_type}"
        all_indices_str = self.run(list_cmd, shell=True)
        
        # Special case asking for all.
        if index == "all":
            return all_indices_str

        available_indices = all_indices_str.split(",")
        try:
            target_index = available_indices[index]
            return f"{obj_type}:{target_index}"
        except IndexError:
            raise ValueError(f"Cannot find the {index}-th '{obj_type}' in cpuset {cpuset}.")

    def union_of_locations(self, locations: list[str]) -> str:
        """
        Calculates the union of a list of hwloc location strings using Python logic.
        Returns a single, SPACE-separated string of hex cpusets.
        """
        union_mask_list = [0]

        for loc in locations:
            loc_cpuset_str = self.get_cpuset(loc)
            loc_cpuset_list = self.parse_cpuset_to_list(loc_cpuset_str)
            union_mask_list = self.operate_on_lists(union_mask_list, loc_cpuset_list, "+")
        return " ".join([hex(chunk) for chunk in union_mask_list])


class NvidiaSmiCommand(Command):
    name = "nvidia-smi"

    def get_pci_bus_ids(self) -> list[str]:
        """
        Specifically queries for and returns a list of GPU PCI bus IDs.
        The command is hardcoded for security.
        """
        command_str = f"{self.name} --query-gpu=pci.bus_id --format=csv,noheader"

        # shell=True is safe here because the entire command is static and defined internally.
        output = self.run(command_str, shell=True)

        # Parse the output into a clean list
        ids = output.strip().split("\n")
        return [bus_id for bus_id in ids if bus_id]


class RocmSmiCommand(Command):
    name = "rocm-smi"

    def get_pci_bus_ids(self) -> list[str]:
        """
        Specifically queries for and returns a list of GPU PCI bus IDs.
        """
        # The '--showbus' and '--json' flags provide a reliable, machine-readable output.
        command_str = f"{self.name} --showbus --json"

        # {"card0": {"PCI Bus": "0000:03:00.0"}, "card1": ..., "card7": {"PCI Bus": "0000:E3:00.0"}}
        output = self.run(command_str, shell=True)
        data = json.loads(output)

        pci_ids = []
        # I'm choosing not to sort so the devices are read in the order provided.
        for card_key in data.keys():
            card_info = data[card_key]
            pci_ids.append(card_info.get("PCI Bus"))
        return pci_ids


hwloc_calc = HwlocCalcCommand()
nvidia_smi = NvidiaSmiCommand()
rocm_smi = RocmSmiCommand()
