import shlex
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
            print(f"Error running '{cmd_str}': {e.stderr}", file=sys.stderr)
            raise RuntimeError("Command execution failed.") from e
        except FileNotFoundError as e:
            cmd_str = command[0] if isinstance(command, list) else command.split()[0]
            raise RuntimeError(f"Command not found: '{cmd_str}'") from e


class HwlocCalcCommand(Command):
    name = "hwloc-calc"

    def count(self, hw_type: str, within: str = "machine:0") -> int:
        """
        Returns the total number of a specific hardware object.

        Args:
            hw_type: The type of object to count (e.g., "core", "numa").
            within_object: Optional object to restrict the count to (e.g., "numa:0").
        """
        try:
            args = ["--number-of", hw_type, within]
            result_stdout = self.run([self.name] + args)
            return int(result_stdout)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to count number of '{hw_type}': {e}")

    def list_cpusets(self, hw_type: str, within: str = "machine:0") -> list[str]:
        """
        Returns a list of cpuset strings for each object of a given type.

        Args:
            hw_type: The type of object to list (e.g., "numa").
            within_object: Optional object to restrict the list to.
        """
        try:
            # Get the indices of all objects of this type
            args_intersect = ["--intersect", hw_type, within]
            indices_str = self.run([self.name] + args_intersect)
            indices = indices_str.split(",")

            # Cut out early
            if not indices or not indices[0]:
                return []

            # For each index, get its specific cpuset
            return [self.run([self.name, f"{hw_type}:{i}"]) for i in indices]
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to list cpusets for '{hw_type}': {e}")

    def get_cpuset(self, location: str) -> str:
        """
        Gets the cpuset for a single, specific location string (e.g., "pci=...", "core:0").
        """
        return self.run([self.name, location])

    def get_object_in_set(self, cpuset: str, obj_type: str, index: int) -> str:
        """
        Gets the Nth object of a type within a given cpuset.
        e.g., find the 1st 'core' within cpuset '0x00ff'.
        """
        # This uses the robust two-step process internally
        all_objects_str = self.run([self.name, cpuset, "--intersect", obj_type])
        available_indices = all_objects_str.split(",")
        try:
            target_index = available_indices[index]
            return f"{obj_type}:{target_index}"
        except IndexError:
            raise ValueError(f"Cannot find the {index}-th '{obj_type}' in cpuset {cpuset}.")


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


hwloc_calc = HwlocCalcCommand()
nvidia_smi = NvidiaSmiCommand()
