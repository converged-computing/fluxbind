# In file: fluxbind/commands.py
import re
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

    def execute(self, args_list: list) -> str:
        """
        Executes hwloc-calc with a list of arguments.
        This is safer as it avoids shell interpretation of the arguments.
        """
        # A more robust validation could be added here if needed,
        command_list = [self.name] + args_list
        return self._run(command_list, shell=False)


class NvidiaSmiCommand(Command):
    name = "nvidia-smi"

    def get_pci_bus_ids(self) -> list[str]:
        """
        Specifically queries for and returns a list of GPU PCI bus IDs.
        The command is hardcoded for security.
        """
        command_str = f"{self.name} --query-gpu=pci.bus_id --format=csv,noheader"

        # shell=True is safe here because the entire command is static and defined internally.
        output = self._run(command_str, shell=True)

        # Parse the output into a clean list
        ids = output.strip().split("\n")
        return [bus_id for bus_id in ids if bus_id]


hwloc_calc = HwlocCalcCommand()
nvidia_smi = NvidiaSmiCommand()
