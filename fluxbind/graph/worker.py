import logging
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


def get_numa_affinity(gp_index, data):
    """
    Worker function to get NUMA affinity for an object of interest.
    E.g., the data is from hwloc. It will have a type and/or pci_busis.
    If it has a type, we can use the os_index to get distance. Note that
    this is functionally and numerically correct, but in practice we can
    sometimes see that a Core is reported closer to ANOTHER NUMA node. This
    becomes a choice of predictibility (trust the hardware layout, the hwloc
    xml) vs. peak performance (trust the output of hwloc calc). I think
    for a tool like this we need to trust predictibility.
    """
    hwloc_obj_str = ""
    if "cpuset" in data:
        hwloc_obj_str = f"\"{data['cpuset']}\""
    elif data.get("type") in ["Core", "PU", "NUMANode"] and "os_index" in data:
        hwloc_obj_str = f"{data['type'].lower()}:{data['os_index']}"
    elif data.get("pci_busid"):
        hwloc_obj_str = f"pci={data['pci_busid']}"

    if not hwloc_obj_str:
        return None

    try:
        cmd = ["hwloc-calc", hwloc_obj_str, "--nodelist"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        output = result.stdout.strip()

        if ":" in output:
            # Handles "numa:0", "node:0", etc.
            numa_str = output.split(":")[-1]
        else:
            # Handles just "0"
            numa_str = output

        if not numa_str:
            log.debug(
                f"hwloc-calc for {hwloc_obj_str} gave empty nodelist string: '{result.stdout}'"
            )
            return None

        numa_index = int(numa_str)
        return gp_index, numa_index

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
        IndexError,
        subprocess.TimeoutExpired,
    ) as e:
        # This will now catch genuine errors, not simple parsing failures.
        log.debug(f"hwloc-calc failed for {hwloc_obj_str}: {e}")
        return None


class AffinityCalculator:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or int(multiprocessing.cpu_count() / 2)

    def calculate_numa_affinity(self, objects):
        """
        Calculate NUMA affinities for objects to locate using a thread pool executor.
        This runs SO much faster like this than in serial!
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_gp = {
                executor.submit(get_numa_affinity, gp, data): gp for gp, data in objects
            }
            for future in as_completed(future_to_gp):
                if result := future.result():
                    yield result
