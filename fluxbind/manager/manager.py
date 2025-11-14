import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from fluxbind.graph.graph import HwlocTopology
from fluxbind.shape.commands import HwlocCalcCommand

log = logging.getLogger(__name__)


class NodeResourceManager:
    """
    A thread-safe, persistent manager for hardware resource allocations on a single node.

    This class provides a generic reservation system. It loads the node's hardware
    topology once, maintains a ledger of all reservations, and provides atomic
    operations to create and release reservations.
    """

    def __init__(self, state_file: str, xml_file: Optional[str] = None):
        """
        Initializes the NodeResourceManager.

        Args:
            state_file: The path to the file where allocation state will be persisted.
            xml_file: Optional. A path to a static lstopo XML file for testing.
                      If None, the topology is discovered from the live system.
        """
        log.info("Initializing NodeResourceManager...")
        self.state_file = state_file
        self._lock = threading.Lock()
        self._topology = HwlocTopology(xml_file)
        self._hwloc_calc = HwlocCalcCommand()

        # The core state: maps a unique reservation_id (e.g., K8s claim_id, job_id)
        # to the set of graph pointer (gp) indices it holds.
        # e.g., {'some-unique-id-123': {12345, 67890, ...}}
        self._reservations: Dict[str, Set[int]] = {}

        self._load_state()
        log.info(
            f"ResourceManager is ready. Tracking {len(self._reservations)} existing reservations."
        )

    def _load_state(self):
        """
        Loads reservation state from the state file, if it exists.
        """
        if not os.path.exists(self.state_file):
            log.info("State file not found. Starting with a fresh state.")
            return
        try:
            with open(self.state_file, "r") as f:
                loaded_state = json.load(f)
                # Convert the lists from JSON back into sets of integers
                self._reservations = {
                    reservation_id: set(gps) for reservation_id, gps in loaded_state.items()
                }
                log.info(
                    f"Successfully loaded state for {len(self._reservations)} reservations from {self.state_file}"
                )
        except (json.JSONDecodeError, IOError, TypeError) as e:
            log.error(
                f"Failed to load state from {self.state_file}: {e}. Starting with a fresh state."
            )
            self._reservations = {}

    def _save_state(self):
        """
        Persists the current reservation state to the state file.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                # Must convert sets to lists for JSON serialization
                serializable_state = {
                    reservation_id: list(gps) for reservation_id, gps in self._reservations.items()
                }
                json.dump(serializable_state, f, indent=2)
        except IOError as e:
            log.error(f"CRITICAL: Failed to save state to {self.state_file}: {e}")

    def _get_all_reserved_gps(self) -> Set[int]:
        """
        Flattens the reservations dict into a single set of all used GPs.
        """
        all_gps = set()
        for gps in self._reservations.values():
            all_gps.update(gps)
        return all_gps

    def _create_binding_string(
        self, total_allocation: List[Tuple[int, Dict]], bind_level: str = "core"
    ) -> str:
        """
        Transforms a list of allocated resources into a final cpuset and GPU string.
        """
        # Convert the high-level allocation to bindable leaf nodes (Cores or PUs)
        leaf_nodes = self._topology.find_bindable_leaves(total_allocation, bind_level)
        if not leaf_nodes:
            log.warning("Could not find any bindable leaf nodes for the allocation.")
            return "0x0;NONE"

        # Extract cpusets from the leaf nodes
        cpusets = []
        if bind_level == "pu":
            cpusets = [d["cpuset"] for _, d in leaf_nodes if "cpuset" in d]
        elif bind_level == "core":
            for core_gp, _ in leaf_nodes:
                pus = self._topology.get_descendants(core_gp, type="PU")
                for _pu_gp, pu_data in pus:
                    if pu_cpuset := pu_data.get("cpuset"):
                        cpusets.append(pu_cpuset)

        # Calculate the final cpuset mask
        mask = self._hwloc_calc.get_cpuset(" ".join(cpusets)) if cpusets else "0x0"

        # Extract assigned GPUs to create the CUDA_VISIBLE_DEVICES string
        assigned_gpus = [node for node in total_allocation if node[1].get("device_type") == "gpu"]
        gpu_indices = []
        if assigned_gpus:
            assigned_bus_ids = {gpu_data["pci_busid"] for _, gpu_data in assigned_gpus}
            for i, ordered_gpu in enumerate(self._topology.ordered_gpus):
                if ordered_gpu["pci_id"] in assigned_bus_ids:
                    gpu_indices.append(str(i))

        gpu_string = ",".join(sorted(gpu_indices)) or "NONE"
        return f"{mask};{gpu_string}"

    def _reconstruct_allocation(self, gps: Set[int]) -> List[Tuple[int, Dict]]:
        """Looks up a set of GPs in the graph to reconstruct an allocation list."""
        return [(gp, self._topology.graph.nodes[gp]) for gp in gps if gp in self._topology.graph]

    def create_reservation(self, reservation_id: str, jobspec: Dict) -> Optional[str]:
        """
        Finds and reserves resources for a given ID, returning a binding string.

        This method is thread-safe and idempotent.

        Args:
            reservation_id: A unique identifier for the resource reservation (e.g., a K8s claim ID or a job ID).
            jobspec: The resource request shape, compatible with `match_resources`.

        Returns:
            A binding string (e.g., "0xff...;0,1") on success, or None on failure.
        """
        with self._lock:
            # Idempotency: If this reservation already exists, it's a retry.
            # Reconstruct the binding from the stored state and return it.
            if reservation_id in self._reservations:
                log.warning(
                    f"Reservation '{reservation_id}' already exists. Reconstructing binding for idempotency."
                )
                stored_gps = self._reservations[reservation_id]
                reconstructed_allocation = self._reconstruct_allocation(stored_gps)
                return self._create_binding_string(reconstructed_allocation)

            # Get the set of all resources currently in use across all reservations.
            currently_reserved_gps = self._get_all_reserved_gps()

            # Call the powerful, stateless search function on the topology.
            log.info(f"Attempting to find resources for new reservation '{reservation_id}'...")
            allocation = self._topology.match_resources(jobspec, currently_reserved_gps)

            # If the search fails, we cannot satisfy the request.
            if allocation is None:
                log.error(f"Failed to find available resources for reservation '{reservation_id}'.")
                return None

            # Success! Update and persist the state.
            reserved_gps_for_id = {gp for gp, _ in allocation}
            self._reservations[reservation_id] = reserved_gps_for_id
            self._save_state()
            log.info(
                f"Successfully created reservation '{reservation_id}' with {len(reserved_gps_for_id)} resources."
            )

            # Convert the successful allocation into the final binding string.
            return self._create_binding_string(allocation)

    def release_reservation(self, reservation_id: str) -> bool:
        """
        Releases the resources held by a specific reservation.

        This method is thread-safe and idempotent.

        Args:
            reservation_id: The unique identifier of the reservation to release.

        Returns:
            True on success, even if the reservation was already released.
        """
        with self._lock:
            if reservation_id in self._reservations:
                log.info(f"Releasing reservation '{reservation_id}'.")
                del self._reservations[reservation_id]
                self._save_state()
            else:
                # Idempotency: If the caller retries a release, the reservation might
                # already be gone. This is expected and should not be an error.
                log.warning(
                    f"Release requested for unknown or already-released reservation '{reservation_id}'."
                )

        return True
