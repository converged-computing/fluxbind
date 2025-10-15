from dataclasses import dataclass

@dataclass
class GPUAssignment:
    """
    A data structure to hold information about a rank's assigned GPU(s).
    Instances are created via the for_rank() classmethod.
    """
    indices: list[int]          # The logical indices in the ordered list (e.g., [4, 5])
    pci_ids: list[str]          # The corresponding PCI bus IDs of the GPUs
    numa_indices: set[int]      # The set of unique NUMA nodes these GPUs are on
    cuda_devices: str           # The final string for CUDA_VISIBLE_DEVICES (e.g., "4,5")

    @classmethod
    def for_rank(
        cls, 
        local_rank: int, 
        gpus_per_task: int, 
        ordered_gpus: list[dict]
    ) -> "GPUAssignment":
        """
        A factory method that assigns a slice of GPUs to a given local rank
        from a pre-ordered, topology-aware list of all GPUs.
        """
        if not ordered_gpus:
            raise RuntimeError("Attempted to assign a GPU, but no GPUs were discovered.")
        
        start_idx = local_rank * gpus_per_task
        end_idx = start_idx + gpus_per_task

        if end_idx > len(ordered_gpus):
            raise ValueError(
                f"Cannot satisfy request for {gpus_per_task} GPUs for local_rank {local_rank}. "
                f"Only {len(ordered_gpus)} GPUs available in total."
            )

        assigned_gpu_slice = ordered_gpus[start_idx:end_idx]
        
        # The global indices for CUDA_VISIBLE_DEVICES are their positions in the ordered list
        assigned_indices = list(range(start_idx, end_idx))
        
        return cls(
            indices=assigned_indices,
            pci_ids=[gpu['pci_id'] for gpu in assigned_gpu_slice],
            numa_indices={gpu['numa_index'] for gpu in assigned_gpu_slice},
            cuda_devices=",".join(map(str, assigned_indices))
        )