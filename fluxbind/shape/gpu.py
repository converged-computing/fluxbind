from dataclasses import dataclass


@dataclass
class GPUAssignment:
    """
    Data structure to hold information about a rank's assigned GPU.
    """

    indices: list[int]  # logical index of the GPU
    pci_ids: list[str]  # The PCI bus ID of the GPU
    cuda_devices: str  # CUDA_VISIBLE_DEVICES

    @classmethod
    def for_rank(cls, local_rank, gpus_per_task=None, gpu_pci_ids=None):
        """
        A factory method that assigns a GPU to a given local rank
        using a round-robin strategy.
        """
        if not gpu_pci_ids:
            raise RuntimeError("Attempted to assign a GPU, but no GPUs were discovered.")

        # Assume one gpu per task, since we are calling this, period
        gpus_per_task = gpus_per_task or 1

        # 1. Calculate the starting GPU index for this rank.
        start_gpu_index = local_rank * gpus_per_task
        end_gpu_index = start_gpu_index + gpus_per_task

        if end_gpu_index > len(gpu_pci_ids):
            raise ValueError(
                f"Cannot satisfy request for {gpus_per_task} GPUs for local_rank {local_rank}. "
                f"Only {len(gpu_pci_ids)} GPUs available in total."
            )

        # Return the assignment
        assigned_indices = list(range(start_gpu_index, end_gpu_index))
        return cls(
            indices=assigned_indices,
            pci_ids=[gpu_pci_ids[i] for i in assigned_indices],
            cuda_devices=",".join([str(x) for x in assigned_indices]),
        )
