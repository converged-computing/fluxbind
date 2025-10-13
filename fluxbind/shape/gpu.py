from dataclasses import dataclass


@dataclass
class GPUAssignment:
    """
    Data structure to hold information about a rank's assigned GPU.
    """

    index: int  # logical index of the GPU
    pci_id: str  # The PCI bus ID of the GPU
    cuda_devices: str  # CUDA_VISIBLE_DEVICES

    @classmethod
    def for_rank(cls, local_rank, gpu_pci_ids):
        """
        A factory method that assigns a GPU to a given local rank
        using a round-robin strategy.
        """
        if not gpu_pci_ids:
            raise RuntimeError("Attempted to assign a GPU, but no GPUs were discovered.")

        num_gpus = len(gpu_pci_ids)
        target_gpu_index = local_rank % num_gpus

        # Return the assignment
        return cls(
            index=target_gpu_index,
            pci_id=gpu_pci_ids[target_gpu_index],
            cuda_devices=str(target_gpu_index),
        )
