"""CPU capacity visible to the interactive session."""
import os


def cpu_capacity():
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    # Some OOD launches export a request even when the kernel sees the entire node.
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(key, "")
        if value.isdecimal() and int(value) > 0:
            visible = min(visible, int(value))
    return visible
