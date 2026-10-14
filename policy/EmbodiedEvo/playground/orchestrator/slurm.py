"""Parse GPU indices for execution inside an existing Slurm allocation."""

def allocated_gpus(value):
    indices = set()
    for item in filter(None, value.split(",")):
        start, _, end = item.partition("-")
        indices.update(str(i) for i in range(int(start), int(end or start) + 1))
    return indices
