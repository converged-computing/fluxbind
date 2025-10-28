from fluxbind.shape import Shape
from fluxbind.graph import Shape as GraphShape


def main(args, extra, **kwargs):
    """
    Parse a shape file to return the binding for a specific rank (local)
    """
    if args.graph:
        shape = GraphShape(args.file)
    else:
        shape = Shape(args.file)

    # 2. Call the public method to get the final binding string
    binding_string = shape.get_binding_for_rank(
        rank=args.rank,
        node_id=args.node_id,
        local_rank=args.local_rank,
        gpus_per_task=args.gpus_per_task,
    )

    # 3. Print the result to stdout for the wrapper script
    print(binding_string)
