from fluxbind.graph import Shape as GraphShape
from fluxbind.shape import Shape


def main(args, extra, **kwargs):
    """
    Parse a shape file to return the binding for a specific rank (local)
    """
    # This input file is the shapefile
    if args.graph:
        # This is the (mostly) Flux jobspec with pattern, etc.
        shape = GraphShape(args.file, debug=args.debug)
    else:
        # This is a simple, custom design.
        shape = Shape(args.file)

    # Call the public method to get the final binding string
    binding_string = shape.get_binding_for_rank(
        rank=args.rank,
        node_id=args.node_id,
        local_rank=args.local_rank,
        local_size=args.local_size,
        gpus_per_task=args.gpus_per_task,
        xml_file=args.topology_file,
    )

    # Print the result to stdout for the wrapper script
    print(binding_string)
