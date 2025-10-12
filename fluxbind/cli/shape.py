from fluxbind.shape import Shape


def main(args, extra, **kwargs):
    """
    Parse a shape file to return the binding for a specific rank (local)
    """
    shape = Shape(args.file)

    # 2. Call the public method to get the final binding string
    binding_string = shape.get_binding_for_rank(
        rank=args.rank, node_id=args.node_id, local_rank=args.local_rank
    )

    # 3. Print the result to stdout for the wrapper script
    print(binding_string)
