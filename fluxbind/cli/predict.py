#!/usr/bin/env python

import sys

from fluxbind.topology import TopologyParser


def main(args, extra, **kwargs):
    """
    Determine if a jobspec can be satisfied by local resources.
    This is a fairly simple (flat) check.
    """
    topology = TopologyParser()

    if args.topology_file:
        topology.load_from_file(args.topology_file)
    else:
        topology.load_from_lstopo()

    print(f"Successfully parsed: {topology}.")

    # These are calculated masks, with index by the expression
    masks = topology.evaluate_expression(args.expression)
    if args.mask_only:
        print(masks)
        sys.exit(0)

    for group, mask in masks.items():
        logical_list, core_list = topology.predict_binding_outputs(mask)
        print("\n" + "=" * 50)
        print(f"Group {group} resolved to Combined Cpuset Mask: {mask}")
        print("=" * 50)
        print("Predicted 'logical_cpu_list' (PUs in mask):")
        print(f"  -> {logical_list or '(none)'}\n")
        print("Predicted 'cores' list (fully contained):")
        print(f"  -> {core_list or '(none)'}")
        print("=" * 50)
