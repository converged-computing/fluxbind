#!/usr/bin/env python

import json
import sys

from fluxbind.bind import CommandLineRunner, JobSpecRunner, validate_result
from fluxbind.topology import TopologyParser


def main(args, extra, **kwargs):
    """
    Determine if a jobspec can be satisfied by local resources.
    This is a fairly simple (flat) check.
    """
    expression = []

    # Need to delete args.command
    delattr(args, "command")

    # Did we get a jobspec, or assembling on the fly?
    # Note that if we get a jobspec, a lot of the affinity args are added
    # as attributes. We likely should add more.
    if args.jobspec is not None:
        runner = JobSpecRunner(args.jobspec, **vars(args))
    else:
        runner = CommandLineRunner(**vars(args), command=extra)

    # This is organized by flux task rank. Here is one with cpu affinity per task -N1 -n2
    # {1: {'cpuset': '0x0000000c', 'cores': '2,3'},
    #  0: {'cpuset': '0x00000003', 'cores': '0,1'}}
    actual_layouts = runner.run()

    # If no expression, just print the layout
    if not expression:
        sys.exit(0)

    # Make predictions based on the topology we are expecting
    topology = TopologyParser()
    if args.xml is not None:
        topology.load_from_file(args.xml)
    else:
        topology.load_from_lstopo()

    print(f"Successfully loaded topology: {topology}.")
    print(f"Evaluating expression: {expression}.")

    # These are calculated masks for the topology desired.
    # The function assumes you are providing in the order you want,
    # e.g., ranks 0..N
    masks = topology.evaluate_expression(expression)

    # We assume that the topology provided is in the order of ranks. E.g.,
    # core:0-1 x numa:0 AND numa:0 x core:1-2
    # Says: rank 0 has core:0-1 x numa:0
    #       rank 1 has numa:0 x core:1-2
    rank = 0
    predicted_layouts = {}
    for group, mask in masks.items():
        logical_list, core_list = topology.predict_binding_outputs(mask)
        predicted_layouts[rank] = {
            "cpuset": mask,
            "cores": core_list,
            "pmu": logical_list,
        }
        rank += 1

    print(f"Predicted layout: {json.dumps(predicted_layouts, indent=4)}")

    # Now evaluate each one.
    if len(predicted_layouts) != len(actual_layouts):
        print("Error: predicted length != actual, check your expressions?:")
        print(f" Predicted: {json.dumps(predicted_layouts, indent=4)}\n")
        print(f" Actual: {json.dumps(actual_layouts, indent=4)}")
        sys.exit(1)

    print("\nValidating Results")
    mismatches = 0
    for rank, actual_layout in actual_layouts.items():
        if rank not in predicted_layouts:
            raise ValueError(f"Rank {rank} was not predicted.")
        predicted_layout = predicted_layouts[rank]
        success = validate_result(rank, predicted_layout, actual_layout)
        if not success:
            mismatches += 1

    if mismatches == 0:
        print("\n🎉 Implicit exclusivity and dense packing are working as predicted.")
    else:
        print(f"\n💔 Found {mismatches} binding mismatches.")
