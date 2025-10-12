#!/usr/bin/env python

import argparse
import os
import sys

# This will pretty print all exceptions in rich
from rich.traceback import install

from . import help

install()

import fluxbind
from fluxbind.logger import setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Flux Bind",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        help="use verbose logging to debug.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--version",
        dest="version",
        help="show software version.",
        default=False,
        action="store_true",
    )

    subparsers = parser.add_subparsers(
        help="actions",
        title="actions",
        description="actions",
        dest="command",
    )
    subparsers.add_parser("version", description="show software version")
    run = subparsers.add_parser(
        "run",
        formatter_class=argparse.RawTextHelpFormatter,
        description="flux run with binding",
    )
    run.add_argument(
        "--exclusive",
        help="Request exclusive",
        default=False,
        action="store_true",
    )
    run.add_argument(
        "--shape",
        default=None,
        help="Specify shape file for execution",
    )
    run.add_argument(
        "--cpu-affinity",
        default=None,
        help="Add cpu-affinity",
        # choices=["none", "per-task"],
    )
    run.add_argument("-N", "--nodes", type=int, default=1, help="The number of nodes (default: 1).")
    run.add_argument(
        "-e",
        "--env",
        type=str,
        action="append",
        default=None,
        help="One or more environment variables.",
    )
    run.add_argument("--jobspec", "-j", help="Provide a jobspec instead.", default=None)
    run.add_argument("--taskmap", help="Provide a custom taskmap.", default=None)
    run.add_argument(
        "-n",
        "--ntasks",
        dest="tasks",
        type=int,
        default=None,
        help="The number of tasks to use for the test run (default: 2).",
    )
    run.add_argument(
        "-c",
        "--cores-per-task",
        type=int,
        default=None,
        help="The number of CORES (not PUs) to bind per task.",
    )
    run.add_argument(
        "--tasks-per-core",
        type=int,
        default=None,
        help="The number of tasks per core.",
    )
    run.add_argument(
        "--silent",
        dest="silent",
        help="no additional output.",
        default=False,
        action="store_true",
    )
    run.add_argument(
        "--quiet",
        dest="quiet",
        help="suppress additional output (only print fluxbind mapping)",
        default=False,
        action="store_true",
    )
    run.add_argument(
        "--nocolor",
        help="suppress color output (e.g., piping to log)",
        default=False,
        action="store_true",
    )

    predict = subparsers.add_parser(
        "predict",
        formatter_class=argparse.RawTextHelpFormatter,
        description=help.predict_help,
    )
    predict.add_argument(
        "-m",
        "--mask-only",
        action="store_true",
        help="Only output the calculated hexadecimal cpuset mask.",
    )
    predict.add_argument("expression", nargs="+", help="An expression of locations and operators.")

    shape = subparsers.add_parser(
        "shape",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Parse a shape file and request from a local rank, get back the binding.",
    )
    shape.add_argument("--file", required=True, help="Path to the YAML shape file.")
    shape.add_argument("--rank", required=True, type=int, help="Global rank of the process.")
    shape.add_argument(
        "--node-id", required=True, type=int, help="Logical ID of the node in the allocation."
    )
    shape.add_argument(
        "--local-rank", required=True, type=int, help="Rank of the process on the local node."
    )

    for command in [predict, run]:
        command.add_argument(
            "--xml",
            "--topology-file",
            dest="topology_file",
            help="Path to a lstopo XML file.\nIf not provided, runs 'lstopo' do detect.",
        )

    return parser


def run_fluxbind():
    """
    this is the main entrypoint.
    """
    parser = get_parser()

    def help(return_code=0):
        version = fluxbind.__version__

        print("\nFlux Bind v%s" % version)
        parser.print_help()
        sys.exit(return_code)

    # If the user didn't provide any arguments, show the full help
    if len(sys.argv) == 1:
        help()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, extra = parser.parse_known_args()

    if args.debug is True:
        os.environ["MESSAGELEVEL"] = "DEBUG"

    # Show the version and exit
    if args.command == "version" or args.version:
        print(fluxbind.__version__)
        sys.exit(0)

    setup_logger(
        quiet=False,
        debug=args.debug,
    )

    # Here we can assume instantiated to get args
    if args.command == "run":
        from .run import main
    elif args.command == "predict":
        from .predict import main
    elif args.command == "shape":
        from .shape import main
    else:
        help(1)
    main(args, extra)


if __name__ == "__main__":
    run_fluxbind()
