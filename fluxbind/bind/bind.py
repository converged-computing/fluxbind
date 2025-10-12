#!/usr/bin/env python3

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

import yaml

here = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(here)

scripts = {
    "report": os.path.join(root, "scripts", "report_mapping.sh"),
    "run": os.path.join(root, "scripts", "run_mapping.sh"),
}

bash = shutil.which("bash")

for _, script in scripts.items():
    if not os.path.exists(script):
        raise ValueError(f"{script} does not exist.")

if not bash:
    raise ValueError("Cannot find bash command... uhh.")


def parse_indices(index_str):
    """
    This can parse a number or range into actual indices.
    E.g., 0, 0-3, etc.
    """
    indices = set()
    for part in index_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return ",".join(str(x) for x in sorted(list(indices)))


def read_yaml_file(filename):
    """
    Read YAML from a filename
    """
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def stream(cmd):
    """
    Executes a command and streams its stdout and stderr to the console
    """
    captured_stdout = []
    captured_stderr = []
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"
    )

    def reader_thread(pipe, output_list, prefix):
        try:
            for line in pipe:
                line_stripped = line.strip()
                # I'm printing in the output parser instead
                # because it's formatted better. You can print here for debugging.
                # print(line_stripped, flush=True)
                output_list.append(line_stripped)
        finally:
            pipe.close()

    # Last argument is a prefix
    stdout_thread = threading.Thread(
        target=reader_thread, args=(process.stdout, captured_stdout, "")
    )
    stderr_thread = threading.Thread(
        target=reader_thread, args=(process.stderr, captured_stderr, "")
    )
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    return_code = process.wait()
    return return_code, "\n".join(captured_stdout) + "\n".join(captured_stderr)


class CommandLineRunner:
    def __init__(
        self,
        nodes,
        tasks,
        exclusive=False,
        cores_per_task=None,
        cpu_affinity=None,
        taskmap=None,
        command=None,
        shape=None,
        env=None,
        **kwargs,
    ):
        """
        Assemble a flux run command from command line flags, etc.
        """
        self.nodes = nodes
        self.tasks = tasks
        self.exclusive = exclusive
        self.cores_per_task = cores_per_task
        self.cpu_affinity = cpu_affinity
        self.taskmap = taskmap
        self.command = command
        self.shape = shape
        self.env = env

    def parse_binding_output(self, raw_output):
        """
        Given an output string, parse the binding information.
        """
        actual_layout = {}

        collecting = True
        for line in raw_output.strip().split("\n"):
            parts = line.split()
            if "PEWPEWPEW" not in line and "PEWSTOP" not in line:
                print(line)
            if "PEWPEWPEW" not in line:
                continue
            if "PEWSTOP" in line:
                collecting = False
                print()

            if not collecting:
                continue

            # PEWPEWPEW
            parts.pop(0)

            # The cpuset is a mask that picks out the cores
            # $rank $node $cpuset_mask $logical_cpu_list $taskset_list" >&2
            print(parts)
            rank_id, node_name, cpuset, logical_pus, physical_cores, taskset_pus = parts
            rank_id = int(rank_id)
            actual_layout[rank_id] = {
                "cpuset": cpuset,
                "cores": physical_cores,
                "pmu": logical_pus,
            }
        return actual_layout

    def run(self):
        """
        Runs the Flux job and captures the binding report from each rank.

        This is a run from the command line (non jobspec)
        """
        return self.execute(scripts["run"])

    def report(self):
        """
        Runs the Flux job and captures the binding report from each rank.

        This is a run from the command line (non jobspec)
        """
        return self.execute(scripts["report"])

    def get_custom_command(self):
        """
        A custom command uses flux to ask for a specific binding
        """
        cmd = ["flux", "run", "-N", str(self.nodes)]
        if self.tasks is not None:
            cmd += ["-n", str(self.tasks)]
        if self.cpu_affinity is not None:
            cmd += ["-o", f"cpu-affinity={self.cpu_affinity}"]
        if self.exclusive:
            cmd.append("--exclusive")
        if self.cores_per_task is not None:
            cmd += ["--cores-per-task", str(self.cores_per_task)]
        if self.taskmap is not None:
            cmd += [f"--taskmap={self.taskmap}"]
        if self.env is not None:
            for envar in self.env:
                cmd += ["--env", envar]
        return cmd

    def get_shape_command(self):
        """
        A shape command requires exclusive (for now) and then exports
        (provides) the JOB_SHAPE_FILE to the job.
        """
        cmd = [
            "flux",
            "run",
            "-N",
            str(self.nodes),
            "--exclusive",
            "--env",
            f"JOB_SHAPE_FILE={self.shape}",
        ]
        if self.tasks is not None:
            cmd += ["-n", str(self.tasks)]
        if self.cpu_affinity is not None:
            cmd += ["-o", f"cpu-affinity={self.cpu_affinity}"]
        if self.env is not None:
            for envar in self.env:
                cmd += ["--env", envar]
        return cmd

    def execute(self, script):
        """
        Runs the Flux job and captures the binding report from each rank.

        This is a run from the command line (non jobspec)
        """
        print(f"\nRunning Experiment with {self.nodes} nodes and {self.tasks} tasks")
        if self.shape is not None:
            cmd = self.get_shape_command()
        else:
            cmd = self.get_custom_command()

        # We are basically running <flux> <report-mapping-script>
        # Flux starts the job, and the job inherits some mapping of cpusets.
        # We then use hwloc tools to inspect that mapping. The match policy
        # is important and expected to be consistent in the tests (low)
        cmd += [bash, script]
        if self.command not in [None, []]:
            cmd += self.command
        print(f"Executing: {shlex.join(cmd)}")

        try:
            return_code, raw_output = stream(cmd)
        except Exception as e:
            sys.exit(e.stderr)

        if return_code != 0:
            print("Warning, application did not return with 0 exit code.")
        print("\nParsing actual bindings from job output...")
        return self.parse_binding_output(raw_output)


class JobSpecRunner(CommandLineRunner):
    """
    The Jobspec running loads and runs a job from a yaml jobspec
    """

    def __init__(self, filename, **kwargs):
        self.load_jobspec(filename, **kwargs)

    def load_jobspec(self, filename, **kwargs):
        """
        Load jobspec from filename
        """
        # Assume it is yaml, throw up otherwise
        self.jobspec = read_yaml_file(filename)
        # We need to update the command
        self.jobspec["tasks"][0]["command"] = [bash, report_mapping]
        self.args = kwargs

    def run(self):
        """
        From from a Jobspec using Flux
        """
        import flux
        import flux.job

        handle = flux.Flux()
        js = flux.job.Jobspec(**self.jobspec)
        js.environment = dict(os.environ)

        # CPU affinity is added as a shell option.
        cpu_affinity = self.args.get("cpu_affinity")
        if cpu_affinity is not None:
            js.setattr_shell_option("cpu-affinity", cpu_affinity)

        job_id = flux.job.submit(handle, js)
        time.sleep(2)
        output = flux.job.output.job_output(handle, job_id)
        return self.parse_binding_output(output.stdout + output.stderr)


def validate_result(rank, prediction, actual):
    """
    Compares the predicted layout to the actual layout and reports PASS/FAIL.
    """
    if not prediction or not actual:
        print("🔴 Could not generate predictions or get actual results.")
        return False

    for field in ["cpuset", "cores", "pmu"]:
        predicted_value = prediction[field]
        actual_value = actual[field]
        if predicted_value == actual_value:
            print(f"🟢 Rank {rank} - {field} {actual_value}")
        else:
            print(
                f"🔴 Rank {rank} - Predicted {field}: {predicted_value}, Actual {field}: {actual_value}"
            )
            return False
    return True
