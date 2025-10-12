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
    Executes a command and streams its output to the console, preserving the
    interleaved order of stdout and stderr.
    """
    captured_output = []

    # Not affected by system time changes
    start_time = time.monotonic()

    # The key change: stderr=subprocess.STDOUT tells the OS to merge the streams.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        encoding="utf-8",
    )

    # Now we only need one reader thread for the single, merged stream.
    def reader_thread(pipe, output_list):
        for line in pipe:
            line_stripped = line.strip()
            print(line_stripped, flush=True)
            output_list.append(line_stripped)
        pipe.close()

    # We only have one pipe: process.stdout
    thread = threading.Thread(target=reader_thread, args=(process.stdout, captured_output))
    thread.start()

    # Wait for the reader thread to finish (it will when the process closes its pipe)
    thread.join()

    # Wait for the process to terminate and get its return code
    return_code = process.wait()
    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"fluxtime returncode: {return_code}")
    print(f"fluxtime duration: {duration} seconds")

    # The captured_output list now contains the perfectly ordered, interleaved output
    return return_code, "\n".join(captured_output)


class CommandLineRunner:
    def __init__(self, **kwargs):
        """
        Assemble a flux run command from command line flags, etc.
        """
        for name in [
            "nodes",
            "tasks",
            "exclusive",
            "cores_per_task",
            "tasks_per_core",
            "cpu_affinity",
            "taskmap",
            "command",
            "shape",
            "env",
            "quiet",
            "silent",
            "nocolor",
        ]:
            setattr(self, name, kwargs.get(name))

    def parse_binding_output(self, raw_output):
        """
        Given an output string, parse the binding information.
        """
        actual_layout = {}

        collecting = True
        for line in raw_output.strip().split("\n"):
            parts = line.split()
            if "PEWPEWPEW" not in line:
                continue
            if "PEWSTOP" in line:
                collecting = False

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
        if self.tasks_per_core is not None:
            cmd += ["--tasks-per-core", str(self.tasks_per_core)]
        if self.taskmap is not None:
            cmd += [f"--taskmap={self.taskmap}"]
        cmd = self.set_envars(cmd)
        return cmd

    def set_envars(self, cmd):
        """
        Shared function to set environment variables
        """
        if self.quiet:
            cmd += ["--env", "FLUXBIND_QUIET=1"]
        if self.nocolor:
            cmd += ["--env", "FLUXBIND_NOCOLOR=1"]
        if self.silent:
            cmd += ["--env", "FLUXBIND_SILENT=1"]
            cmd += ["--env", "FLUXBIND_QUIET=1"]
        if self.shape is not None:
            cmd += ["--env", f"JOB_SHAPE_FILE={self.shape}"]
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
        ]
        cmd = self.set_envars(cmd)
        if self.tasks is not None:
            cmd += ["-n", str(self.tasks)]
        if self.tasks_per_core is not None:
            cmd += ["--tasks-per-core", str(self.tasks_per_core)]
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
        if self.tasks is not None:
            print(f"\nRunning Experiment with {self.nodes} nodes and {self.tasks} tasks")
        else:
            print(f"\nRunning Experiment with {self.nodes} nodes")

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
