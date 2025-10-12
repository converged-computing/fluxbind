#!/bin/bash
#
# report_mapping.sh

rank="$FLUX_TASK_RANK"
node=$(hostname)


# 1. Get the cpuset mask given to this process by Flux
cpuset_mask=$(hwloc-bind --get --pid $$)

# 2. Convert the mask to a human-readable list of logical CPUs (PUs)
logical_cpu_list=$(hwloc-calc "$cpuset_mask" --intersect PU)

# 3. Convert the mask to a list of fully contained physical cores (not logical)
physical_core_list=$(hwloc-calc "$cpuset_mask" --intersect core)

# 4. Get the affinity list using the standard Linux 'taskset' for verification
taskset_list=$(taskset -c -p $$ | awk '{print $NF}')


YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RESET='\033[0m'
BLUE='\e[0;34m'
CYAN='\e[0;36m'
MAGENTA='\e[0;35m'
ORANGE='\033[0;33m'

prefix="${YELLOW}rank ${rank}${RESET}"
echo -e "${prefix}: Raw cpuset mask is:      ${CYAN}$cpuset_mask${RESET}"
echo -e "${prefix}: Logical CPUs (PUs):      ${BLUE}$logical_cpu_list${RESET}"
echo -e "${prefix}: Physical Cores:          ${ORANGE}${physical_core_list:-none}${RESET}"
echo -e "${prefix}: Taskset Affinity (PUs):  ${MAGENTA}$taskset_list${RESET}"
echo -e "${prefix}: PID for this rank:       ${GREEN}$$ ${RESET}"
echo

# Print the machine-parseable line to standard error for logging
echo "PEWPEWPEW $rank $node $cpuset_mask $logical_cpu_list ${physical_core_list:-NA} $taskset_list" >&2
echo "PEWSTOP"  >&2

if [ "$#" -gt 0 ]; then
    exec "$@"
fi
