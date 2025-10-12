#!/bin/bash
#
# run_mapping.sh

if [ "$#" -eq 0 ]; then
    echo "Error: No application command provided to run_mapping.sh." >&2
    exit 1
fi

# Get Environment Info (Using your preferred logic) ---
rank=${FLUX_TASK_RANK:-0}
local_rank=${FLUX_TASK_LOCAL_ID:-0}
node=$(hostname)

# Get the logical node id using the arithmetic approach
node_id=$(flux job taskmap --nodeid=${rank} ${FLUX_JOB_ID})

# The user provides the path to the shape file in the environment.
if [ -z "$JOB_SHAPE_FILE" ]; then
    echo "Error: JOB_SHAPE_FILE is not set." >&2
    exit 1
fi

# --- 3. Calculate Binding Location ---
# Call the fluxbind helper script to get the target location string (e.g., "core:0" or "UNBOUND")
BIND_LOCATION=$(fluxbind shape --file "$JOB_SHAPE_FILE" --rank "$rank" --node-id "$node_id" --local-rank "$local_rank")

# Exit if the helper script failed
if [ $? -ne 0 ]; then
    echo "Error: The 'fluxbind shape' helper script failed for rank ${rank}." >&2
    exit 1
fi

if [[ "${BIND_LOCATION}" == "UNBOUND" ]]; then
    # For an unbound task, the "effective" binding is the entire machine.
    binding_source="UNBOUND"
    cpuset_mask=$(hwloc-calc machine:0)
    logical_cpu_list=$(hwloc-calc "$cpuset_mask" --intersect PU 2>/dev/null)
    physical_core_list=$(hwloc-calc "$cpuset_mask" --intersect core 2>/dev/null)
else
    # For a bound task, calculate the mask and lists from the target location string.
    binding_source=${BIND_LOCATION}
    cpuset_mask=$(hwloc-calc "${BIND_LOCATION}")
    logical_cpu_list=$(hwloc-calc "${BIND_LOCATION}" --intersect PU 2>/dev/null)
    physical_core_list=$(hwloc-calc "${BIND_LOCATION}" --intersect core 2>/dev/null)
fi

if [[ "$FLUXBIND_NOCOLOR" != "1" ]]
  then
  YELLOW='\033[1;33m'
  GREEN='\033[0;32m'
  RESET='\033[0m'
  BLUE='\e[0;34m'
  CYAN='\e[0;36m'
  MAGENTA='\e[0;35m'
  ORANGE='\033[0;33m'
else
  YELLOW=""
  GREEN=""
  RESET=""
  BLUE=""
  CYAN=""
  MAGENTA=""
  ORANGE=""
fi

if [[ "$FLUXBIND_QUIET" != "1" ]]
  then
  prefix="${YELLOW}rank ${rank}${RESET}"
  echo -e "${prefix}: Binding Source:         ${MAGENTA}$binding_source${RESET}"
  echo -e "${prefix}: PID for this rank:      ${GREEN}$$ ${RESET}"
  echo -e "${prefix}: Effective Cpuset Mask:  ${CYAN}$cpuset_mask${RESET}"
  echo -e "${prefix}: Logical CPUs (PUs):     ${BLUE}${logical_cpu_list:-none}${RESET}"
  echo -e "${prefix}: Physical Cores:         ${ORANGE}${physical_core_list:-none}${RESET}"
  echo
fi

# The 'exec' command replaces this script's process, preserving the env.
# I learned this developing singularity shell, exec, etc :)

if [[ "${BIND_LOCATION}" == "UNBOUND" ]]; then
    # Execute the command directly without changing affinity.
    if [[ "$FLUXBIND_SILENT" != "1" ]]; then echo -e "${GREEN}fluxbind${RESET}: Rank ${rank} is ${BIND_LOCATION} to execute: $@" >&2; fi
    exec "$@"
else
    # Use hwloc-bind to set the affinity and then execute the command.
    if [[ "$FLUXBIND_SILENT" != "1" ]]; then  echo -e "${GREEN}fluxbind${RESET}: Rank ${rank} is bound to ${BIND_LOCATION} to execute: $@" >&2; fi
    exec hwloc-bind "${BIND_LOCATION}" -- "$@"
fi
