# fluxbind

> Intelligent detection and mapping of processors for HPC

[![PyPI version](https://badge.fury.io/py/fractale.svg)](https://badge.fury.io/py/fluxbind)

![img/fluxbind.png](img/fluxbind-small.png)

## How does this work?

OK I think I know what we might do. The top level description for a job resources is the jobspec "job specification" that might look like this:

```yaml
resources:
- type: slot
  count: 4
  with:
  - type: core
    count: 8
```

Flux run / submit is going to use flux-sched (or a scheduler) to assign jobs to nodes and to go into the flux exec shell and execute some number of tasks per node. Each of those tasks is what is going to hit and then execute our bash script, with a view of the entire node, and with need to run fluxbind shape to derive the binding for the task. We can technically derive a shapefile from a jobspec. It is the same, but only needs to describe the shape of one slot, for which the task that receives it is responsible for some N. So a shapefile that describes the shape of a slot looks like this:

```yaml
resources:
- type: core
  count: 8
```

And then that is to say "On this node, we are breaking resources down into this slot shape." We calculate the number of slots that the task is handling based on `FLUX_` environment variables. For now this is assume exclusive resources per node so if we are slightly off its not a huge deal, but in the future (given a slice of a node for a slot) we will need to be right, because we might see an entire node with hwloc but already be in a cpuset. Right now I'm also assuming the `fluxbind run` matches the topology of the shapefile. If you do something that doesn't match it probably can't be satisfied and will get an error, but not guaranteed.

## Run

Use fluxbind to run a job binding to specific cores. For flux, this means we require exclusive, and then for each node customize the binding exactly as we want it. We do this via a shape file.


### Basic Examples

```bash
# Start with a first match policy (I think this just works one node)
flux start --config ./examples/config/match-first.toml

# This works >1 node
flux alloc --conf ./examples/config/match-first.toml

# 1. Bind each task to a unique physical core, starting from core:0 (common case)
fluxbind run -n 8 --quiet --shape ./examples/shape-graph/single-node/simple_cores/shape.yaml --graph sleep 1

# 2. Reverse it!
fluxbind run -n 8 --quiet --shape ./examples/shape/1node/packed-cores-reversed-shapefile.yaml sleep 1

# 3. Packed PUs (hyperthreading), so interleaved.
fluxbind run --tasks-per-core 2 --quiet --shape ./examples/shape/1node/interleaved-shapefile.yaml sleep 1

# 4. Reverse it again!
fluxbind run --tasks-per-core 2 --quiet --shape ./examples/shape/1node/interleaved-reversed-shapefile.yaml sleep 1

# 5. An unbound rank - this tests "unbound" to leave Rank 0 unbound, pack all other ranks onto cores, shifted by one.
fluxbind run -N1 -n 3 --shape ./examples/shape/1node/unbound_rank.yaml sleep 1

# 6. L2 cache affinity. Give each task its own dedicated L2 cache to maximize cache performance.
# On mymachine, each core has its own private L2 cache.
# Therefore, binding one task per L2 cache is equivalent to binding one task per core.
fluxbind run -N1 -n 8 --quiet --shape ./examples/shape/1node/cache-affinity.yaml sleep 1

# 7. Reverse it
fluxbind run -N1 -n 8 --quiet --shape ./examples/shape/1node/cache-reversed-affinity.yaml sleep 1
```

### Kripke Examples

As we prepare to test with apps, here are some tests I'm thinking of doing.

```bash
# 1. Baseline - pack each MPI rank onto its own dedicated physical core (8.693519e-09)
fluxbind run -N 1 -n 8 --shape ./examples/shape/kripke/baseline-shapefile.yaml kripke --procs 2,2,2 --zones 16,16,16 --niter 500

# 2. Spread cores (memory bandwidth)
# If Kripke is limited by memory bandwidth, if we place ranks on every other core, we reduce contention for the shared L3 cache
# If Kripke memory bound, this layout might be faster than packed even with half cores. If compute based, worse (1.341355e-08)
fluxbind run -N 1 -n 4 --shape ./examples/shape/kripke/memory-spread-cores-shapefile.yaml kripke --procs 2,2,1 --zones 16,16,16 --niter 500

# 3. Packed pus (each of 8 cores has 2 pu == 16). We are testing if Kripke can benefit from SMT (simultaneous multi-threading)
fluxbind run -N 1 --tasks-per-core 2 --shape ./examples/shape/kripke/packed-pus-shapefile.yaml kripke --procs 2,4,2 --zones 16,16,16 --niter 500

# 4. Hybrid model: launch just two MPI ranks and give each one a whole L3 cache domain to work with (1.966967e-08)
fluxbind run -N 1 -n 2 --env OMP_NUM_THREADS=4 --env OMP_PLACES=cores --shape ./examples/shape/kripke/hybrid-l3-shapefile.yaml kripke --zones 16,16,16 --niter 500 --procs 2,1,1 --layout GZD
```

## Shape

The run command generates a shape, and we can test the shape command without it to provide a shapefile (basically, a modified jobspec with a pattern and optional affinity). Currently, the basic shape works as expected, but we are trying to work on the `--graph` implementation (uses a Flux jobspec and builds into a graph of nodes).

```bash
fluxbind shape --file examples/shape-graph/basic/4-cores-anywhere/shape.yaml --rank 0 --node-id 1 --local-rank 0 --gpus-per-task 0 --graph --global-ranks $(nproc) --nodes 1
```


## Predict

Use fluxbind to predict binding based on a job shape. This is prediction only, meaning there is no execution of an application or similar.
Here are some examples.

```bash
# Predict binding on this machine for 8 cores
fluxbind predict core:0-7

# Predict binding on corona (based on xml) for 2 NUMA nodes
fluxbind predict --xml ./examples/topology/corona.xml numa:0,1 x core:0-2
```


## License

DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
