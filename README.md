# fluxbind

> Intelligent detection and mapping of processors for HPC

[![PyPI version](https://badge.fury.io/py/fractale.svg)](https://badge.fury.io/py/fluxbind)

![img/fluxbind.png](img/fluxbind-small.png)

## Run

Use fluxbind to run a job binding to specific cores. For flux, this means we require exclusive, and then for each node customize the binding exactly as we want it. We do this via a shape file.


### Basic Examples

```bash
# Start with a first match policy
flux start --config ./examples/config/match-first.toml

# 1. Bind each task to a unique physical core, starting from core:0 (common case)
fluxbind run -n 8 --quiet --shape ./examples/shape/1node/packed-cores-shapefile.yaml sleep 1

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
