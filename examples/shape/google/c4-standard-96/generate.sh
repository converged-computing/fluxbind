#!/bin/bash
#
# This script generates a comprehensive suite of YAML shape files
# tailored for a 48-core, 2-NUMA node system (like a c4-standard-96).
#

echo "Generating shape files for a 48-core, 2-NUMA node system..."

# 1. Baseline: Packed Cores
cat > shape_packed_cores-shapefile.yaml <<EOF
# Binds rank 0->core:0, rank 1->core:1, etc., up to rank 47->core:47.
# This will fill the 24 cores on NUMA 0 first, then the 24 cores on NUMA 1.
default:
  type: core
  pattern: packed
EOF

# 2. SMT-Aware: Interleaved PUs
cat > shape_interleave_pus-shapefile.yaml <<EOF
# SMT-aware binding. Fills the first PU of all 48 cores, then the second PU.
#   rank 0 -> core:0.pu:0
#   rank 1 -> core:1.pu:0
#   ...
#   rank 48 -> core:0.pu:1
# This is hardware-adaptive and a key performance pattern.
default:
  type: pu
  pattern: interleave
EOF

# 3. NUMA-Aware: Interleave across NUMA nodes (Corrected for 2 NUMA nodes)
cat > shape_numa_interleave-shapefile.yaml <<EOF
# Distributes ranks in a "card dealer" fashion across the 2 NUMA nodes.
#   rank 0 -> numa:0
#   rank 1 -> numa:1
#   rank 2 -> numa:0
#   ...
default:
  type: numa
  formula: "\$((\$local_rank % 2))"
EOF

# 4. NUMA-Aware: Packed onto NUMA nodes (Corrected for 24 cores/NUMA)
cat > shape_numa_packed-shapefile.yaml <<EOF
# Binds the first 24 ranks entirely within numa:0, then the next 24 within numa:1.
# This uses integer division based on the 24 cores per NUMA node.
default:
  type: numa
  formula: "\$((\$local_rank / 24))"
EOF

# 5. Advanced: Reverse Packed Cores
cat > shape_reverse_packed_cores-shapefile.yaml <<EOF
# Binds rank 0->core:47, rank 1->core:46, etc.
default:
  type: core
  pattern: packed
  reverse: true
EOF

# 6. Advanced: Special Unbound Rank
cat > shape_special_rank_unbound-shapefile.yaml <<EOF
# A list of rules. The first match wins.

# Rule for the master rank (rank 0).
- ranks: "0"
  type: unbound

# Default rule for all other ranks (1-47).
# Bind them to cores, starting from core 1 to leave core 0 free.
- default:
    type: core
    formula: "\$local_rank"
EOF

# 7. Advanced: Spread Cores (for memory bandwidth tests)
cat > shape_spread_cores-shapefile.yaml <<EOF
# Binds ranks to every other core: rank 0->core:0, rank 1->core:2, etc.
# This tests for memory bandwidth sensitivity.
default:
  type: core
  formula: "\$((\$local_rank * 2))"
EOF

# 8. Advanced: Reverse Interleaved PUs
cat > shape_reverse_interleave_pus-shapefile.yaml <<EOF
# Creates a reverse SMT-aware binding.
# Fills the first PU of each core, starting from the LAST core (47) backwards.
#   rank 0 -> core:47.pu:0
#   rank 1 -> core:46.pu:0
#   ...
#   rank 48 -> core:47.pu:1
default:
  type: pu
  pattern: interleave
  reverse: true
EOF

# 9. Advanced: Core Interleaving Across NUMA Nodes ("Zip" Pattern) (Corrected for 2 NUMA)
cat > shape_numa_core_interleave-shapefile.yaml <<EOF
# Interleaves ranks across cores on the 2 different NUMA nodes ("zip" pattern).
#   rank 0 -> numa:0.core:0
#   rank 1 -> numa:1.core:0  (hwloc maps this to physical core 24)
#   rank 2 -> numa:0.core:1
#   rank 3 -> numa:1.core:1  (hwloc maps this to physical core 25)
default:
  # NUMA index = rank % 2
  # Core-on-NUMA index = rank / 2 (integer division)
  type: "numa:\$((\$local_rank % 2)).core:\$((\$local_rank / 2))"
  
  # The formula is embedded in the type string.
  formula: "" 
EOF


echo "Done. The following 9 shape files have been created:"
ls --color=never *-shapefile.yaml
