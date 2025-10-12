predict_help = """Predict hwloc bindings from a high-level expression of locations and operators.
Operators:
  +     UNION (OR) of two cpusets
  x     INTERSECTION (AND) of two cpusets
  ^     XOR (symmetric difference) of two cpusets
  ~     DIFFERENCE (cpus in first set but not second)

Example: fluxbind predict numa:0 x core:0-7
"""
