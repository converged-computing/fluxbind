# fluxbind

## Binding

How does fluxbind handle the cpuset calculation? I realize that when we bind to PUs (processing units) we are doing something akin to SMT. Choosing to bind to `Core` is without that. The objects obove those are containers - we don't really bind to them, we select them to then bind to child PU or Core (as I understand it). Since we are controlling the binding in the library, we need to think about both how the user specifies this, and defaults if they do not. We will implement a hierarchy of rules (checks) that the library does to determine what to do.

### Highest Priority: Explicit Request

The Shapefile needs an explicit request from the user - "This is my shape, but bind to PU/Core."
For this request, the shape.yaml file can have an options block with `bind`.

```yaml
# Avoid SMT and bind to physical cores.
options:
  bind: core

resources:
  - type: l3cache
    count: 1
```

In the above, the `options.bind` key exists so we honor it no matter what. This selection has to be Core or PU (someone tell me if I'm off about this - I'm pretty sure the cpusets in the hwloc on the containers are going to select the lower levels).


### Second level: Implicit Intent

This comes from the resource request. If a user has a lowest level, we can assume that is what they want to bind to. This would say "Bind to Core"

```yaml
resources:
- type: socket
  count: 1
  with:
  - type: core
    count: 4
```

This would say bind to PU (and the user is required to know the count)

```yaml
resources:
- type: socket
  count: 1
  with:
  - type: core
    count: 4
    with:
    - type: process
      count: 4
```

If they don't know the count, they can use the first strategy and request it explicitly:

```yaml
options:
  bind: process

resources:
  - type: l3cache
    count: 1
```

And note that I'm mapping "process" to "pu" because I don't think people (users) are familiar with pu. Probably I should support both.
In other words, if there is no `options.bind` we will inspect the `resources` and see if the final level (most granular) is Core or PU. If yes, we assume that is what we bind to.


### Lowest Priority: HPC Default

If we don't have an explicit request for binding and the lowest level is not PU or CPU, we have to assume some default. E.g., "Start with this container and bind to `<abstraction>` under it. Since most HPC workloads are run single threaded, I think we should assume Core. People that want SMT need to specify something special. Here is an example where we cannot know:

```yaml
resources:
- type: l3cache
  count: 1
```

We will allocate one `L3Cache` object, and when it's time to bind, we won't bind a bind directive or a PU/Core at the lowest level. We have to assume the default, which will be Core.

### Special Cases

#### Unbound

A special case is unbound. I didn't add this at first because I figured if the user didn't want binding, they wouldn't use the tool. But the exception is devices! I might want to be close to a GPU or NIC but not actually bind any processes. In that case I would use fluxbind and specify the shape, but I'd ask for unbound:


```yaml
options:
  bind: none

resources:
  - type: core
    count: 4
    affinity:
      type: gpu
      count: 1
```

Note that the affinity spec above is still a WIP. I have something implemented for my first approach but am still working on this new graph one. The above is subject to change, but illustrates the point - we don't want to bind processes, but we want the cores to have affinity (be close to) a gpu.

#### GPU

This might be an alternative to the above - I'm not decided yet. GPU affinity (remote or local) means we want a GPU that is close by (same NUMA node) or remote (different NUMA), I haven't tested this yet, but it will look like this:

```yaml
options:
  bind: gpu-local

resources:
  - type: core
    count: 4
```

Right now I have this request akin to `bind` (as a bind type I mean) because then the pattern defaults to `packed`. I think that is OK. I like this maybe a little better than the one before because we don't change the jobspec too much... :)


### Examples

Here are examples for different scenarios.

| `shape.yaml` | Logic Used | Final Binding Unit |
| :--- | :--- | :--- |
| **`options: {bind: process}`**, `resources: [{type: socket}]` | Explicit Request | `pu` |
| **`options: {bind: core}`**, `resources: [{type: socket}]` | Explicit Request | `core` |
| No options, `resources: [{type: core, count: 4}]` | Implicit Intent | `core` |
| No options, `resources: [{type: pu, count: 4}]` | Implicit Intent | `pu` |
| No options, `resources: [{type: l3cache, count: 1}]` | HPC Default | `core` |
| No options, `resources: [{type: numanode, count: 1}]` | HPC Default | `core` |
| `options: {bind: process}`, `resources: [{type: core, count: 2}]` | Explicit Request | `pu` |


## Patterns

The binding rules determine *what* kind of hardware to bind to (physical cores vs. hardware threads) and patterns determine *how* a total pool of those resources is distributed among the tasks on a node. When a `shape.yaml` describes a total pool of resources (e.g., `core: 8`) and a job is launched with multiple tasks on the node (e.g., `local_size=4`), `fluxbind` must have a deterministic strategy to give each task its own unique slice of the total pool. This strategy is controlled by the `pattern` key.

### packed

> Default

The packed pattern assigns resources in contiguous, dense blocks. This is the default behavior if no pattern is specified, because I think it is what generally would be wanted, because cores are physically close. As an example, given 8 available cores and 4 tasks, packed assigns resources like this:
  * `local_rank=0` gets `[Core:0, Core:1]`
  * `local_rank=1` gets `[Core:2, Core:3]`
  * `local_rank=2` gets `[Core:4, Core:5]`
  * `local_rank=3` gets `[Core:6, Core:7]`

```yaml
# pattern: packed is optional as it's the default, so you could leave this out.
resources:
  - type: core
    count: 8
    pattern: packed
```

## scatter (spread)

> The pattern that makes you think of peanut butter

The scatter pattern distributes resources with the largest possible stride, like dealing out cards to each task. I think this can be similar to [cyclic](https://hpc.llnl.gov/sites/default/files/distributions_0.gif) or round robin.  I think we'd want to do this for memory intensive tasks, where we would want cores physically far apart so each gets its own memory (L2/L3 caches).

```yaml
# 'spread' is an alias for 'scatter'.
resources:
  - type: core
    count: 8
    pattern: spread
```

Right now I'm calling this interleaved, but I think they are actually different and if we want this case we need to add it. Interleaved would be like filling up all cores first (one PU) before going back and filling other PUs. Like filling cookies with Jam, but only every other cookie.

## Modifiers

### reverse

The reverse modifier is a boolean (true/false) that can be combined with any pattern. It simply reverses the canonical list of available resources before the distribution pattern is applied. Not sure when it's useful, but maybe we'd want to test one end and then "the other end."

```yaml
resources:
  - type: core
    count: 8
    reverse: true
```
