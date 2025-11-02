import io
import logging
from itertools import combinations

import networkx as nx

import fluxbind.graph.worker as worker
import fluxbind.shape.commands as commands
import fluxbind.utils as utils

log = logging.getLogger(__name__)


class HwlocTopology:
    """
    An HwlocTopology maps xml from lstopo into a graph.
    """

    def __init__(self, xml_input=None, max_workers=None):
        self.graph = nx.DiGraph()
        self.gpus = []
        self.nics = []
        self.load(xml_input, max_workers)

    @property
    def node_count(self):
        return self.graph.number_of_nodes()

    def load(self, xml_input, max_workers=None):
        """
        Load the graph, including distances, and pre-calculate
        entire set of affinities for objects.
        """
        self.last_affinity_target = None

        # If we don't have an xml file, derive from system
        if not xml_input:
            xml_input = commands.lstopo.get_xml()
        root = utils.read_xml(xml_input)

        # I think this is required - I haven't seen one without it.
        top_level_object = root.find("object")
        if top_level_object is not None:
            self.build_graph(top_level_object)
        else:
            raise ValueError('Could not find a top-level "object" in the hwloc XML.')
        log.debug(f"Graph built successfully using XML tree with {self.node_count} nodes.")

        # Use nvidia/rocm-smi to find devices, since not always in hwloc
        self.discover_devices()

        # Get the distances or distances2 matrix
        self.add_distance_edges(root)
        self.hierarchy_view = nx.subgraph_view(
            self.graph, filter_edge=lambda u, v: self.graph.edges[u, v].get("type") == "contains"
        )

        # Cache affinities on startup so we only do it once.
        # This assumes the node cannot change (I assume it cannot)
        self.precompute_affinities(max_workers=max_workers)
        self.create_ordered_gpus()

    def build_graph(self, element, parent_gp=None):
        """
        Recursively builds the topology graph by parsing the XML tree structure directly.

        gp == global peristent index. pci == peripheral component interconnect.
        """
        # Process current element and add it as a node
        node_data = element.attrib.copy()

        # This is a peristent identifier that includes location, bus, and device function
        if "pci_busid" in node_data:
            node_data["pci_busid"] = node_data["pci_busid"].lower()
        for info in element.findall("info"):
            node_data[info.get("name")] = info.get("value")

        for key in ["os_index", "gp_index", "depth", "cache_size", "local_memory"]:
            if key in node_data:
                try:
                    node_data[key] = int(node_data[key])
                except (ValueError, TypeError):
                    pass

        # Use the element's memory address as a guaranteed unique graph pointer (gp)
        current_gp = id(element)
        node_data["gp_index"] = current_gp

        # Also parse cpuset if it exists, as it's useful data, but don't use it for hierarchy.
        if "cpuset" in node_data:
            try:
                full_hex = "0x" + node_data["cpuset"].replace("0x", "").replace(",", "")
                node_data["cpuset_val"] = int(full_hex, 16)
            except ValueError:
                node_data["cpuset_val"] = None

        # Add the node to the graph
        self.graph.add_node(current_gp, **node_data)

        # If there's a parent, add a 'contains' edge to represent the hierarchy
        if parent_gp is not None:
            self.graph.add_edge(parent_gp, current_gp, type="contains")

        # Recurse for all direct child 'object' elements
        for child_element in element.findall("./object"):
            self.build_graph(child_element, parent_gp=current_gp)

    def discover_devices(self):
        """
        Discover different kinds of GPU and NIC.

        TODO: a better design would be to discover devices, then annotate the graph
        nodes with device type. Then when we search for gpu or nic, we can just
        search by that attribute. Right now we search for PCIDev, and then filter.
        """
        # Don't assume only one vendor of GPU.
        for vendor, command in {"NVIDIA": commands.nvidia_smi, "AMD": commands.rocm_smi}.items():
            try:
                pci_ids = command.get_pci_bus_ids()
                for bus_id in pci_ids:
                    self.gpus.append({"pci_bus_id": bus_id.lower(), "vendor": vendor})
            except RuntimeError:
                pass

        # There is probably a better way to do this.
        log.debug("Discovering Network Interface Cards (NICs)...")
        nic_keywords = ["ethernet", "infiniband", "connectx", "mellanox", "network connection"]
        for gp, data in self.find_objects(type="PCIDev"):
            if (os_dev := data.get("OSDev")) and any(
                # mlx4_0 (InfiniBand HCA)
                # hfi1_0 (Omni-Path interface)
                # bxi0 (Atos/Bull BXI HCA
                os_dev.startswith(p)
                for p in ["eth", "en", "ib", "mlx", "hfi", "bxi"]
            ):
                self.nics.append((gp, data))
                continue
            if (pci_type := data.get("pci_type")) and pci_type.split(" ")[0].startswith("02"):
                self.nics.append((gp, data))
                continue
            if (pci_name := data.get("PCIDevice")) and any(
                k in pci_name.lower() for k in nic_keywords
            ):
                self.nics.append((gp, data))
                continue

        if self.nics:
            unique_nics = {data["pci_busid"]: (gp, data) for gp, data in self.nics}
            self.nics = list(unique_nics.values())
            log.info(f"Successfully discovered {len(self.nics)} NIC(s).")
        else:
            log.warning("No NICs were discovered.")

    def add_distance_edges(self, root):
        """
        Hwloc can have a distances or distances2 section that has a NUMA
        node distance matrix. distances2 is for hwloc 2.x and distances for
        hwloc 1.x
        """

        def add_latency_edges(matrix, indexes):
            """
            Helper funcion to convert Python matrix to add edges to graph
            """
            # Create a fast lookup table to convert from a NUMA node's OS index
            # (e.g., 0, 1, 2) to its unique graph pointer (a large integer).
            os_to_gp = {
                d["os_index"]: gp for gp, d in self.find_objects(type="NUMANode") if "os_index" in d
            }

            # Iterate through every cell (i, j) in the distance matrix.
            for i, from_os in enumerate(indexes):
                for j, to_os in enumerate(indexes):
                    # Find the graph pointers for the 'from' and 'to' NUMA nodes.
                    from_gp, to_gp = os_to_gp.get(from_os), os_to_gp.get(to_os)

                    # If both nodes were found in our graph...
                    if from_gp and to_gp:
                        # ...add a special directed edge between them.
                        self.graph.add_edge(
                            from_gp,
                            to_gp,
                            type="latency_link",  # Mark this as a special non-hierarchical edge.
                            weight=matrix[i][j],  # The weight is the latency value from the matrix.
                        )

        # First, try to parse newer hwloc v2.x format
        for dist_el in root.iter("distances2"):
            try:
                # nbobjs is how many objects are in the matrix (e.g., 2 for 2 NUMA nodes).
                num = int(dist_el.get("nbobjs"))
                # <indexes> is the OS indexes of the NUMA nodes (e.g., "0 1").
                idxs = [int(v) for v in dist_el.find("indexes").text.strip().split()]
                # <u64values> is latency values as a flat list (e.g., "10 21 21 10").
                vals = [float(v) for v in dist_el.find("u64values").text.strip().split()]

                # Sanity check: a 2x2 matrix should have 4 values.
                if len(vals) == num * num and len(idxs) == num:
                    # Convert the flat list vals into a 2D list-of-lists (our matrix).
                    matrix = [vals[i * num : (i + 1) * num] for i in range(num)]
                    add_latency_edges(matrix, idxs)
                    log.debug("Parsed hwloc v2 distances.")
                    return  # Success! We are done, so exit the function.

            except (ValueError, TypeError, AttributeError):
                # If anything goes wrong (e.g., missing tags, bad text), just skip it.
                continue

        # Fall back to the legacy hwloc v1.x format
        for dist_el in root.iter("distances"):
            try:
                num = int(dist_el.get("nbobjs"))
                # In the old format, the values are relative to a base latency.
                latency_base = float(dist_el.get("latency_base"))
                # The values are stored in a 'value' attribute as a space-separated string.
                # We multiply each value by the base to get the true latency.
                vals = [float(v) * latency_base for v in dist_el.get("value").strip().split()]

                # The indexes are found the same way as in v2.
                idxs = [int(v) for v in dist_el.find("indexes").text.strip().split()]

                if len(vals) == num * num and len(idxs) == num:
                    matrix = [vals[i * num : (i + 1) * num] for i in range(num)]
                    add_latency_edges(matrix, idxs)
                    log.debug("Parsed legacy hwloc v1 distances.")
                    return  # Success! We are done.
            except (ValueError, TypeError, AttributeError):
                continue

    def precompute_affinities(self, max_workers=None):
        """
        Precompute numa affinities
        """
        objects_to_locate = (
            self.find_objects(type="PU") + self.find_objects(type="Core") + self.nics
        )

        for gpu in self.gpus:
            if matches := self.find_objects(type="PCIDev", pci_busid=gpu["pci_bus_id"]):
                objects_to_locate.append(matches[0])

        w = worker.AffinityCalculator(max_workers)
        for result in w.calculate_numa_affinity(objects_to_locate):
            self.graph.nodes[result[0]]["numa_os_index"] = result[1]

    def match_resources(self, jobspec, allocated_gps=None):
        """
        Finds the single, next available allocation that satisfies the jobspec.
        """
        if allocated_gps is None:
            allocated_gps = set()

        # Pass the set of already-taken resources to the search.
        # The search function will use copies so the original set is not modified.

        job_requests = jobspec.get("resources") or jobspec.get("resource")
        if not job_requests:
            raise ValueError("Jobspec does not contain a 'resources' section.")

        final_allocation = []
        machine_gp, _ = self.find_objects(type="Machine")[0]

        # Use a temporary set to track allocations within this single search
        temp_allocations = allocated_gps.copy()

        for request in job_requests:
            assignment = self.find_assignment_recursive(
                request, machine_gp, temp_allocations, depth=1
            )

            if assignment is None:
                log.debug(f"Failed to find a match for request: {request}")
                return None  # Indicate that no valid slot could be found

            final_allocation.extend(assignment)
            temp_allocations.update({gp for gp, _ in assignment})

        log.debug(f"Successfully found a slot with {len(final_allocation)} objects.")
        return final_allocation

    def sort_by_affinity(self, candidates, affinity, allocated, domain_gp):
        """
        Sort list of candidates by affinity so we get closest one.
        """
        target_type = self.translate_type(affinity.get("type"))
        if not target_type:
            log.warning("Affinity spec missing 'type'.")
            return candidates
        
        # Search within the domain we were provided, not across the machine
        log.debug(f"    -> Searching for affinity target '{target_type}' within the current domain.")
        targets = self.get_available_children(domain_gp, target_type, allocated)        
        if not targets:
            log.warning(f"Affinity target '{target_type}' not found.")
            return candidates
        target_gp, target_data = targets[0]
        self.last_affinity_target = (target_gp, target_data)
        log.debug(
            f"    -> Distances to target {target_data.get('type')}:{target_data.get('PCIDevice') or target_data.get('os_index')}"
        )
        decorated = sorted(
            [(self.get_distance(c[0], target_gp), c) for c in candidates],
            key=lambda x: x[0],
        )
        return [item for _, item in decorated]

    def translate_type(self, requested_type: str):
        """
        Translates a user-friendly type string from the shapefile into the
        exact string used by the hwloc graph. This is the single source of
        truth for type name mapping.
        """
        # A dictionary to map all known aliases and lowercase variants
        # to the official hwloc type string.
        mapping = {
            "process": "PU",
            "pu": "PU",
            "core": "Core",
            "socket": "Package",
            "package": "Package",
            "numanode": "NUMANode",
            "l1cache": "L1Cache",
            "l2cache": "L2Cache",
            "l3cache": "L3Cache",
            "machine": "Machine",
            "nic": "PCIDev",
            "gpu": "PCIDev",
        }

        # capitalizing the word (e.g., 'l3cache' -> 'L3cache').
        return mapping.get(requested_type.lower(), requested_type.capitalize())

    def summarize(self, nodes):
        """
        Given a set of nodes in the graph (a set of resources) print a textual visual.
        """
        for gp, data in nodes:
            p_info = ""
            if data["type"] in ["Core", "PU"]:
                package = self.get_ancestor_of_type(gp, "Package")
                if package:
                    p_info = f"Package:{package[1].get('os_index')} -> "
                if data["type"] == "PU":
                    core = self.get_ancestor_of_type(gp, "Core")
                    if core:
                        p_info += f"Core:{core[1].get('os_index')} -> "
            log.info(f"  -> {p_info}{data['type']}:{data.get('os_index', data.get('pci_busid'))}")

    def calculate_bindable_nodes(self, total_allocation):
        """
        Given an allocation, get nodes in the graph we can bind to.
        """
        log.info(
            "No explicit CPU resources requested. Deriving a binding from the allocation context."
        )
        leaf_nodes = []

        # Find the NUMA domain(s) from the allocation context. This is correct.
        numa_gps = {node[0] for node in total_allocation if node[1].get("type") == "NUMANode"}
        if not numa_gps:
            for _, data in total_allocation:
                if (numa_idx := data.get("numa_os_index")) is not None:
                    if numa_matches := self.find_objects(type="NUMANode", os_index=numa_idx):
                        numa_gps.add(numa_matches[0][0])

        # This should not happen - throw up if it does.
        if not numa_gps:
            raise RuntimeError(
                "Allocation succeeded but could not determine a NUMA domain for CPU binding."
            )

        # Find the parent Package(s) of the NUMA domains.
        package_gps = set()
        for numa_gp in numa_gps:
            package = self.get_ancestor_of_type(numa_gp, "Package")
            if package:
                package_gps.add(package[0])

        if not package_gps:
            raise RuntimeError(f"Could not find parent Package for NUMA nodes: {list(numa_gps)}")

        package_indices = [self.graph.nodes[gp].get("os_index") for gp in package_gps]
        log.info(f"Binding to all PUs within parent Package(s): {package_indices}")

        # Find all PUs that are hierarchical children of those Package(s).
        for package_gp in package_gps:
            pus_in_package = self.get_descendants(package_gp, type="PU")
            leaf_nodes.extend(pus_in_package)
        return leaf_nodes

    def get_available_children(self, parent_gp, child_type, allocated):
        """
        Given a parent and child type, find the child type!
        """
        parent_node = self.graph.nodes[parent_gp]
        parent_info = f"{parent_node.get('type')}:{parent_node.get('os_index', parent_node.get('pci_busid', parent_gp))}"
        log.debug(f"    - get_children(child='{child_type}', parent={parent_info})")

        parent_type = parent_node.get("type")
        child_type_lower = child_type.lower()
        all_candidates = self.find_objects(type=child_type)
        log.debug(
            f"    -   -> Found {len(all_candidates)} total unique system-wide candidates for '{child_type}'."
        )

        available = []
        for gp, data in all_candidates:
            if gp in allocated:
                continue

            is_valid_child = False

            # Rule 1: Relationship to NUMA node is through shared PACKAGE parent (for Cores) or LOCALITY (for Devices)
            if parent_type == "NUMANode":
                if child_type_lower in ["core", "pu"]:
                    package_of_numa = self.get_ancestor_of_type(parent_gp, "Package")
                    if package_of_numa and nx.has_path(self.hierarchy_view, package_of_numa[0], gp):
                        is_valid_child = True
                elif child_type_lower == "pcidev":
                    if data.get("numa_os_index") == parent_node.get("os_index"):
                        is_valid_child = True
            
            # Rule 2 (NEW): Relationship of a Core/PU to a Device is through shared NUMA LOCALITY
            elif parent_type == "PCIDev" and child_type_lower in ["core", "pu"]:
                parent_numa_idx = parent_node.get("numa_os_index")
                if data.get("numa_os_index") == parent_numa_idx:
                    is_valid_child = True

            # Default Rule: Strict HIERARCHY for all other cases
            else:
                if nx.has_path(self.hierarchy_view, parent_gp, gp):
                    is_valid_child = True

            if is_valid_child:
                available.append((gp, data))

        available.sort(key=lambda item: self.get_sort_key_for_node(item))
        log.debug(f"    -   -> Returning {len(available)} available children.")
        return available

    def find_objects(self, **attributes):
        """
        Search nodes in the graph for a specific attribute (or more than one)
        """
        return [
            (gp, data)
            for gp, data in self.graph.nodes(data=True)
            if all(data.get(k) == v for k, v in attributes.items())
        ]

    def find_assignment_recursive(self, request, domain_gp, allocated, depth=0):
        """
        Given a "with" section at a particular depth, determine if the request is satsified.

        We call this function recursively until the graph "with" sections terminate.
        """
        indent = "  " * depth
        domain_node = self.graph.nodes[domain_gp]
        domain_info = f"{domain_node.get('type')}:{domain_node.get('os_index', domain_node.get('pci_busid', domain_gp))}"
        log.debug(f"{indent}[ENTER] find_assignment(req={request}, domain={domain_info})")

        # This can also be gpu/nic
        raw_request_type = request['type']
        req_type, count = self.translate_type(raw_request_type), request["count"]

        # If the req_type is gpu or nic, this isn't an actual type in the graph - it is PCIDev.
        candidates = self.get_available_children(domain_gp, req_type, allocated)

        # Now we handle the type of the pcidev request and filter candidates to those devices.
        if raw_request_type.lower() == 'gpu':
            gpu_bus_ids = {g['pci_bus_id'] for g in self.gpus}
            candidates = [node for node in candidates if node[1].get('pci_busid') in gpu_bus_ids]
            log.debug(f"{indent}  -> Filtered for 'gpu', {len(candidates)} candidates remain.")
        elif raw_request_type.lower() == 'nic':
            nic_bus_ids = {n[1]['pci_busid'] for n in self.nics}
            candidates = [node for node in candidates if node[1].get('pci_busid') in nic_bus_ids]
            log.debug(f"{indent}  -> Filtered for 'nic', {len(candidates)} candidates remain.")

        log.debug(f"{indent}  -> Found {len(candidates)} initial candidates for '{req_type}'.")

        affinity_spec = request.get("affinity")
        if affinity_spec:
            affinity_type_from_yaml = affinity_spec.get("type", "").lower()
            domain_type_from_hwloc = domain_node.get("type", "").lower()

            # A local affinity search is needed if:
            # 1. The affinity type and domain type match exactly (e.g., 'numanode').
            # 2. Or, if the affinity is for a 'gpu' or 'nic' and the domain is a 'pcidev'.
            is_local_affinity_target = (affinity_type_from_yaml == domain_type_from_hwloc) or (
                affinity_type_from_yaml in ["gpu", "nic"] and domain_type_from_hwloc == "pcidev"
            )

            if is_local_affinity_target:
                log.debug(f"{indent}  -> Applying LOCAL affinity to parent domain {domain_info}")
                target_gp = domain_gp
                decorated = sorted(
                    [(self.get_distance(c[0], target_gp), c) for c in candidates],
                    key=lambda x: x[0],
                )
                candidates = [item for _, item in decorated]
                self.last_affinity_target = (target_gp, domain_node)
            else:
                log.debug(f"{indent}  -> Sorting candidates by GLOBAL affinity to {affinity_spec}")
                candidates = self.sort_by_affinity(candidates, affinity_spec, allocated, domain_gp)

        if len(candidates) < count:
            log.debug(
                f"{indent}[FAIL] Not enough candidates available. Found {len(candidates)}, need {count}."
            )
            return None

        for i, combo in enumerate(combinations(candidates, count)):
            combo_info = ", ".join(
                [f"{d.get('type')}:{d.get('os_index', d.get('pci_busid'))}" for _, d in combo]
            )
            log.debug(f"{indent}  -> Trying Combination #{i+1}: ({combo_info})")

            path_allocations = allocated.copy()
            full_solution_for_combo = list(combo)
            for gp, _ in combo:
                path_allocations.add(gp)

            all_children_found = True
            if "with" in request:
                for parent_gp, _ in combo:
                    for child_req in request["with"]:
                        child_assignment = self.find_assignment_recursive(
                            child_req, parent_gp, path_allocations, depth + 1
                        )
                        if child_assignment is None:
                            all_children_found = False
                            break
                    if not all_children_found:
                        break

            if all_children_found:
                log.debug(f"{indent}[SUCCESS] Combination #{i+1} satisfied all constraints.")
                if "with" in request:
                    temp_alloc = allocated.copy()
                    for gp, _ in combo:
                        temp_alloc.add(gp)
                    for parent_gp, _ in combo:
                        for child_req in request["with"]:
                            child_assignment = self.find_assignment_recursive(
                                child_req, parent_gp, temp_alloc, depth + 1
                            )
                            if child_assignment:  # Ensure we don't extend with None
                                full_solution_for_combo.extend(child_assignment)
                                for c_gp, _ in child_assignment:
                                    temp_alloc.add(c_gp)
                log.debug(f"{indent}[EXIT] find_assignment -> returning solution")
                return full_solution_for_combo

        log.debug(f"{indent}[FAIL] Exhausted all {i+1 if 'i' in locals() else 0} combinations.")
        return None

    def get_descendants(self, gp_index, **filters):
        """
        Given a global position index, return descendents that match a filter.
        """
        if gp_index not in self.graph:
            return []
        desc = list(nx.descendants(self.hierarchy_view, gp_index))
        return [
            (di, self.graph.nodes[di])
            for di in desc
            if all(self.graph.nodes[di].get(k) == v for k, v in filters.items())
        ]

    def get_ancestor_of_type(self, start_node_gp, ancestor_type):
        """
        Given a starting node, return all ancestors of a certain type
        """
        current_gp = start_node_gp

        # Walk up the hierarchy tree one parent at a time.
        while current_gp in self.hierarchy_view:
            # Get the parent (should only be one in a tree)
            parents = list(self.hierarchy_view.predecessors(current_gp))
            if not parents:
                break
            parent_gp = parents[0]
            parent_data = self.graph.nodes[parent_gp]
            if parent_data.get("type") == ancestor_type:
                return (parent_gp, parent_data)
            current_gp = parent_gp
        return None

    def get_sort_key_for_node(self, leaf_node):
        """
        Return tuple sorting key e.g., (0, package_id, core_id) -> e.g., (0, 0, 5)
        """
        gp, data = leaf_node

        # TYPE_ORDER: CPU types < PCI types < Other OS types < Nameless types
        # This ensures consistent sorting across different object types.

        # Handle CPU-like objects (Cores, PUs)
        if data.get("type") in ["Core", "PU"]:
            package = self.get_ancestor_of_type(gp, "Package")
            package_idx = package[1].get("os_index", -1) if package else -1
            return (0, int(package_idx), int(data.get("os_index", -1)))

        # Handle PCI devices (GPUs, NICs)
        elif "pci_busid" in data:
            try:
                # Convert '0000:c4:00.0' into a sortable tuple of integers
                pci_tuple = tuple(
                    int(p, 16) for p in data["pci_busid"].replace(":", ".").split(".")
                )
                # Returns (1, (0, 196, 0, 0))
                return (1, pci_tuple)
            except (ValueError, TypeError):
                # Fallback for weirdly formatted pci_busid
                return (1, data["pci_busid"])

        # Handle other objects with an os_index (like NUMANodes, if they were leaves)
        elif "os_index" in data:
            # Returns (2, os_index) -> e.g., (2, 1)
            return (2, int(data.get("os_index", -1)))

        # Fallback for any other object type
        else:
            # Returns (3, gp_index) as a last resort for stable sorting
            return (3, gp)

    def get_distance(self, node1_gp, node2_gp):
        node1, node2 = self.graph.nodes.get(node1_gp), self.graph.nodes.get(node2_gp)
        if not node1 or not node2:
            return float("inf")
        numa1, numa2 = node1.get("numa_os_index"), node2.get("numa_os_index")
        try:
            if numa1 is None or numa2 is None or numa1 == numa2:
                lca = nx.lowest_common_ancestor(self.hierarchy_view, node1_gp, node2_gp)
                return nx.shortest_path_length(
                    self.hierarchy_view, lca, node1_gp
                ) + nx.shortest_path_length(self.hierarchy_view, lca, node2_gp)
            else:
                numa1_gp = self.find_objects(type="NUMANode", os_index=numa1)[0][0]
                numa2_gp = self.find_objects(type="NUMANode", os_index=numa2)[0][0]
                return (
                    self.get_distance(node1_gp, numa1_gp)
                    + self.graph.edges[numa1_gp, numa2_gp].get("weight", float("inf"))
                    + self.get_distance(node2_gp, numa2_gp)
                )
        except (nx.NetworkXError, KeyError, IndexError):
            return float("inf")

    def create_ordered_gpus(self):
        """
        Creates a deterministically sorted list of all discovered GPUs,
        enriched with their NUMA locality. We use this list for
        assigning GPUs to ranks consistently.

        This method must be called after discover_devices and
        precompute_numa_affinities are done.
        """
        ordered_gpus = []

        # The gpus we found were discovered with nvidia/rocm-smi and we need
        # to map to things in the graph.
        for gpu_info in self.gpus:
            pci_id = gpu_info.get("pci_bus_id")
            if not pci_id:
                continue

            # Find the corresponding PCIDev object in our graph
            # Note: We now store and search for types in lowercase
            matches = self.find_objects(type="PCIDev", pci_busid=pci_id)
            if not matches:
                log.warning(
                    f"Could not find a graph object for discovered GPU with PCI ID: {pci_id}"
                )
                continue

            _, gpu_data = matches[0]

            # Retrieve the NUMA index that was calculated in precompute_affinities
            numa_index = gpu_data.get("numa_os_index")
            if numa_index is None:
                log.warning(
                    f"Could not determine NUMA locality for GPU {pci_id}. It will be excluded from assignments."
                )
                continue

            # The GPUAssignment class expects 'pci_id' and 'numa_index' keys.
            ordered_gpus.append({"pci_id": pci_id, "numa_index": numa_index})

        # Sort the list deterministically. This is the crucial step.
        # We sort by NUMA index first, and then by PCI bus ID as a tie-breaker.
        ordered_gpus.sort(key=lambda gpu: (gpu["numa_index"], gpu["pci_id"]))

        # Finally, create the attribute that the rest of the code expects.
        self.ordered_gpus = ordered_gpus
        log.info(f"Created an ordered list of {len(self.ordered_gpus)} GPUs for assignment.")


    def find_bindable_leaves(self, total_allocation, bind_level):
        """
        Transforms a list of allocated resources into a final list of bindable
        nodes by first choosing a strategy based on the allocation contents,
        then executing that single, correct strategy.
        """
        leaf_nodes = []
        log.debug(f"Transforming {len(total_allocation)} allocated objects to bind_level '{bind_level}'...")
        
        bind_type_concrete = self.translate_type(bind_level)

        # Check for high-level structural containers. Their presence dictates the entire strategy.
        high_level_containers = [node for node in total_allocation if node[1]['type'] in ['Package', 'NUMANode']]
        
        if high_level_containers:
            # If we find a Package or NUMANode, we IGNORE all other items in the allocation
            # and bind to the contents of this container ONLY. We have to do this because
            # hwloc-calc can report that some CPU/PU are closer to the OTHER Numa node, or
            # in other words, the physical layout of the xml != what hwloc-calc reports.
            # So here we use get_ancestor_of_type to JUST use the hardware layout (which
            # is more predictable).
            container_gp, container_data = high_level_containers[0]
            container_type = container_data.get("type")
            log.debug(f"High-level container '{container_type}' found. Binding exclusively to its physical contents.")            
            package_gp = container_gp if container_type == "Package" else self.get_ancestor_of_type(container_gp, "Package")[0]
            if package_gp:
                leaf_nodes = self.get_descendants(package_gp, type=bind_type_concrete)
        else:
            # No high-level containers - we can safely process each object individually.
            # This is the logic that correctly handles the simple Core, PU, and device-affinity tests.
            log.debug("No high-level containers found. Processing each allocated object individually.")
            for gp, data in total_allocation:
                
                # Case 1: The object is already the type we want to bind to.
                if data.get('type') == bind_type_concrete:
                    leaf_nodes.append((gp, data))
                    continue

                # Case 2 (Container): Must be a low-level container (Core or PCIDev).
                descendants = self.get_descendants(gp, type=bind_type_concrete)
                if descendants:
                    leaf_nodes.extend(descendants)
                    continue

                # Case 2c (Child): The object is a child of the target type (e.g., PU -> Core).
                ancestor = self.get_ancestor_of_type(gp, bind_type_concrete)
                if ancestor:
                    leaf_nodes.append(ancestor)

        # De-duplicate the final list and sort for deterministic assignment.
        unique_nodes = list({gp: (gp, data) for gp, data in leaf_nodes}.values())
        unique_nodes.sort(key=self.get_sort_key_for_node)
        
        log.debug(f"Transformation resulted in {len(unique_nodes)} unique bindable leaf nodes.")
        return unique_nodes