import logging

import networkx as nx

try:
    import matplotlib.pyplot as plt
    import pydot

    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

log = logging.getLogger(__name__)


class TopologyVisualizer:
    """
    Creates a simplified, contextual block diagram of a hardware allocation
    that shows assigned nodes in the context of their unassigned siblings.
    """

    def __init__(self, topology: "HwlocTopology", assigned_nodes: list, affinity_target=None):
        if not VISUALIZATION_ENABLED:
            raise ImportError("Visualization libraries (matplotlib, pydot) are not installed.")

        self.topology = topology
        self.assigned_nodes = assigned_nodes
        self.assigned_gps = {gp for gp, _ in assigned_nodes}
        self.affinity_target_gp = affinity_target[0] if affinity_target else None
        self.title = "Hardware Allocation"  # Public attribute for a descriptive title

    def _build_contextual_subgraph(self):
        """
        Constructs a new, clean graph for drawing that includes assigned nodes,
        their unassigned siblings, and their parent containers for context.
        """
        if not self.assigned_nodes:
            return nx.DiGraph()

        leaf_type = self.assigned_nodes[0][1].get("type")
        if not leaf_type:
            return nx.DiGraph()

        first_node_gp = self.assigned_nodes[0][0]

        # We respect your capitalization fix here.
        parent = self.topology.get_ancestor_of_type(
            first_node_gp, "Package"
        ) or self.topology.get_ancestor_of_type(first_node_gp, "NUMANode")

        search_domain_gp = None
        if parent:
            search_domain_gp = parent[0]
        elif leaf_type in ["Package", "NUMANode", "Machine"]:
            search_domain_gp = first_node_gp

        if search_domain_gp:
            all_siblings = self.topology.get_descendants(search_domain_gp, type=leaf_type)
            if not all_siblings and leaf_type in ["Package", "NUMANode"]:
                all_siblings = self.assigned_nodes
        else:
            all_siblings = self.assigned_nodes

        nodes_to_draw_gps = set()
        for gp, _ in all_siblings:
            nodes_to_draw_gps.add(gp)
            nodes_to_draw_gps.update(nx.ancestors(self.topology.hierarchy_view, gp))

        final_subgraph = self.topology.graph.subgraph(nodes_to_draw_gps).copy()
        return final_subgraph

    def draw(self, filename: str):
        # This method's logic was already correct and does not need to change.
        log.info(f"Generating allocation graphic at '{filename}'...")

        subgraph = self._build_contextual_subgraph()
        if subgraph.number_of_nodes() == 0:
            log.warning("Cannot generate graphic: No nodes to draw.")
            return

        labels = {}
        colors = {}
        sorted_nodes = sorted(
            subgraph.nodes(data=True),
            key=lambda item: (item[1].get("depth", 0), self.topology.get_sort_key_for_node(item)),
        )

        for gp, data in sorted_nodes:
            node_type = data.get("type", "Unknown")
            os_index = data.get("os_index")
            labels[gp] = (
                f"{node_type.capitalize()}:{os_index}"
                if os_index is not None
                else node_type.capitalize()
            )
            # Color logic is unchanged...
            if gp == self.affinity_target_gp:
                colors[gp] = "gold"
            elif gp in self.assigned_gps:
                colors[gp] = "lightgreen"
            elif node_type == "numanode":
                colors[gp] = "skyblue"
            elif node_type == "package":
                colors[gp] = "coral"
            else:
                colors[gp] = "lightgray"

        node_colors = [colors.get(gp, "lightgray") for gp in subgraph.nodes()]
        pos = nx.drawing.nx_pydot.graphviz_layout(subgraph, prog="dot")

        plt.figure(figsize=(12, 8))
        nx.draw_networkx(
            subgraph,
            pos,
            labels=labels,
            node_color=node_colors,
            node_size=2000,
            node_shape="s",
            edgecolors="black",
            font_size=8,
            font_weight="bold",
            arrows=False,
            width=1.5,
        )

        plt.title(self.title, fontsize=16)
        plt.box(False)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=150)
        plt.close()

        log.info("...graphic saved successfully.")
