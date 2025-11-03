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
    Creates a visual representation of a hardware allocation on a topology graph.
    This version draws the ENTIRE hardware topology as a stable background.
    """

    def __init__(self, topology, assigned_nodes: list, affinity_target=None):
        if not VISUALIZATION_ENABLED:
            raise ImportError("Visualization libraries (matplotlib, pydot) are not installed.")

        self.topology = topology
        self.assigned_gps = {gp for gp, _ in assigned_nodes}
        self.affinity_target_gp = affinity_target[0] if affinity_target else None
        self.title = "Hardware Allocation"

    def draw(self, filename, width=12, height=8):
        """
        Generates and saves the allocation graph to a file, drawing the
        entire topology and highlighting the assigned resources.
        """
        log.info(f"Generating allocation graphic at '{filename}'...")

        # 1. Start with the complete hierarchy view as the base.
        subgraph = self.topology.hierarchy_view

        # 2. Create a clean copy of the graph to avoid modifying the original.
        clean_subgraph = subgraph.copy()

        # 3. Iterate through all nodes in the copy and remove the conflicting 'name' attribute.
        #    This is a known issue when interfacing networkx with pydot.
        for _, data in clean_subgraph.nodes(data=True):
            if "name" in data:
                del data["name"]

        # All subsequent drawing operations will use this sanitized graph copy.
        subgraph = clean_subgraph

        # Make it pink! Err, green and blue and gray... :)
        labels, colors = {}, {}
        for g, d in subgraph.nodes(data=True):
            labels[g] = f"{d.get('type')}"
            if "os_index" in d:
                labels[g] += f":{d['os_index']}"
            elif "pci_busid" in d:
                labels[g] += f"\n{d.get('pci_busid', '')[:10]}"

            if g == self.affinity_target_gp:
                colors[g] = "gold"
            elif g in self.assigned_gps:
                node_type_key = d.get("device_type") or d.get("type")
                color_map = {
                    "gpu": "orange",
                    "nic": "violet",
                    "Core": "lightgreen",
                    "PU": "lightgreen",
                }
                colors[g] = color_map.get(node_type_key, "lightgreen")
            elif d["type"] == "NUMANode":
                colors[g] = "skyblue"
            else:
                colors[g] = "lightgray"

        node_colors = [colors.get(node, "lightgray") for node in subgraph.nodes()]
        edge_colors = ["black" for _, _, d in subgraph.edges(data=True)]

        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(subgraph, prog="dot")
        except Exception as e:
            log.warning(f"graphviz_layout failed: {e}. Falling back to a simpler layout.")
            pos = nx.spring_layout(subgraph, seed=42)

        plt.figure(figsize=(width, height))
        nx.draw_networkx_nodes(
            subgraph, pos, node_color=node_colors, node_size=1500, edgecolors="black"
        )
        nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, arrows=False, width=1.0)
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)

        plt.title(self.title, fontsize=20)
        plt.box(False)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        log.info("...graphic saved successfully.")
