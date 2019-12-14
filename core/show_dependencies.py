import networkx as nx
from dbt.config import RuntimeConfig
from dbt.logger import GLOBAL_LOGGER as logger
from core.find import FindTask
import dbt.ui
from pathlib import Path
import matplotlib.pyplot as plt
import random

GRAPH_FILE = "graph.gpickle"


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        print(pos)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


class ShowDependenciesTask(FindTask):
    def __init__(self, args):
        self.args = args
        if self.args.command == "show_upstream":
            self.direction = "upstream"
        elif self.args.command == "show_downstream":
            self.direction = "downstream"
        else:
            raise
        self.config = RuntimeConfig.from_args(args)
        self.model_path = self.config.source_paths[0]
        self.target_path = self.config.target_path
        self.manifest = self._get_manifest()  # Inherited from FindTask

    def _get_graph(self):
        graph_path = Path(self.target_path) / GRAPH_FILE
        graph = nx.read_gpickle(graph_path)
        return graph

    # this traverses an arbitrary tree (parents or children) to get all ancestors or descendants
    def traverse_tree(self, node, d_tree, been_done=set()):
        tree_outputs = set(d_tree.get(node, set()))  # direct relatives
        for key in d_tree.get(node, set()):  # 2nd step relatives
            if key not in been_done:
                been_done.add(key)  # to break any circular references
                tree_outputs = tree_outputs.union(self.traverse_tree(key, d_tree, been_done))
        return tree_outputs

    # this reverses a parent tree to a child tree
    def get_child_dict(self, parent_dict):
        child_dict = {}
        for node in parent_dict:
            for parent in parent_dict[node]:
                if parent not in child_dict.keys():
                    child_dict[parent] = set([node])
                else:
                    child_dict[parent].add(node)
        return child_dict

    def get_node_set(self, parent_dict, focal_set):

        downstream_dict = {}  # intended to store all descendants (any generation)
        upstream_dict = {}  # intended to store all ancestry (any generation)
        node_set = set()

        # reverse parent dict to child so we can look for descendants
        child_dict = self.get_child_dict(parent_dict)  # immediate children

        # build descendant dict
        for node in child_dict:
            downstream_dict[node] = self.traverse_tree(node, child_dict, been_done=set())

        # build ancestor dict
        for node in parent_dict:
            upstream_dict[node] = self.traverse_tree(node, parent_dict, been_done=set())

        for focal_node in focal_set:
            if self.direction == "upstream":
                node_set = upstream_dict.get(focal_node, set())
            else:
                node_set = downstream_dict.get(focal_node, set())
            node_set.add(focal_node)

        return node_set

    def dereference_model_name(self, model_name):
        for name, node in self.manifest["nodes"].items():
            if node["name"] == model_name:
                return name

    def get_node_info(self):

        node_info_dict = {}  # intended to store direct node type
        parent_dict = {}  # intended to store parent data

        for name, node in self.manifest["nodes"].items():
            d = {}
            d["name"] = name
            if node["resource_type"] == "source":
                mat = "source"
                schema = node["schema"]
                alias = node["name"]
                parents = []
            else:
                mat = node["config"]["materialized"]
                if len(node["fqn"]) == 3:
                    schema = node["fqn"][1]
                else:
                    schema = node["fqn"][0]

                alias = node["alias"]
                parents = node["depends_on"]["nodes"]

            d["type"] = mat
            d["alias"] = "{}.{}".format(schema, alias)

            # add the object name and type and direct parents to the dict
            node_info_dict[d["name"]] = d
            parent_dict[d["name"]] = parents

        return (parent_dict, node_info_dict)

    def build_d_graph(self, parent_dict, node_set, node_type_dict):
        G = nx.DiGraph()  # initialize directional graph object

        for node in node_set:
            G.add_node(node)
            for parent in parent_dict.get(node, set()):
                G.add_edge(parent, node)
        return G

    def display_deps(self, viz_dict):
        rev = self.direction == "downstream"
        keylist = list(viz_dict.keys())
        layers = sorted(keylist, reverse=rev)
        print("-" * 80)
        for layer in layers:
            print(viz_dict[layer][0].center(80))
            if len(viz_dict[layer]) > 0:
                print("^".center(80))
                flat_list = [item for sublist in viz_dict[layer][1:] for item in sublist]
                print(" | ".join(flat_list).center(80))
            print("-" * 80)

    def subset_dict(self, d, nodes):
        out_d = {}
        for k, v in d.items():
            if k in nodes:
                dep_set = set()
                for n in d[k]:
                    if n in nodes:
                        dep_set.add(n)
                out_d[k] = dep_set
        return out_d

    def pretty_node_name(self, name):
        # print(self.node_info_dict[name]["alias"])
        return self.node_info_dict[name]["alias"]

    @staticmethod
    def plot_graph(graph) -> None:
        nx.draw(graph, with_labels=True)

    def run(self, args):
        parent_dict, self.node_info_dict = self.get_node_info()
        dbt_name = self.dereference_model_name(self.args.model_name)
        if not dbt_name:
            logger.info(
                dbt.ui.printer.yellow(
                    "Warning: The model argument {} does not match any models "
                    "found in this project:".format(self.args.model_name)
                )
            )
            return {}
        focal_set = set([dbt_name])

        node_set = self.get_node_set(parent_dict, focal_set)

        parent_subset_dict = self.subset_dict(parent_dict, node_set)

        G = self.build_d_graph(parent_subset_dict, node_set, self.node_info_dict)

        viz_dict = {}

        def update_viz_dict(G, current_node, level=0):
            # print(viz_dict)
            if len(G.nodes()) == 0:
                viz_dict[0] = [self.pretty_node_name(dbt_name)]
                return
            if level in viz_dict:
                viz_dict[level].append([self.pretty_node_name(current_node)])
            else:
                viz_dict[level] = [self.pretty_node_name(current_node)]
            if self.direction == "upstream":
                if G.predecessors(current_node) == []:
                    return
                for pred in G.predecessors(current_node):
                    # print(list(G.predecessors(current_node)))
                    update_viz_dict(G, pred, level + 1)
            if self.direction == "downstream":
                if G.successors(current_node) == []:
                    return
                for pred in G.successors(current_node):
                    update_viz_dict(G, pred, level + 1)

        update_viz_dict(G, dbt_name)
        pos = hierarchy_pos(
            G, "source.predictive_modelling.kafka.flight_search_event_result_leg"
        )
        print("POS SPOSPSPS")
        print(pos)
        nx.draw(G, with_labels=True)
        plt.draw()
        plt.show()
        # self.display_deps(viz_dict)
        # return viz_dict
