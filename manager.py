from functools import partial
from typing import Callable
from taiat.base import TAIAT_TERMINAL_NODE
from taiat.builder import AgentGraphNode, AgentGraphNodeSet, State, START_NODE
from threading import RLock
import time
from langgraph.graph import START



class TaiatManager:
    def __init__(self,
                 node_set: AgentGraphNodeSet,
                 reverse_plan_edges: dict[str, list[str]],
                 wait_interval: int = 5,
                 verbose: bool = False,
        ):
        self.interval = wait_interval
        self.node_set = node_set
        self.reverse_plan_edges = reverse_plan_edges
        # Build reverse graph to find next node to run.
        self.plan_edges = {k.name: [] for k in self.node_set.nodes + [START_NODE]}
        for node, neighbors in reverse_plan_edges.items():
            for neighbor in neighbors:
                self.plan_edges[neighbor].append(node)
        self.output_status = {k.name: "pending" for k in self.node_set.nodes}
        self.output_status[START] = "running"
        self.output_status[TAIAT_TERMINAL_NODE] = "pending"
        self.status_lock = RLock()
        self.verbose = verbose
    def make_router_function(
            self,
            node) -> Callable:
        return partial(
            self.router_function,
            node=node,
        )

    def router_function(
            self,
            state : State,
            node: str,
        ) -> str | None:
        with self.status_lock:
            self.output_status[node] = "done"
        if self.verbose:
            print("node", node, "complete, looking for next node")
            print("next nodes:", self.plan_edges[node])
        # Look at forward neighbors first.
        # Find one that has all its dependencies satisfied.
        for i in range(len(self.plan_edges[node])):
            neighbor = self.plan_edges[node][i]
            back_neighbors = self.reverse_plan_edges.get(neighbor)
            while True:
                if not back_neighbors:
                    # No dependencies, run immediately.
                    with self.status_lock:
                        if self.verbose:
                            print(f"running {neighbor}, no dependencies, from {node}")
                        self.output_status[neighbor] = "running"
                    return neighbor
                else:
                    if all(
                        self.output_status[back_neighbor] == "done" \
                        for back_neighbor in back_neighbors) and \
                        self.output_status[neighbor] == "pending":
                        # This node is ready to run.
                        with self.status_lock:
                            if self.verbose:
                                print(f"running {neighbor}, whose needs {back_neighbors} are now satisfied, from {node}")
                            self.output_status[neighbor] = "running"
                        return neighbor
                    else:
                        # This node is not ready to run, so we need to find the next node to run
                        # by looking backwards in the graph and finding one, any one.
                        next_neighbors = []
                        for back_neighbor in back_neighbors:
                            if self.output_status[back_neighbor] == "pending":
                                neighbor = back_neighbor
                                next_neighbors = self.reverse_plan_edges.get(back_neighbor)
                                break
                        back_neighbors = next_neighbors
        # If we're here, we failed to find a next node to run, which means
        # we should be done, or else something went wrong.
        return TAIAT_TERMINAL_NODE