from collections import defaultdict

NODE_COUNTER_TYPES = [
    "calls",
    "errors",
    "successes",
]


class TaiatMetrics:
    def __init__(self):
        self.node_counters = defaultdict(lambda: defaultdict(int))

    def add_node_counter(self, node_name: str):
        for counter_type in NODE_COUNTER_TYPES:
            self.node_counters[node_name][counter_type] = 0
