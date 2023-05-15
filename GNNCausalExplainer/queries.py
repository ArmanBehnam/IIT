from causal1 import CausalGraph
import torch as T
class Intervention:
    def __init__(self, causal_graph, compute_ctf):
        self.cg = causal_graph
        self.compute_ctf = compute_ctf
        self.reachable_nodes = self.cg.fn

    def calculate(self):
        results = {}
        for v1, reachable_from_v1 in self.reachable_nodes.items():
            for v2 in reachable_from_v1:
                query = CTF(...)  # Define CTF query for intervention from v1 to v2
                result = self.compute_ctf(query)
                results[(v1, v2)] = result
        return results


if __name__ == "__main__":
    cg = CausalGraph(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D")])
    intervention = Intervention(cg)
    interventions = [{'A': 1}, {'C': 1}]
    results = intervention.calculate(interventions)
    for result in results:
        print(result)