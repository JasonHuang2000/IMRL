import networkx as nx

from simulation import Intersection
from utility import read_intersection_from_json


class CycleFinder:
    def __init__(self, intersection: Intersection):
        self.intersection = intersection
        self.sorted_transitions = sorted(intersection.transitions)
        G = nx.DiGraph()
        for cz1, cz2 in intersection.transitions:
            G.add_edge(cz1, cz2)
        cycles = list(nx.simple_cycles(G))
        self.cycles_bitset = [
            {trans: [] for trans in self.sorted_transitions}
            for _ in range(max(len(C) for C in cycles) + 1)
        ]
        for C in cycles:
            bitset = 0
            for trans in ((C[i], C[(i + 1) % len(C)]) for i in range(len(C))):
                idx = self.sorted_transitions.index(trans)
                bitset |= 1 << idx
            for trans in ((C[i], C[(i + 1) % len(C)]) for i in range(len(C))):
                self.cycles_bitset[bin(bitset).count("1")][trans].append(bitset)

    def has_cycle(self, transitions, new_trans):
        bitset = 0
        for trans in transitions:
            idx = self.sorted_transitions.index(trans)
            bitset |= 1 << idx
        for cycle_len in range(min(bin(bitset).count("1"), len(self.cycles_bitset)-1), 1, -1):
            for C_bs in self.cycles_bitset[cycle_len][new_trans]:
                if C_bs & bitset == C_bs:
                    return True
        return False


if __name__ == "__main__":
    intersection = read_intersection_from_json(
        "../intersection_configs/4-approach-1-lane.json"
    )
    ctk = CycleFinder(intersection)
    print(ctk.has_cycle([("1", "3"), ("3", "4"), ("4", "2"), ("2", "1")], ("4", "2")))
