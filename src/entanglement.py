from typing import Tuple

import networkx as nx
import numpy as np

from simulation import Intersection
from utility import read_intersection_from_json


def is_entangled_cycle(cycle_1, cycle_2, intersection: Intersection):
    cycle_1: Tuple[str] = tuple(cycle_1)
    cycle_2: Tuple[str] = tuple(cycle_2)
    common_czs = set(cycle_1).intersection(cycle_2)
    if len(common_czs) != 1:
        return None
    for traj in intersection.trajectories:
        for cz in common_czs:
            if cz not in traj:
                continue
            traj_i = traj.index(cz)
            if not 1 <= traj_i <= len(traj)-2:
                continue
            i_1 = cycle_1.index(cz)
            i_2 = cycle_2.index(cz)
            if traj[traj_i - 1] == cycle_1[(i_1 - 1) % len(cycle_1)] \
                and traj[traj_i + 1] == cycle_2[(i_2 + 1) % len(cycle_2)]:
                    return cz
    return None


def find_entanglements(intersection: Intersection):
    G = nx.DiGraph()
    for cz1, cz2 in intersection.transitions:
        G.add_edge(cz1, cz2)
    cycles = list(nx.simple_cycles(G))
    entangled_table = []
    for cycle_1 in cycles:
        entangled_table.append([])
        for cycle_2 in cycles:
            entangled = is_entangled_cycle(cycle_1, cycle_2, intersection)
            entangled_table[-1].append(entangled)
    cnt = 0
    for i in range(len(cycles)):
        for j in range(i + 1, len(cycles)):
            if entangled_table[i][j] and entangled_table[i][j] == entangled_table[j][i]:
                print('--------')
                print(entangled_table[i][j])
                print(cycles[i])
                print(cycles[j])
                cnt += 1
                print('--------')

    print(cnt)

if __name__ == "__main__":
    intersection = read_intersection_from_json("../intersection_configs/4-approach-1-lane-24cz.json")
    find_entanglements(intersection)
