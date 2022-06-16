from typing import Dict, Iterable, List, Tuple, Union
from copy import deepcopy

import numpy as np
import networkx as nx
from simulation import vehicle

from simulation.intersection import Intersection
from simulation.vehicle import Vehicle


class DeadlockException(Exception):
    pass


class TimingConflictGraphNumpy:
    """
    Timing Conflict Graph implented by NumPy
    """

    def __init__(
        self,
        intersection: Intersection,
        vehicles: Iterable[Vehicle],
        def_edge_waiting_time: Dict[int, int] = {1: 1, 2: 2, 3: 3},
    ):
        self.intersection: Intersection = intersection
        self.vehicle_list: List[Vehicle] = sorted(
            vehicles, key=lambda vehicle: vehicle.id
        )
        self.def_edge_waiting_time: Dict[int, int] = def_edge_waiting_time

        self.num_vehicles: int = len(vehicles)
        self.num_conflict_zones = len(intersection.conflict_zones)

        # calculate the number of vertices
        self.num_vertices: int = 0
        self.vehicle_cz_to_vertex: Dict[Tuple[int, str], int] = {}
        self.vertex_to_vehicle_cz: List[Tuple[int, str]] = []

        self.first_vertices: List[int] = []
        self.last_vertices: List[int] = []

        passing_time_list: List[int] = []
        arrival_time_list: List[int] = []

        for vehicle_idx, vehicle in enumerate(self.vehicle_list):
            self.num_vertices += len(vehicle.trajectory)
            arrival_time_list.append(vehicle.earliest_arrival_time)
            arrival_time_list.extend([0] * (len(vehicle.trajectory) - 1))
            first_vertex = None
            last_vertex = None

            for cz_id in vehicle.trajectory:
                idx: int = len(self.vertex_to_vehicle_cz)
                self.vehicle_cz_to_vertex[(vehicle_idx, cz_id)] = idx
                self.vertex_to_vehicle_cz.append((vehicle_idx, cz_id))
                first_vertex = idx if first_vertex is None else first_vertex
                last_vertex = idx

                passing_time_list.append(vehicle.vertex_passing_time)

            self.first_vertices.append(first_vertex)
            self.last_vertices.append(last_vertex)

        self.vehicle_progress: np.ndarray = np.array(
            self.first_vertices, dtype=np.int32
        )

        # initialize (reversed) adjacency matrices
        mat_shape = (self.num_vertices, self.num_vertices)
        self.t1_edge: np.ndarray = np.zeros(mat_shape, dtype=np.int32)
        self.t2_edge: np.ndarray = np.zeros(mat_shape, dtype=np.int32)
        self.t3_edge: np.ndarray = np.zeros(mat_shape, dtype=np.int32)
        self.t3_edge_undecided: np.ndarray = np.zeros(mat_shape, dtype=np.int32)
        self.t4_edge: np.ndarray = np.zeros(mat_shape, dtype=np.int32)

        # record the "minimum" set of edges to maintain the decided precedence
        self.t3_edge_min: np.ndarray = np.zeros(mat_shape, dtype=np.int32)
        self.t4_edge_min: np.ndarray = np.zeros(mat_shape, dtype=np.int32)

        # initialize passing time array, arrival time array
        self.passing_time: np.ndarray = np.array(passing_time_list, dtype=np.int32)
        self.arrival_time: np.ndarray = np.array(arrival_time_list, dtype=np.int32)

        # add type-1 edges
        for vehicle_idx, vehicle in enumerate(self.vehicle_list):
            for i, cz_id in enumerate(vehicle.trajectory[:-1]):
                next_cz_id: str = vehicle.trajectory[i + 1]
                vertex_1: int = self.vehicle_cz_to_vertex[vehicle_idx, cz_id]
                vertex_2: int = self.vehicle_cz_to_vertex[vehicle_idx, next_cz_id]
                self.t1_edge[vertex_2, vertex_1] = self.def_edge_waiting_time[1]

        tmp_mask = self.t1_edge != 0
        self.t1_next: np.ndarray = np.where(
            tmp_mask.any(axis=0), tmp_mask.argmax(axis=0), -1
        )
        self.t1_prev: np.ndarray = np.where(
            tmp_mask.any(axis=1), tmp_mask.argmax(axis=1), -1
        )

        # add type-2, type-3 edges
        for vertex_1 in range(self.num_vertices - 1):
            for vertex_2 in range(vertex_1 + 1, self.num_vertices):
                vehicle_1, cz_1 = self.vertex_to_vehicle_cz[vertex_1]
                vehicle_2, cz_2 = self.vertex_to_vehicle_cz[vertex_2]
                vehicle_1 = self.vehicle_list[vehicle_1]
                vehicle_2 = self.vehicle_list[vehicle_2]
                if vehicle_1.id == vehicle_2.id or cz_1 != cz_2:
                    continue
                if vehicle_1.src_lane_id == vehicle_2.src_lane_id:
                    if (
                        vehicle_1.earliest_arrival_time
                        <= vehicle_2.earliest_arrival_time
                    ):
                        self.t2_edge[vertex_2, vertex_1] = self.def_edge_waiting_time[2]
                    else:
                        self.t2_edge[vertex_1, vertex_2] = self.def_edge_waiting_time[2]
                else:
                    self.t3_edge_undecided[
                        vertex_2, vertex_1
                    ] = self.def_edge_waiting_time[3]
                    self.t3_edge_undecided[
                        vertex_1, vertex_2
                    ] = self.def_edge_waiting_time[3]

        # add type-4 edges for type-2 edges
        for vertex in range(self.num_vertices):
            self.add_t4_edge(vertex, self.t2_edge)

        # initialize earliest entering time for each vertex
        self.entering_time_lb: np.ndarray = np.zeros(self.num_vertices, dtype=np.int32)
        self.entering_time_lb_loose: np.ndarray = np.zeros(
            self.num_vertices, dtype=np.int32
        )
        self.update_entering_time_lb()

        # the leaving time without delay
        leaving_time_lb: List[int] = []
        for vehicle, start, end in zip(
            self.vehicle_list, self.first_vertices, self.last_vertices
        ):
            leaving_time = (
                vehicle.earliest_arrival_time
                + self.passing_time[start : end + 1].sum()
                + self.t1_edge[start : end + 1].sum()
            )
            leaving_time_lb.append(leaving_time)
        self.leaving_time_lb: np.ndarray = np.array(leaving_time_lb)

        # initialize scheduled indicator
        self.is_scheduled: np.ndarray = np.zeros(self.num_vertices, dtype=np.bool_)

        self.cz_history: Dict[str, List[int]] = {
            cz: [] for cz in self.intersection.conflict_zones
        }

    @property
    def schedulable_vertices(self) -> List[int]:
        adj_rev = self.t1_edge + self.t2_edge + self.t3_edge + self.t4_edge
        res = []
        for vertex in range(self.num_vertices):
            if (
                not self.is_scheduled[vertex]
                and self.is_scheduled[adj_rev[vertex].nonzero()].all()
            ):
                res.append(vertex)
        return res

    @property
    def schedulable_mask(self) -> np.ndarray:
        mask: np.ndarray = np.full(self.num_vehicles, fill_value=1, dtype=bool)
        for vertex in self.schedulable_vertices:
            for i, front_vertex in enumerate(self.vehicle_progress):
                if vertex == front_vertex:
                    mask[i] = 0
        return mask

    @property
    def front_unscheduled_vertices(self) -> np.ndarray:
        return np.array(self.vehicle_progress)

    def add_t4_edge(self, vertex: int, edge: np.ndarray) -> None:
        next = self.t1_next[vertex]
        if next == -1:
            return
        children = edge[:, vertex].nonzero()[0]
        self.t4_edge[children, next] = (
            edge[children, vertex]
            - self.t1_edge[next, vertex]
            - self.passing_time[next]
        )

    def remove_t4_edge(self, vertex: int, edge: np.ndarray) -> None:
        next = self.t1_next[vertex]
        if next == -1:
            return
        children = edge[:, vertex].nonzero()[0]
        self.t4_edge[children, next] = 0

    def add_t4_edge_min(self, vertex: int, edge: np.ndarray) -> None:
        next = self.t1_next[vertex]
        if next == -1:
            return
        children = edge[:, vertex].nonzero()[0]
        self.t4_edge_min[children, next] = (
            edge[children, vertex]
            - self.t1_edge[next, vertex]
            - self.passing_time[next]
        )

    def remove_t4_edge_min(self, vertex: int, edge: np.ndarray) -> None:
        next = self.t1_next[vertex]
        if next == -1:
            return
        children = edge[:, vertex].nonzero()[0]
        self.t4_edge_min[children, next] = 0

    def update_entering_time_lb(self) -> None:
        adj_mat_rev: np.ndarray = (
            self.t1_edge + self.t2_edge + self.t3_edge + self.t4_edge
        )
        topo_order: Union[None, List[int]] = self.get_topological_order(adj_mat_rev)
        if topo_order is None:
            raise DeadlockException()

        for vertex in topo_order:
            parents = np.nonzero(adj_mat_rev[vertex])
            lb: np.ndarray = (
                self.entering_time_lb[parents]
                + self.passing_time[parents]
                + adj_mat_rev[vertex][parents]
            )
            self.entering_time_lb[vertex] = lb.max(initial=self.arrival_time[vertex])

    def get_topological_order(self, adj_mat_rev: np.ndarray) -> Union[None, List[int]]:
        ref_count: np.ndarray = np.count_nonzero(adj_mat_rev, axis=1)

        res = []
        while len(res) < self.num_vertices:
            no_ref = np.nonzero(ref_count == 0)[0].tolist()
            if len(no_ref) == 0:
                return None  # cycle detected (deadlock)
            for vertex in no_ref:
                children = np.nonzero(adj_mat_rev[:, vertex])
                ref_count[children] -= 1
            ref_count[no_ref] = -1
            res.extend(no_ref)

        return res

    def schedule_vertex(self, vertex: int):
        assert not self.is_scheduled[vertex]

        self.is_scheduled[vertex] = np.True_

        # make all outgoing undecided type-3 edges become decided
        self.t3_edge[:, vertex] = self.t3_edge_undecided[:, vertex]

        # delete all incoming undecided type-3 edges
        self.t3_edge_undecided[:, vertex].fill(0)
        self.t3_edge_undecided[vertex].fill(0)

        # add Type-4 edges
        self.add_t4_edge(vertex, self.t3_edge)

        # update entering time lower bounds
        deadlock = False
        try:
            self.update_entering_time_lb()
        except DeadlockException:
            deadlock = True

        vehicle_idx, cz_id = self.vertex_to_vehicle_cz[vertex]
        if self.vehicle_progress[vehicle_idx] != self.last_vertices[vehicle_idx]:
            self.vehicle_progress[vehicle_idx] += 1

        preds: int = self.cz_history.get(cz_id, [])
        if len(preds) > 0:
            pred = preds[-1]
            self.t3_edge_min[vertex, pred] = (
                self.t3_edge[vertex, pred] + self.t2_edge[vertex, pred]
            )
            if self.t3_edge_min[vertex, pred] == 0:
                print(vertex, pred)
                print(self.t3_edge_min[vertex])
            assert self.t3_edge_min[vertex, pred] != 0
            self.add_t4_edge_min(pred, self.t3_edge_min)

        self.cz_history[cz_id].append(vertex)

        if deadlock:
            raise DeadlockException()

    def unschedule_vertex(self, vertex: int) -> bool:
        assert self.is_scheduled[vertex]

        self.is_scheduled[vertex] = np.False_

        self.t3_edge_undecided[:, vertex] = self.t3_edge[:, vertex]
        self.t3_edge_undecided[vertex] = self.t3_edge[:, vertex]

        self.remove_t4_edge(vertex, self.t3_edge)
        self.t3_edge[:, vertex].fill(0)

        self.update_entering_time_lb()

        vehicle_idx, cz_id = self.vertex_to_vehicle_cz[vertex]
        if vertex != self.last_vertices[vehicle_idx]:
            self.vehicle_progress[vehicle_idx] -= 1

        preds: int = self.cz_history.get(cz_id, [])
        assert preds.pop(-1) == vertex

        if len(preds) > 0:
            pred = preds[-1]
            self.remove_t4_edge_min(pred, self.t3_edge_min)
            self.t3_edge_min[vertex, pred] = 0

    def schedule_vertex_test(self, vertex: int) -> bool:
        """
        return False and roll back if it causes a deadlock
        """
        t3_edge_cpy = self.t3_edge.copy()
        t3_edge_undecided_cpy = self.t3_edge_undecided.copy()
        t4_edge_cpy = self.t4_edge.copy()
        vehicle_progress_cpy = self.vehicle_progress.copy()

        try:
            self.schedule_vertex(vertex)
        except DeadlockException:
            self.t3_edge = t3_edge_cpy
            self.t3_edge_undecided = t3_edge_undecided_cpy
            self.t4_edge = t4_edge_cpy
            self.vehicle_progress = vehicle_progress_cpy
            self.is_scheduled[vertex] = np.False_
            return False

        return True

    def test_deadlock(self, vertex: int) -> bool:
        """
        return True if scheduling this vertex causes a deadlock
        """
        t3_edge_cpy = self.t3_edge.copy()
        t3_edge_undecided_cpy = self.t3_edge_undecided.copy()
        t4_edge_cpy = self.t4_edge.copy()
        vehicle_progress_cpy = self.vehicle_progress.copy()
        t3_edge_min_cpy = self.t3_edge_min.copy()
        cz_history_cpy = deepcopy(self.cz_history)
        res: bool = False

        try:
            self.schedule_vertex(vertex)
        except DeadlockException:
            res = True

        self.t3_edge = t3_edge_cpy
        self.t3_edge_undecided = t3_edge_undecided_cpy
        self.t4_edge = t4_edge_cpy
        self.vehicle_progress = vehicle_progress_cpy
        self.is_scheduled[vertex] = np.False_
        self.t3_edge_min = t3_edge_min_cpy
        self.cz_history = cz_history_cpy

        return res

    def get_delay_time(self) -> float:
        leaving_time = (
            self.entering_time_lb[self.last_vertices]
            + self.passing_time[self.last_vertices]
        )
        return np.sum(leaving_time - self.leaving_time_lb) / 10

    def get_last_leaving_time(self) -> float:
        return np.max(self.entering_time_lb + self.passing_time) / 10

    def get_vehicle_by_vertex(self, vertex: int) -> str:
        idx, _ = self.vertex_to_vehicle_cz[vertex]
        return self.vehicle_list[idx].id

    def plot(self, fname="fig.png") -> None:
        G = nx.DiGraph()

        for vertex in range(self.num_vertices):
            vehicle_idx, cz_id = self.vertex_to_vehicle_cz[vertex]
            vehicle = self.vehicle_list[vehicle_idx]
            G.add_node(
                vertex,
                label=f"[{vertex}]\n"
                + f"{vehicle.id}, {cz_id}\n"
                + f"p={self.passing_time[vertex]}, "
                + f"LB={self.entering_time_lb[vertex]}",
            )

        def add_edge(G: nx.DiGraph, adj_mat_rev: np.ndarray, color):
            for edge in np.argwhere(adj_mat_rev):
                G.add_edge(
                    edge[1],
                    edge[0],
                    label=f"{adj_mat_rev.item(tuple(edge.tolist()))}",
                    color=color,
                    weight=adj_mat_rev.item(tuple(edge.tolist())),
                )

        add_edge(G, self.t1_edge, "#000000")
        # add_edge(G, self.t2_edge, "#0c1eeb")
        add_edge(G, self.t3_edge_min, "#fa0e0a")
        add_edge(G, self.t4_edge_min, "#04cf1f")

        a_graph = nx.nx_agraph.to_agraph(G)
        a_graph.layout("dot")
        a_graph.draw(fname)
