from typing import Iterable

import numpy as np

from simulation import Intersection, Vehicle
from cycle_finder import CycleFinder
from .tcg_numpy import TimingConflictGraphNumpy, DeadlockException


ROLLBACK_SANITY_CHECK = False
ROLLBACK_THRESHOLD = 6


class TcgEnv:
    def __init__(self, ensure_deadlock_free: bool = True):
        self.ensure_deadlock_free: bool = ensure_deadlock_free

    def done(self) -> bool:
        return self.tcg.is_scheduled.all()

    def _test_deadlock(self, vertex: int) -> bool:
        vertex2 = self.tcg.t1_next[vertex]
        if vertex2 == -1:
            return False
        cz2 = self.tcg.vertex_to_vehicle_cz[vertex2][1]
        cz1 = self.tcg.vertex_to_vehicle_cz[vertex][1]

        vertex0 = self.tcg.t1_prev[vertex]
        if vertex0 != -1:
            cz0 = self.tcg.vertex_to_vehicle_cz[vertex0][1]
            self.transitions.remove((cz0, cz1))

        self.transitions.add((cz1, cz2))
        deadlock = self.cycle_finder.has_cycle(self.transitions, (cz1, cz2))
        self.transitions.remove((cz1, cz2))
        if vertex0 != -1:
            self.transitions.add((cz0, cz1))

        return deadlock

    def make_state(self):
        adj: np.ndarray = self.tcg.t1_edge + self.tcg.t3_edge_min + self.tcg.t4_edge_min
        adj = (adj != 0).astype(np.single)
        adj += np.eye(adj.shape[0], dtype=np.single)

        entering_time_lb: np.ndarray = self.tcg.entering_time_lb
        is_scheduled: np.ndarray = self.tcg.is_scheduled
        feature: np.ndarray = np.concatenate(
            [
                entering_time_lb.reshape(-1, 1).astype(np.single) / 10,
                is_scheduled.reshape(-1, 1).astype(np.single),
            ],
            axis=1,
        )

        front_vertices: np.ndarray = self.tcg.front_unscheduled_vertices.astype(
            np.int64
        )
        mask: np.ndarray = self.tcg.schedulable_mask

        if self.ensure_deadlock_free:
            for i, vertex in enumerate(front_vertices):
                if not mask[i]:
                    deadlock: bool = self._test_deadlock(vertex)
                    if deadlock:
                        mask[i] = 1
        return adj, feature, front_vertices, mask

    def step(self, action: int):
        self.tcg.schedule_vertex(action)

        cz0 = self.tcg.vertex_to_vehicle_cz[self.tcg.t1_prev[action]][1]
        cz1 = self.tcg.vertex_to_vehicle_cz[action][1]
        cz2 = self.tcg.vertex_to_vehicle_cz[self.tcg.t1_next[action]][1]
        if self.tcg.t1_prev[action] != -1:
            self.transitions.remove((cz0, cz1))
        if self.tcg.t1_next[action] != -1:
            self.transitions.add((cz1, cz2))

        adj, feature, front_vertices, mask = self.make_state()
        cur_delay_time = self.tcg.get_delay_time()
        reward = self.prev_delay_time - cur_delay_time
        self.prev_delay_time = cur_delay_time
        return adj, feature, reward, self.done(), front_vertices, mask

    def reset(self, intersection: Intersection, vehicles: Iterable[Vehicle]):
        self.tcg = TimingConflictGraphNumpy(intersection, vehicles)
        self.prev_delay_time = 0
        self.cycle_finder = CycleFinder(intersection)
        self.transitions = set()
        return self.make_state()


class TcgEnvWithRollback:
    def done(self) -> bool:
        return self.tcg.is_scheduled.all()

    def make_state(self):
        adj: np.ndarray = self.tcg.t1_edge + self.tcg.t3_edge_min + self.tcg.t4_edge_min
        adj = (adj != 0).astype(np.single)
        adj += np.eye(adj.shape[0], dtype=np.single)

        entering_time_lb: np.ndarray = self.tcg.entering_time_lb
        is_scheduled: np.ndarray = self.tcg.is_scheduled
        feature: np.ndarray = np.concatenate(
            [
                entering_time_lb.reshape(-1, 1).astype(np.single) / 10,
                is_scheduled.reshape(-1, 1).astype(np.single),
            ],
            axis=1,
        )

        front_vertices: np.ndarray = self.tcg.front_unscheduled_vertices.astype(
            np.int64
        )
        mask: np.ndarray = self.tcg.schedulable_mask
        for a in self.blocked_actions[-1]:
            mask[np.where(front_vertices == a)] = 1

        return adj, feature, front_vertices, mask

    def step(self, action: int):
        assert action not in self.blocked_actions[-1]
        deadlock = False

        try:
            self.tcg.schedule_vertex(action)
        except DeadlockException:
            deadlock = True

        self.blocked_actions.append([])
        self.action_history.append(action)
        if isinstance(self.vertices_history[-1], np.ndarray):
            self.a_idx_history.append(
                np.argwhere(self.vertices_history[-1] == action)[0][0]
            )
        else:
            self.a_idx_history.append(
                np.argwhere(self.vertices_history[-1].cpu() == action)[0][0]
            )

        adj, feature, front_vertices, mask = self.make_state()
        self.adj_history.append(adj)
        self.feature_history.append(feature)
        self.vertices_history.append(front_vertices)
        self.mask_history.append(mask)

        cur_delay_time = self.tcg.get_delay_time()
        self.delay_time_history.append(cur_delay_time)

        reward = self.prev_delay_time - cur_delay_time
        self.prev_delay_time = cur_delay_time
        self.reward_history.append(reward)

        phy_state = self.get_phy_state()

        rollback_cnt = 0
        while deadlock or phy_state in self.dead_states:
            rollback_cnt += 1
            self.dead_states.add(phy_state)
            reward = 0
            action_causing_deadlock = self.action_history[-1]

            if rollback_cnt > ROLLBACK_THRESHOLD:
                entrance_v = action
                while entrance_v not in self.tcg.first_vertices:
                    entrance_v -= 1
                self.rollback_until(entrance_v)
                self.blocked_actions[-1].append(entrance_v)
            else:
                self.rollback()
                self.blocked_actions[-1].append(action_causing_deadlock)

            adj, feature, front_vertices, mask = self.make_state()
            for a in self.blocked_actions[-1]:
                mask[(front_vertices == a).nonzero()] = 1
            deadlock = mask.all()
            phy_state = self.get_phy_state()

        # if deadlock occurs, the returned reward is meaningless
        return adj, feature, reward, self.done(), front_vertices, mask

    def reset(self, intersection: Intersection, vehicles: Iterable[Vehicle]):
        self.tcg = TimingConflictGraphNumpy(intersection, vehicles)
        self.prev_delay_time = 0
        self.blocked_actions = [[]]

        adj, feature, front_vertices, mask = self.make_state()

        # history
        self.adj_history = [adj]
        self.feature_history = [feature]
        self.vertices_history = [front_vertices]
        self.mask_history = [mask]
        self.a_idx_history = []
        self.action_history = []
        self.reward_history = []
        self.delay_time_history = [0]
        self.dead_states = set()

        return adj, feature, front_vertices, mask

    def rollback(self):
        if len(self.action_history) == 0:
            return

        self.tcg.unschedule_vertex(self.action_history[-1])
        self.action_history.pop(-1)
        self.reward_history.pop(-1)
        self.adj_history.pop(-1)
        self.feature_history.pop(-1)
        self.vertices_history.pop(-1)
        self.mask_history.pop(-1)
        self.a_idx_history.pop(-1)

        self.delay_time_history.pop(-1)
        self.prev_delay_time = self.delay_time_history[-1]

        self.blocked_actions.pop(-1)

        if ROLLBACK_SANITY_CHECK and isinstance(self.adj_history[-1], np.ndarray):
            adj, feature, front_vertices, mask = self.make_state()
            assert (adj == self.adj_history[-1]).all()

    def rollback_until(self, vertex: int):
        while self.tcg.is_scheduled[vertex]:
            self.rollback()

    def get_phy_state(self):
        progress = self.tcg.vehicle_progress.tolist()
        for i in range(len(progress)):
            if self.tcg.is_scheduled[progress[i]]:
                progress[i] = -1
        return tuple(progress) + (-9999,) + tuple(sorted(self.blocked_actions[-1]))
