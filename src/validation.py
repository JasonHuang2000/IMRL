from copy import deepcopy
from typing import List, Tuple

from jenkspy import JenksNaturalBreaks
import numpy as np
import torch

from agent_utils import greedy_select_action
from mb_agg import g_pool_cal
from models.actor_critic import ActorCritic
from Params import configs

from tcg.tcg_env import TcgEnvWithRollback
from simulation import Intersection, Vehicle

class Validation:

    def __init__(
        self,
        intersection: Intersection,
        policy: ActorCritic,
    ) -> None:
        self.env = TcgEnvWithRollback()
        self.device = torch.device(configs.device)
        self.intersection = intersection
        self.policy = policy
        self.strats = {
            'base': self.base,
            'jenks': self.jenks
        }
 
    def schedule(self, vehicles: List[Vehicle]) -> Tuple[float, float]:
        env = self.env
        device = self.device

        adj, fea, candidate, mask = env.reset(self.intersection, deepcopy(vehicles))
        g_pool_step = g_pool_cal(
            graph_pool_type=configs.graph_pool_type,
            batch_size=torch.Size([1, env.tcg.num_vertices, env.tcg.num_vertices]),
            n_nodes=env.tcg.num_vertices,
            device=device
        )
        tot_reward = .0
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = self.policy(
                    x=fea_tensor,
                    n_j=len(vehicles),
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask, max_leaving_time = env.step(action.item())
            tot_reward += reward
            if done:
                break
        
        return tot_reward, max_leaving_time

    def base(self, vehicles: List[Vehicle]) -> float:
        reward, _ = self.schedule(vehicles)
        return reward

    def jenks(self, vehicles: List[Vehicle]) -> float:
        exp_group_num = len(vehicles) // 10
        if exp_group_num <= 1:
            return self.base(vehicles)
        arr_time = [veh.earliest_arrival_time for veh in vehicles]
        jnb = JenksNaturalBreaks(exp_group_num)
        jnb.fit(arr_time)
        prev_max_lt = 0
        tot_reward = 0
        for group_idx in range(exp_group_num):
            veh_group = [veh for veh, gi in zip(vehicles, jnb.labels_) if gi == group_idx]
            min_at = veh_group[0].earliest_arrival_time / 10
            reward, duration = self.schedule(self._normalize_entering_time(veh_group))
            if prev_max_lt > min_at:
                reward -= len(veh_group) * (prev_max_lt - min_at)
            tot_reward += reward
            prev_max_lt = max(min_at + duration, prev_max_lt + duration)
        return tot_reward

    def _normalize_entering_time(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        min_time = vehicles[0].earliest_arrival_time
        new_vehicles = []
        for i, veh in enumerate(vehicles):
            new_veh = Vehicle(
                f"vehicle-{i}",
                earliest_arrival_time=(veh.earliest_arrival_time - min_time),
                trajectory=veh.trajectory,
                src_lane_id=veh.src_lane_id,
                dst_lane_id=veh.dst_lane_id,
                vertex_passing_time=veh.vertex_passing_time,
            )
            new_vehicles.append(new_veh)
        return new_vehicles

def validate(
    intersection: Intersection,
    valid_set: List[List[Vehicle]],
    policy: ActorCritic,
    strat: str = "base",
    **kwargs,
) -> np.ndarray:

    validation = Validation(intersection, policy, **kwargs)
    rewards = []

    # rollout using model
    for data in valid_set:
        reward = validation.strats[strat](data)
        rewards.append(reward)

    return np.array(rewards)
