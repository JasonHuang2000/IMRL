from typing import List

def validate(intersection, valid_set, model):
    from tcg.tcg_env import TcgEnv, TcgEnvWithRollback
    from mb_agg import g_pool_cal
    from agent_utils import sample_select_action
    from agent_utils import greedy_select_action
    import numpy as np
    import torch
    from Params import configs
    from copy import deepcopy

    env = TcgEnvWithRollback()
    device = torch.device(configs.device)
    make_spans = []
    # rollout using model
    for vehicles in valid_set:
        adj, fea, candidate, mask = env.reset(intersection, deepcopy(vehicles))
        # adj, fea, candidate, mask = env.reset(intersection, deepcopy(vehicles), start_idx=0, window_size=len(vehicles))
        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.tcg.num_vertices, env.tcg.num_vertices]),
                             n_nodes=env.tcg.num_vertices,
                             device=device)
        rewards = 0
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask, _ = env.step(action.item())
            rewards += reward
            if done:
                break
        make_spans.append(sum(env.reward_history))
        # print(rewards - env.posRewards)
    return np.array(make_spans)

def validate_sliding_window(intersection, valid_set, model, valid_num_veh=10, stride=5):
    from tcg.tcg_env import TcgEnvWithRollback
    from mb_agg import g_pool_cal
    from agent_utils import greedy_select_action
    import numpy as np
    import torch
    from Params import configs
    from copy import deepcopy
    from simulation.vehicle import Vehicle

    device = torch.device(configs.device)
    make_spans = []

    def normalize_entering_time(vehicles: List[Vehicle]) -> List[Vehicle]:
        min_time = vehicles[0].earliest_arrival_time
        new_vehicles = []
        for veh in vehicles:
            new_veh = Vehicle(
                veh.id,
                earliest_arrival_time=(veh.earliest_arrival_time - min_time),
                trajectory=veh.trajectory,
                src_lane_id=veh.src_lane_id,
                dst_lane_id=veh.dst_lane_id,
                vertex_passing_time=veh.vertex_passing_time,
            )
            new_vehicles.append(new_veh)
        return new_vehicles

    # rollout using model
    for all_vehicles in valid_set:
        tot_delay = 0
        prev_max_leaving_time = 0
        group_num = (len(all_vehicles) - valid_num_veh) // stride + 1
        for group_idx in range(group_num):
            start_idx = stride * group_idx
            # print(f'start_idx: {start_idx}')
            vehicles = all_vehicles[start_idx : start_idx+valid_num_veh]
            env = TcgEnvWithRollback()
            adj, fea, candidate, mask = env.reset(
                intersection,
                normalize_entering_time(vehicles),
                start_idx=start_idx,
                start_offset=0,
                window_size=stride
            )
            g_pool_step = g_pool_cal(
                graph_pool_type=configs.graph_pool_type,
                batch_size=torch.Size([1, env.tcg.num_vertices, env.tcg.num_vertices]),
                n_nodes=env.tcg.num_vertices,
                device=device
            )    
            delay = 0 
            max_leaving_time = 0
            while True:
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                with torch.no_grad():
                    pi, _ = model(
                        x=fea_tensor,
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor,
                        candidate=candidate_tensor.unsqueeze(0),
                        mask=mask_tensor.unsqueeze(0)
                    )
                action = greedy_select_action(pi, candidate)
                adj, fea, reward, done, candidate, mask, max_leaving_time = env.step(action.item())
                delay += (-reward)
                if done:
                    break

            # reward := delay time in current group without considering previous delay
            # actual reward = reward + max(0, prev_max_leaving_time - arrival time of 1st vehicles)
            current_min = vehicles[0].earliest_arrival_time / 10
            max_leaving_time += current_min # recover normalized max leaving time
            # print(f'delay: {delay} / prev_max: {prev_max_leaving_time} / current_min: {current_min} / current_max: {max_leaving_time}')
            tot_delay += (delay + stride * max(0, prev_max_leaving_time - current_min))
            prev_max_leaving_time = max(max_leaving_time, prev_max_leaving_time + (max_leaving_time - current_min))

        left_num = len(all_vehicles) - stride * group_num
        if left_num > 0:
            vehicles = all_vehicles[-valid_num_veh:]
            env = TcgEnvWithRollback()
            adj, fea, candidate, mask = env.reset(
                intersection, 
                normalize_entering_time(vehicles), 
                start_idx=len(all_vehicles) - left_num, 
                start_offset=valid_num_veh - left_num,
                window_size=left_num
            )
            g_pool_step = g_pool_cal(
                graph_pool_type=configs.graph_pool_type,
                batch_size=torch.Size([1, env.tcg.num_vertices, env.tcg.num_vertices]),
                n_nodes=env.tcg.num_vertices,
                device=device
            )    
            delay = 0 
            max_leaving_time = 0
            while True:
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                with torch.no_grad():
                    pi, _ = model(
                        x=fea_tensor,
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor,
                        candidate=candidate_tensor.unsqueeze(0),
                        mask=mask_tensor.unsqueeze(0)
                    )
                action = greedy_select_action(pi, candidate)
                adj, fea, reward, done, candidate, mask, max_leaving_time = env.step(action.item())
                delay += (-reward)
                if done:
                    break

            # print(f'delay: {delay} / prev_max: {prev_max_leaving_time}')
            tot_delay += (delay + left_num * max(0, prev_max_leaving_time - vehicles[valid_num_veh - left_num].earliest_arrival_time / 10))

        make_spans.append(tot_delay)
        
    return np.array(make_spans)

if __name__ == '__main__':

    from uniform_instance_gen import uni_instance_gen
    import numpy as np
    import time
    import argparse
    from Params import configs

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=20, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=20, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()

    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = params.low
    HIGH = params.high
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m

    from PPO_jssp_multiInstances import PPO
    import torch

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    ppo.policy.load_state_dict(torch.load(path))

    SEEDs = range(0, params.seed, 10)
    result = []
    for SEED in SEEDs:

        np.random.seed(SEED)

        vali_data = [uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH) for _ in range(params.n_vali)]

        makespan = - validate(vali_data, ppo.policy)
        print(makespan.mean())


    # print(min(result))

