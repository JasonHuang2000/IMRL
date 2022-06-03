from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
import sys

device = torch.device(configs.device)


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            # process each env data
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        # get batch argument for net forwarding: mb_g_pool is same for all env
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def main():
    from traffic_gen import datadir_traffic_generator, random_traffic_generator
    from tcg.tcg_env import TcgEnv, TcgEnvWithRollback
    from simulation import Intersection, Vehicle
    from utility import read_intersection_from_json

    num_vehicles: int = 20
    configs.n_j = num_vehicles

    intersection: Intersection = read_intersection_from_json("../intersection_configs/4-approach-1-lane-16cz.json")
    configs.n_m = len(intersection.conflict_zones)
    envs = [TcgEnvWithRollback() for _ in range(configs.num_envs)]

    validation_gen = datadir_traffic_generator(intersection, "../testdata/4-approach-1-lane-16cz-n20/")
    valid_data = [testcase for testcase in validation_gen]

    training_gen = random_traffic_generator(
        intersection,
        num_iter = 0, # infinite
        vehicle_num = num_vehicles,
        poisson_parameter_list = [0.7]
    )

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j,
              n_m=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    # training loop
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000
    for i_update in range(configs.max_updates):

        t3 = time.time()

        ep_rewards = [0 for _ in range(configs.num_envs)]
        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []
        
        vehicles = next(training_gen)
        envs[0].reset(intersection, deepcopy(vehicles))
        num_vertices: int = envs[0].tcg.num_vertices

        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=torch.Size([1, num_vertices, num_vertices]),
                            n_nodes=num_vertices,
                            device=device)

        def collect_memory(env, intersection, vehicles, memory):
            adj, fea, candidate, mask = env.reset(intersection, vehicles)
            while not env.done():
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

                with torch.no_grad():
                    pi, _ = ppo.policy_old(
                        x=fea_tensor,
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor,
                        candidate=candidate_tensor.unsqueeze(0),
                        mask=mask_tensor.unsqueeze(0)
                    )
                    action, a_idx = select_action(pi, candidate, memory)
                
                env.adj_history[-1] = adj_tensor
                env.feature_history[-1] = fea_tensor
                env.vertices_history[-1] = candidate_tensor
                env.mask_history[-1] = mask_tensor

                adj, fea, reward, done, candidate, mask = env.step(action.item())
                while len(env.reward_history) < len(memory.logprobs):
                    memory.logprobs.pop(-1)
                
            for i in range(num_vertices):
                memory.adj_mb.append(env.adj_history[i])
                memory.fea_mb.append(env.feature_history[i])
                memory.candidate_mb.append(env.vertices_history[i])
                memory.mask_mb.append(env.mask_history[i])
                memory.a_mb.append(env.a_idx_history[i])
                memory.r_mb.append(env.reward_history[i])
                memory.done_mb.append(i == num_vertices-1)
            
            return sum(env.reward_history)

        for i in range(configs.num_envs):
            ep_rewards[i] = collect_memory(envs[i], intersection, deepcopy(vehicles), memories[i])

        loss, v_loss = ppo.update(memories, num_vertices, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        if (i_update + 1) % 100 == 0:
            file_writing_obj = open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))

        # log results
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, mean_rewards_all_env, v_loss))
        
        # validate and save use mean performance
        t4 = time.time()
        if (i_update + 1) % 100 == 0:
            vali_result = -validate(intersection, valid_data, ppo.policy).mean()
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
                record = vali_result
            print('The validation quality is:', vali_result)
            file_writing_obj1 = open(
                './' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj1.write(str(validation_log))
        t5 = time.time()

        # print('Training:', t4 - t3)
        # print('Validation:', t5 - t4)


if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    # print(total2 - total1)