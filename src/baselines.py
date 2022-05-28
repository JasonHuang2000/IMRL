import pathlib

import numpy as np

from utility import read_intersection_from_json
from traffic_gen import load_vehicles_from_file, datadir_traffic_generator
from tcg.tcg_env import TcgEnv
from tcg.tcg_numpy import TimingConflictGraphNumpy


def igreedy_policy(env, schedulable_vertices, feature):
    action = schedulable_vertices[feature[schedulable_vertices, 0].argmin()]
    return action


def fcfs_policy(env, schedulable_vertices, feature):
    arr_vec = []
    for vertex in schedulable_vertices:
        vehicle_idx, _ = env.tcg.vertex_to_vehicle_cz[vertex]
        vehicle = env.tcg.vehicle_list[vehicle_idx]
        arrival_time = vehicle.earliest_arrival_time
        arr_vec.append(arrival_time)
    action = schedulable_vertices[np.argmin(arr_vec)]
    return action

def eval_priority_based_policy(intersection, datadir, priority_based_policy):
    avg_reward_sum = 0.0
    num_case = 0
    for vehicles in datadir_traffic_generator(intersection, datadir):
        env = TcgEnv(ensure_deadlock_free=True)
        adj, feature, vertices, mask = env.reset(intersection, vehicles)
        done = False
        reward_sum = 0
        while not done:
            schedulable_vertices = vertices[(mask == 0).nonzero()]
            action = priority_based_policy(env, schedulable_vertices, feature)
            adj, feature, reward, done, vertices, mask = env.step(action)
            reward_sum += reward
        avg_reward_sum += reward_sum
        num_case += 1
    return -avg_reward_sum / num_case


def get_leaving_time_ub(tcg: TimingConflictGraphNumpy):
    last_leaving_time = 0
    for first, last in zip(tcg.first_vertices, tcg.last_vertices):
        entering_time = max(last_leaving_time, tcg.arrival_time[first])
        leaving_time = entering_time + tcg.t1_edge[first:last+1].sum()
        leaving_time += tcg.passing_time[first:last+1].sum()
        last_leaving_time = leaving_time
    return last_leaving_time + 1

def solve_by_cp(tcg: TimingConflictGraphNumpy):
    if tcg.num_vehicles == 0:
        return 0
    
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    leaving_time_ub = get_leaving_time_ub(tcg)
    entering_time_vars = []
    for vertex in range(tcg.num_vertices):
        var = model.NewIntVar(0, leaving_time_ub, f"entering_time_{vertex}")
        entering_time_vars.append(var)
    
    edge_selected_vars = []
    for i, edge in enumerate(np.argwhere(tcg.t3_edge_undecided)):
        var = model.NewBoolVar(f"edge_selected_{edge[0]}_{edge[1]}")
        edge_selected_vars.append(var)

    # arrival time constraints  
    for i, vertex in enumerate(tcg.first_vertices):
        var = entering_time_vars[vertex]
        model.Add(var >= tcg.vehicle_list[i].earliest_arrival_time)

    # Add type-4 edges for only type-2 edges
    for vertex in range(tcg.num_vertices):
        tcg.add_t4_edge(vertex, tcg.t2_edge)

    # edge constraints
    all_edge = tcg.t1_edge + tcg.t2_edge + tcg.t4_edge
    for edge in np.argwhere(all_edge):
        var_pred = entering_time_vars[edge[1]]
        var_succ = entering_time_vars[edge[0]]
        model.Add(var_succ >= var_pred + tcg.passing_time[edge[1]] + all_edge[edge[0], edge[1]])

    for vertex in range(tcg.num_vertices):
        tcg.add_t4_edge(vertex, tcg.t3_edge_undecided)

    pairing = {}
    for i, edge in enumerate(np.argwhere(tcg.t3_edge_undecided)):
        disjunctive_edge_idx = pairing.get((edge[1], edge[0]), None)
        if disjunctive_edge_idx is not None:
            disjunctive_var = edge_selected_vars[disjunctive_edge_idx]
            selected_var = edge_selected_vars[i]
            model.AddBoolXOr([selected_var, disjunctive_var])
        else:
            pairing[(edge[0], edge[1])] = i
    
    for i, edge in enumerate(np.argwhere(tcg.t3_edge_undecided)):
        entering_var_pred = entering_time_vars[edge[1]]
        entering_var_succ = entering_time_vars[edge[0]]
        selected_var = edge_selected_vars[i]
        model.Add(entering_var_succ >= entering_var_pred
                    + tcg.passing_time[edge[1]] + tcg.t3_edge_undecided[edge[0], edge[1]]).OnlyEnforceIf(
                    selected_var)
        if edge[1] not in tcg.last_vertices:
            var_pred = entering_time_vars[edge[1] + 1]
            var_succ = entering_time_vars[edge[0]]
            model.Add(var_succ >= var_pred + tcg.passing_time[edge[1] + 1]
                + tcg.t4_edge[edge[0], edge[1] + 1]).OnlyEnforceIf(selected_var)
    
    model.Minimize(
        sum(entering_time_vars[last] + tcg.passing_time[last] - tcg.arrival_time[first]
            for first, last in zip(tcg.first_vertices, tcg.last_vertices))
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        obj = solver.ObjectiveValue()
        obj -= tcg.t1_edge.sum() + tcg.passing_time.sum()
        return obj / 10

    raise Exception()


def optimal_sol(intersection, datadir):
    num_case = 0
    avg = 0.0
    for vehicles in datadir_traffic_generator(intersection, datadir):
        tcg = TimingConflictGraphNumpy(intersection, vehicles)
        avg += solve_by_cp(tcg)
        num_case += 1
    return avg / num_case


if __name__ == "__main__":
    datadir = pathlib.Path("../testdata/validation-d0.9-100/")
    intersection = read_intersection_from_json("../intersection_configs/2x2.json")

    igreedy_result = eval_priority_based_policy(intersection, datadir, igreedy_policy)
    fcfs_result = eval_priority_based_policy(intersection, datadir, fcfs_policy)
    opt_result = optimal_sol(intersection, datadir)

    print(f"OPT     : {opt_result:0.6f}")
    print(f"IGreedy : {igreedy_result:0.6f}")
    print(f"FCFS    : {fcfs_result:0.6f}")
