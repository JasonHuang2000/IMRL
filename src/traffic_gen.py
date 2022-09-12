from typing import Dict, List, Set, Union
from pathlib import Path
import random
import json

import fire

from simulation import Intersection, Vehicle
from utility import read_intersection_from_json


def get_src_traj_dict(intersection: Intersection):
    traj_per_src = {}
    src_lanes: Dict[str, Set[str]] = intersection.src_lanes
    for src_lane_id in src_lanes:
        traj_per_src[src_lane_id] = {}
    for traj in intersection.trajectories:
        for src_lane_id, associated_czs in src_lanes.items():
            if traj[0] in associated_czs:
                dst_lane_id = intersection.get_traj_dst_lane(traj)
                traj_per_src[src_lane_id][traj] = dst_lane_id
    return traj_per_src

def load_vehicles_from_file(intersection: Intersection, path: Path) -> List[Vehicle]:
    with open(path, "rt", encoding="utf-8") as f:
        vehicle_dicts = json.load(f)

    res = []
    for vehicle_dict in vehicle_dicts:
        new_vehicle: Vehicle = Vehicle(
            vehicle_dict["id"],
            vehicle_dict["earliest_arrival_time"],
            tuple(vehicle_dict["trajectory"]),
            vehicle_dict["src_lane_id"],
            vehicle_dict["dst_lane_id"],
            vehicle_dict["vertex_passing_time"]
        )

        # validation
        assert all(vehicle.id != new_vehicle.id for vehicle in res)
        assert len(new_vehicle.trajectory) > 0
        assert new_vehicle.earliest_arrival_time >= 0
        assert new_vehicle.vertex_passing_time >= 0
        assert new_vehicle.src_lane_id in intersection.src_lanes
        assert new_vehicle.dst_lane_id in intersection.dst_lanes

        res.append(new_vehicle)

    return res

def dump_traffic(path: Path, vehicles: List[Vehicle]) -> None:
    vehicle_dicts = [vehicle.asdict() for vehicle in vehicles]
    with open(path, "wt", encoding="utf-8") as f:
        json.dump(vehicle_dicts, f, indent=2, sort_keys=True)

def random_traffic_generator(
    intersection: Intersection,
    num_iter: int = 10000,
    vehicle_num: int = 10,
    poisson_parameter_list: List[float] = [0.5],
    traffic_behaviour = None,
    traffic_config = None,
):
    continue_condition = lambda _: True
    if num_iter > 0:
        continue_condition = lambda i: i < num_iter

    i = 0
    while continue_condition(i):
        vehicles = []
        src_to_traj = get_src_traj_dict(intersection)
        poisson_param: float = random.choice(poisson_parameter_list)
        t = 0
        cnt = 0

        if traffic_behaviour == 'wave-by-wave':
            wave_size = random.choice(traffic_config['wave_size'])

        if traffic_behaviour == 'unbalanced-traffic':
            prob = { 
                lane_id: traffic_config['ubl_density'][0] / 10 if i == 0
                    else traffic_config['ubl_density'][1] / 10 
                        for i, lane_id in enumerate(intersection.src_lanes)
            }
        else:
            prob = poisson_param / 10

        while True:
            for src_lane_id in intersection.src_lanes:
                if ((traffic_behaviour != 'unbalanced-traffic' and random.random() < prob) or \
                   (traffic_behaviour == 'unbalanced-traffic' and random.random() < prob[src_lane_id])) and \
                   len(vehicles) < vehicle_num:
                    traj = random.choice(list(src_to_traj[src_lane_id].keys()))
                    dst_lane_id = src_to_traj[src_lane_id][traj]
                    new_vehicle = Vehicle(
                        f"vehicle-{len(vehicles)}",
                        t, traj, src_lane_id, dst_lane_id, 
                        vertex_passing_time=random.randint(7, 13)
                    )
                    vehicles.append(new_vehicle)
                    cnt += 1

                if traffic_behaviour == 'wave-by-wave':
                    if cnt == wave_size:
                        t += (traffic_config['wave_gap'] * 10)
                        cnt = 0
                        wave_size = random.choice(traffic_config['wave_size'])

            if len(vehicles) == vehicle_num:
                break
            t += 1
        i += 1
        yield vehicles

def datadir_traffic_generator(intersection: Intersection, data_dir, inf: bool = False):
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise Exception("data_dir is not a directory")

    while True:
        for traffic_file in sorted(data_dir.iterdir(), key=lambda f: f.stem):
            vehicles: List[Vehicle] = load_vehicles_from_file(intersection, traffic_file)
            yield vehicles
        if not inf:
            break

def main(
    intersection_file_path: str,
    output_dir: str = "./",
    seed: int = 0,
    num: int = 10,
    vehicle_num: int = 10,
    poisson_parameter_list: List = [0.5],
    traffic_behaviour: Union[str, None] = None,
):
    random.seed(seed)
    intersection: Intersection = read_intersection_from_json(intersection_file_path)
    data_dir = Path(output_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    traffic_config = {
        'wave_size': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'wave_gap': 10,
        'ubl_density': [0.9, 0.1],
    }

    gen = random_traffic_generator(
        intersection,
        num_iter=num,
        vehicle_num=vehicle_num,
        poisson_parameter_list=poisson_parameter_list,
        traffic_behaviour=traffic_behaviour,
        traffic_config=traffic_config,
    )
    for i, vehicles in enumerate(gen):
        dump_traffic(data_dir / f"{i}.json", vehicles)


if __name__ == "__main__":
    fire.Fire(main)
