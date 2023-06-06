# Reinforcement-Learning-Based Job-Shop Scheduling for Intelligent Intersection Management

This is the source code of the research work "Reinforcement-Learning-Based Job-Shop Scheduling for Intelligent Intersection Management", which is accepted and presented in Design, Automation and Test in Europe Conference (DATE), 2023. For more information, please refer to the presentation slides used in the conference: [link](https://docs.google.com/presentation/d/1ezAF2HkGFpWcbUNxXZ1Wwf22SgyFwFAU/edit?usp=sharing&ouid=111123609203172919021&rtpof=true&sd=true).

Authors: Shao-Ching Huang, Kai-En Lin, Cheng-Yen Kuo, Li-Heng Lin, Muhammed O. Sayin, Chung-Wei Lin.

## Generate Validation Data

Validation data can be generated with the script `traffic_gen.py`. For example, 100 records of traffic data with 10 vehicles, 4 conflict, and 0.7 traffic density zones can be generated with
```
python3 src/traffic_gen.py \
  --intersection_file_path intersection_configs/4-approach-1-lane.json \
  --output_dir gen_data/4-approach-1-lane/n10-d0.7 \
  --num 100 \
  --vehicle_num 10 \
  --poisson_parameter_list '[0.7]'  
```

## Intersection Manager Training and Validation

Use the training script `src/PPO_jssp_multiInstances.py` to run the PPO training of the intersection manager. For instance, you can run a basic training with default arguments using following command,
```
cd src
python3 PPO_jssp_multiInstances.py --exp_name [name]
```
This will launch a PPO training with 10 jobs (vehicles), 4 machines (conflict zones), and 0.5 traffic density. Training logs and model checkpoints will be saved in `./loggings/[name]` and `./checkpoints/[name]` respectively. Other experiment-related arguments are listed below.

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `log_dir` | str | directory to store training logs (default: `loggings`) | 
| `ckpt_dir` | str | directory to store model checkpoints (default: `checkpoints`) |
| `ckpt_path` | str or None | if specified, will continue training with this checkpoint (defalt: None) |
| `valid_only` | bool | set to True if only validation process is required (default: False) |
| `valid_dir` | str | directory where validation data is stored (default: `testdata/4-approach-1-lane-n10`) |
| `train_density` | float | traffic density for training data generation (default: 0.5) |
| `intersection_config` | str | intersection config for training data generation (default: `intersection_configs/4-approach-1-lane.json`) |
| `num_vehicles` | int | number of vehicle for training data generation (default: 10) |
| `group_strat` | str | set to `jenks` to utilize Jenk-Breaks-based grouping strategy (default: `base`) |

For other training- or model-specific arguments, run `python3 PPO_jssp_multiInstances.py --help` for detail information.

## Baseline and Optimal Solutions

Generate baseline (FCFS, iGreedy) and optimal solutions with `src/baseline.py` script. For example,
```
python3 src/baseline.py \
  --data_dir gen_data/4-approach-1-lane/n10-d0.7 \
  --intersection_config intersection_configs/4-approach-1-lane.json
```

## Special Thanks

* Part of the code are ported from https://github.com/zcaicaros/L2D for implementation of applying RL on solving JSSP.
* Thanks the support from the two supervisors of this research project, Professor Chung-Wei Lin (National Taiwan University) and Professor Muhammed O. Sayin (Bilkent University). 
