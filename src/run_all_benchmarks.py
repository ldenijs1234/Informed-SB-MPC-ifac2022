#!/usr/bin/env python3
import subprocess
import time
import os
import sys

# Define all 6 scenarios in Akdağ's native scale
CASES = {
    'case01': {
        'T_sim': 400,
        'ship1': [[0.0, -2000.0, 8.0], [0.0, 2500.0, 8.0]],
        'ship2': [[0.0, 2000.0, 8.0], [0.0, -2000.0, 8.0]]
    },
    'case02': {
        'T_sim': 400,
        'ship1': [[0.0, -2000.0, 8.0], [0.0, 2500.0, 8.0]],
        'ship2': [[-100.0, 2000.0, 8.0], [-100.0, 500.0, 8.0], [1000.0, -2000.0, 8.0]]
    },
    'case03': {
        'T_sim': 400,
        'ship1': [[0.0, -2000.0, 8.0], [0.0, 2000.0, 8.0]],
        'ship2': [[2000.0, 0.0, 8.0], [0.0, 0.0, 8.0], [0.0, 2000.0, 8.0]]
    },
    'case04': {
        'T_sim': 450,
        'ship1': [[-1500.0, -1500.0, 8.0], [-500.0, -500.0, 8.0], [500.0, 500.0, 8.0], [1500.0, 1500.0, 8.0]],
        'ship2': [[1500.0, -500.0, 8.0], [500.0, 0.0, 8.0], [-500.0, 1000.0, 8.0]]
    },
    'case05': {
        'T_sim': 400,
        'ship1': [[200.0, -2000.0, 8.0], [200.0, 2000.0, 8.0]],
        'ship2': [[-200.0, 2000.0, 8.0], [-200.0, 200.0, 8.0], [1500.0, -200.0, 8.0]]
    },
    'case06': {
        'T_sim': 500,
        'ship1': [[50.0, -2500.0, 8.0], [50.0, 2500.0, 8.0]],
        'ship2': [[50.0, -1200.0, 4.0], [50.0, 2500.0, 4.0]]
    }
}

MODES = ['RA', 'IS']
CONFIG_PATH = "/root/catkin_ws/src/informed_sbmpc/src/config.py"

def write_config(case_name, case_data):
    """Dynamically overwrite config.py for the target case."""
    content = f"""import math

t = 0
dt = 1
rate_var = 1.0
dsbmpc_pred_hor = 600
dsbmpc_t_step = 5
ship_ids = ['ship_1', 'ship_2']

T_sim = {case_data['T_sim']}

ship1_trajectory = {case_data['ship1']}
psi1 = math.atan2(ship1_trajectory[1][1] - ship1_trajectory[0][1], ship1_trajectory[1][0] - ship1_trajectory[0][0])
ship1_init_states = [0, 'ship_1', ship1_trajectory[0][0], ship1_trajectory[0][1], psi1, ship1_trajectory[0][2], 0.0, 0.0, "PDV", ship1_trajectory]

ship2_trajectory = {case_data['ship2']}
psi2 = math.atan2(ship2_trajectory[1][1] - ship2_trajectory[0][1], ship2_trajectory[1][0] - ship2_trajectory[0][0])
ship2_init_states = [0, 'ship_2', ship2_trajectory[0][0], ship2_trajectory[0][1], psi2, ship2_trajectory[0][2], 0.0, 0.0, "PDV", ship2_trajectory]

all_states = {{'ship_1': ship1_init_states, 'ship_2': ship2_init_states}}
"""
    with open(CONFIG_PATH, "w") as f:
        f.write(content)

def kill_stale_ros():
    """Ensure no background nodes or roscore lock files hang between runs."""
    subprocess.run(["pkill", "-9", "-f", "ros"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def main():
    total_runs = len(CASES) * len(MODES)
    run_idx = 1

    for case_name, case_data in CASES.items():
        for mode in MODES:
            print(f"\n==================================================")
            print(f"[{run_idx}/{total_runs}] STARTING: {case_name.upper()} | MODE: {mode}")
            print(f"==================================================")

            # 1. Update config file with active case waypoints
            write_config(case_name, case_data)

            # 2. Launch the scenario
            cmd = [
                "roslaunch", "informed_sbmpc", "batch_sim.launch",
                f"scenario_case:={case_name}",
                f"colav_mode:={mode}"
            ]
            
            proc = subprocess.Popen(cmd)
            proc.wait()  # Wait until ship1 finishes T_sim and exits cleanly

            # 3. Clean up processes before starting next case
            kill_stale_ros()
            run_idx += 1
            time.sleep(1)

    print("\nAll 12 simulation experiments finished successfully!")
    print("Files saved in: /root/catkin_ws/src/informed_sbmpc/src/sim_results/")

if __name__ == "__main__":
    main()