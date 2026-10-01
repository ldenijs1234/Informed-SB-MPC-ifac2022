import math

t = 0
dt = 1
rate_var = 1.0
dsbmpc_pred_hor = 600
dsbmpc_t_step = 5
ship_ids = ['ship_1', 'ship_2']

T_sim = 500

ship1_trajectory = [[50.0, -2500.0, 8.0], [50.0, 2500.0, 8.0]]
psi1 = math.atan2(ship1_trajectory[1][1] - ship1_trajectory[0][1], ship1_trajectory[1][0] - ship1_trajectory[0][0])
ship1_init_states = [0, 'ship_1', ship1_trajectory[0][0], ship1_trajectory[0][1], psi1, ship1_trajectory[0][2], 0.0, 0.0, "PDV", ship1_trajectory]

ship2_trajectory = [[50.0, -1200.0, 4.0], [50.0, 2500.0, 4.0]]
psi2 = math.atan2(ship2_trajectory[1][1] - ship2_trajectory[0][1], ship2_trajectory[1][0] - ship2_trajectory[0][0])
ship2_init_states = [0, 'ship_2', ship2_trajectory[0][0], ship2_trajectory[0][1], psi2, ship2_trajectory[0][2], 0.0, 0.0, "PDV", ship2_trajectory]

all_states = {'ship_1': ship1_init_states, 'ship_2': ship2_init_states}
