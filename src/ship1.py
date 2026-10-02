#!/usr/bin/env python

from ftplib import all_errors
import json
import os
from queue import Empty
import rospy
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from informed_sbmpc.msg import ship_states, bcast_sitaw
from rospy.numpy_msg import numpy_msg
from ship_model import *
from utility import *
from map_polygons import *
from config import *
from sbmpc import *
from dsbmpc import *
from colav import *
from random import randint

def sitaw_callback(rx_data):
    global all_states
    all_states = json.loads(rx_data.data)


def response_direct_msg(req):
    """
        Direct messaging (ROS service server)
    """
    print("\n", req, "\n")
    # Filter the sender targetship
    ts = [ts for ts in ts_list if ts.id == req.sender][0]
    # Update last proposed control actions of targetship
    ts.ppsd_psi = req.sender_psi_u[0]
    ts.ppsd_u = req.sender_psi_u[1]
    
    # Prepare and send an answer for the request 
    header, os_psi, os_u, ts_psi, ts_u, msg_id = dsbmpc.answer_request(req, ownship, ts, ts_list)
    return DirectMessageResponse(header, ownship.id, req.sender, [os_psi, os_u], [ts_psi, ts_u], msg_id)


def direct_msg_response(ownship):
    """
        Direct messaging (ROS service server). Initialize direct messaging
    """
    s = rospy.Service(f'direct_msg_{ownship.id}', DirectMessage, response_direct_msg)
    return s



# Defining initial states and creating the ownship class [t, id, x, y, psi, U, colregs18, [trajectory]]
ownship = DynamicModel(ship1_init_states)
sbmpc = SBMPC()
dsbmpc = DSBMPC(ownship)
ts_id_list = []
ts_list = []
solve_times = []  # List to store solve times for each time step
history = []  # List to store the history of ownship states
all_states = {}
dist_log = np.zeros((2, T_sim))

# create the publisher and subscribe to the server
pub_states = rospy.Publisher('ship_state_topic', ship_states, queue_size=10)
rospy.Subscriber('bcast_states_topic', bcast_sitaw, sitaw_callback)
state_msg = ship_states()

# initialiaze the ship node
rospy.init_node('ship_node', anonymous=True)
rate = rospy.Rate(rate_var)

# At the top of ship1.py after rospy.init_node
active_case = rospy.get_param('~scenario_case', 'case01')
active_mode = rospy.get_param('~colav_mode', 'RA')  # 'RA' or 'IS'

# Tell the ship model whether route exchange is enabled
# For RA: ownship.route_exchange = False
# For IS: ownship.route_exchange = True
ownship.route_exchange = (active_mode == 'IS')

# starting direct message service server
s = direct_msg_response(ownship)

while not rospy.is_shutdown():
    if t > T_sim:
        print(f"[{ownship.id}] Reached T_sim ({T_sim}). Breaking simulation loop...")
        break

    # --- SYNCHRONIZATION LOCK ---
    # Ensure ship_2 is not lagging more than 1 step behind before calculating COLAV
    ship2_t = all_states['ship_2'][0] if 'ship_2' in all_states else 0
    if ship2_t < (t - 1):
        rate.sleep()
        continue  # Skip this loop iteration and wait for ship_2 to catch up
    # ----------------------------
    
    # publish own states [t, id, x, y, psi, U, colregs18, [trajectory]]
    publish_states(t, ownship, state_msg, pub_states)

    # printing sitaw data
    # print(all_states)
    
    # Creating target ships from received data
    ts_id_list, ts_list = create_ts_data(ts_id_list, ts_list, all_states, ownship, dt)

    # Run COLAV and record solver execution time
    step_solve_time_ms = colav(t, ownship, ts_list, sbmpc, dsbmpc, initial_reaction=True)
    solve_times.append(step_solve_time_ms)

    # ALWAY RECORD HISTORY (Fallback to all_states if out of detection range)
    ts_x, ts_y, ts_psi, ts_u = 0.0, 0.0, 0.0, 0.0
    if len(ts_list) > 0:
        ts_x = ts_list[0].x
        ts_y = ts_list[0].y
        ts_psi = ts_list[0].psi
        ts_u = ts_list[0].u
    elif 'ship_2' in all_states and len(all_states['ship_2']) > 7:
        # Fallback to pure situation awareness coordinates
        ts_x = all_states['ship_2'][2]
        ts_y = all_states['ship_2'][3]
        ts_psi = all_states['ship_2'][4]
        ts_u = all_states['ship_2'][5]
        
    history.append({
        'time': t,
        'ship_1_x': ownship.x,
        'ship_1_y': ownship.y,
        'ship_1_psi': ownship.psi,
        'ship_1_u': ownship.u,
        'ship_2_x': ts_x,
        'ship_2_y': ts_y,
        'ship_2_psi': ts_psi,
        'ship_2_u': ts_u,
        'solve_time_ms': step_solve_time_ms
    })

    # move ship
    ownship.move(dt)

    # Find best offset course and speed values
    # colav(t, ownship, ts_list, sbmpc, dsbmpc, initial_reaction=True)
    #ownship.set_opt_ctrl(0, 1)

    # Distance log for graph
    for i in range(len(ts_list)):
        dist_log[i, t-1] = distance(ownship.x, ownship.y, ts_list[i].x, ts_list[i].y)

    rate.sleep()
    t = t+1

print(f"DEBUG: Loop ended. Total history records captured: {len(history)}")
# Wait for manual termination/shutdown
#s.spin()

# --- Post-Simulation Export Block ---
if len(history) > 0:
    # 2. Build the DataFrame
    df = pd.DataFrame(history)

    # 3. Calculate Euclidean separation distance
    df['separation_dist_m'] = np.sqrt(
        (df['ship_1_x'] - df['ship_2_x'])**2 + 
        (df['ship_1_y'] - df['ship_2_y'])**2
    )

    # 4. Calculate rate of change of heading (steering rate in deg/s)
    delta_psi = np.diff(df['ship_1_psi'].values, prepend=df['ship_1_psi'].values[0])
    delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi
    df['steering_rate_deg_s'] = np.abs(np.degrees(delta_psi) / dt)

    # 5. Extract Key Validation Metrics
    min_dist = df['separation_dist_m'].min()
    max_steering_rate = df['steering_rate_deg_s'].max()
    active_solves = df[df['solve_time_ms'] > 0]['solve_time_ms']
    mean_solve_time = active_solves.mean() if len(active_solves) > 0 else 0.0
    peak_solve_time = df['solve_time_ms'].max()

    # 6. Save to CSV
    results_dir = os.path.join(os.path.dirname(__file__), 'sim_results')
    os.makedirs(results_dir, exist_ok=True)
    # Unique filenames per case and mode
    csv_path = os.path.join(results_dir, f"{active_case}_{active_mode}_metrics.csv")
    # plot_path = os.path.join(results_dir, f"{active_case}_{active_mode}_trajectory.png")
    
    df.to_csv(csv_path, index=False)
    
    # Save the trajectory plot
    # If using matplotlib inside ship1.py or server.py:
    # plt.savefig(plot_path, dpi=300)
    print(f"[{active_case} - {active_mode}] Saved metrics to {csv_path}")

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Data saved to: {csv_path}")
    print(f"Minimum Distance to Target Ship: {min_dist:.2f} m")
    print(f"Maximum Steering Rate:          {max_steering_rate:.2f} deg/s")
    print(f"Average Active Solve Time:      {mean_solve_time:.2f} ms")
    print(f"Peak Cycle Solve Time:          {peak_solve_time:.2f} ms")
    print("===================================================\n")

#############################
# PAPER PLOT (FIG. a / FIG. b)
#############################
try:
    fig_paper, ax = plt.subplots(figsize=(6, 9))
        
    # Plot physical paths directly from our safe dataframe
    ax.plot(df['ship_1_x'], df['ship_1_y'], 'b-', label='OS (ship_1)', linewidth=1.5)
    ax.plot(df['ship_2_x'], df['ship_2_y'], 'r-', label='TS (ship_2)', linewidth=1.5)
        
    # Mark endpoints
    ax.plot(df['ship_1_x'].iloc[-1], df['ship_1_y'].iloc[-1], 'bo', mfc='none', markersize=8)
    ax.plot(df['ship_2_x'].iloc[-1], df['ship_2_y'].iloc[-1], 'ro', mfc='none', markersize=8)
        
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"{active_case} Trajectory ({active_mode})")
    ax.set_xlabel("East (X) [m]")
    ax.set_ylabel("North (Y) [m]")
    ax.legend()

    plot_filename = f"{active_case}_{active_mode}_trajectory.png"
    output_file = os.path.join(results_dir, plot_filename)
    fig_paper.savefig(output_file, dpi=300)
    plt.close(fig_paper)
    print(f"\n[SUCCESS] Saved trajectory plot to: {output_file}")
except Exception as e:
    print(f"\n[ERROR] Plotting failed: {e}")

rospy.signal_shutdown("Simulation and plotting finished successfully.")

#############################
# VISUALIZATION
#############################
df_anim, anim_length = create_animation_data(ownship, ts_list)
print(df_anim)

ship_markers = []
past_trajectory = []
ship_wps = []
headings = []
speeds = []
colors = ['blue', 'purple', 'darkolivegreen', 'teal', 'darkorange', 'saddlebrown']
fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(2, 3)
ax1 = fig.add_subplot(gs[:, 0:2])
ax1.grid()
plt.xlim(-2500.0, 2500.0)
plt.ylim(-2500.0, 2500.0)
# Drawing map polygons:
for geom in poly_full.geoms:
    xs, ys = geom.exterior.xy
    ax1.fill(xs, ys, c='gray', alpha=0.8, fc='wheat')
ax2 = fig.add_subplot(gs[0, 2])
ax2.grid()
plt.title("Ship headings over time")
plt.xlabel("Time")
plt.ylabel("Heading [Radians]")
plt.xlim(0, T_sim)
plt.ylim(-4, 4)
ax3 = fig.add_subplot(gs[1, 2])
ax3.grid()
plt.title("Ship speed changes over time")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.xlim(0, T_sim)
plt.ylim(-20, 20)

for j in range(len(ts_list) + 1):
    ship_markers.append(ax1.plot([], [], 'o', mfc='none', markersize=15, c=colors[j])[0])
    past_trajectory.append(ax1.plot([], [], c=colors[j], alpha=0.8)[0])
    ship_wps.append(ax1.plot([], [], "--", c=colors[j], alpha=0.5)[0])
    headings.append(ax2.plot([], [], c=colors[j])[0])
    speeds.append(ax3.plot([], [], c=colors[j])[0])

def init_ani():
    pass

def animate(i):
    for n in range(len(ship_markers)):
        x_vals = df_anim[f"ship_{n+1}_x"].loc[i]
        y_vals = df_anim[f"ship_{n+1}_y"].loc[i]
        ship_markers[n].set_xdata(x_vals)
        ship_markers[n].set_ydata(y_vals)
        past_trajectory[n].set_xdata(df_anim[f"ship_{n+1}_x"][:i])
        past_trajectory[n].set_ydata(df_anim[f"ship_{n+1}_y"][:i])
        try:
            wps = df_anim[f"ship_{n+1}_wp"].loc[i]
            wp_x_vals = [wp[0] for wp in wps]
            wp_y_vals = [wp[1] for wp in wps]
            ship_wps[n].set_xdata(wp_x_vals)
            ship_wps[n].set_ydata(wp_y_vals)
        except:
            ship_wps[n].set_xdata([])
            ship_wps[n].set_ydata([])
        headings[n].set_xdata(df_anim['time'][:i])
        headings[n].set_ydata(df_anim[f"ship_{n + 1}_psi"][:i])
        speeds[n].set_xdata(df_anim['time'][:i])
        speeds[n].set_ydata(df_anim[f"ship_{n + 1}_u"][:i])

ani = animation.FuncAnimation(fig, animate, init_func=init_ani,
                              frames=anim_length, interval=50, blit=False)
sim_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results")
os.makedirs(sim_results_dir, exist_ok=True)
fig_paper.savefig(os.path.join(sim_results_dir, "animation.png"))
plt.close(fig_paper)


"""
# Heading graph
plt.figure(figsize=(7, 7))
for j in range(len(ts_list) + 1):
    plt.plot(df_anim['time'][:], df_anim[f"ship_{j + 1}_psi"], c=colors[j])
plt.grid()
plt.ylim(-4, 4)
plt.title("Ship headings over time")
plt.xlabel("Time")
plt.ylabel("Heading [Radians]")
plt.savefig(os.path.join(sim_results_dir, 'Headings.png'), dpi=300)

# Speed graph
plt.figure(figsize=(7, 7))
for j in range(len(ts_list) + 1):
    plt.plot(df_anim['time'][:], df_anim[f"ship_{j + 1}_u"], c=colors[j])
plt.grid()
plt.ylim(-5, 20)
plt.title("Speed changes over time")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.savefig(os.path.join(sim_results_dir, 'Speeds.png'), dpi=300)

# Distance graph
plt.figure(figsize=(7, 7))
for j in range(len(ts_list)):
    plt.plot(df_anim['time'][:], dist_log[j, :], c=colors[j+1])
#plt.axhline(y=300, color='k', linestyle='--')
plt.grid()
plt.title('Distance between ships over time')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.savefig(os.path.join(sim_results_dir, 'Distance.png'), dpi=300)
"""