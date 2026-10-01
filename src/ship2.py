#!/usr/bin/env python

import rospy
import numpy as np
import json
from informed_sbmpc.msg import ship_states, bcast_sitaw
from rospy.numpy_msg import numpy_msg
from ship_model import *
from utility import *
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



# Defining initial states and creating the ownship class
ownship = DynamicModel(ship2_init_states)
sbmpc = SBMPC()
dsbmpc = DSBMPC(ownship)
ts_id_list = []
all_states = {}
ts_list = []

# create the publisher and subscribe to the server
pub_states = rospy.Publisher('ship_state_topic', ship_states, queue_size=10)
rospy.Subscriber('bcast_states_topic', bcast_sitaw, sitaw_callback)
state_msg = ship_states()

# initialiaze the node
rospy.init_node('ship_node', anonymous=True)
rate = rospy.Rate(rate_var)

# starting direct message service server
s = direct_msg_response(ownship)

last_t_seen = -1

while not rospy.is_shutdown() and t <= T_sim:
    if t >= T_sim:
        rospy.signal_shutdown("Simulation reached T_sim")
        break
    # Wait until ship_1 has published its state for this step
    if 'ship_1' in all_states:
        ship1_t = all_states['ship_1'][0]
        
        # Only step when ship_1 has reached or exceeded our current time step
        if ship1_t >= t and ship1_t != last_t_seen:
            last_t_seen = ship1_t

            # publish own states [t, id, x, y, psi, U, colregs18, [trajectory]]
            publish_states(t, ownship, state_msg, pub_states)

            # Creating target ships from received data
            ts_id_list, ts_list = create_ts_data(ts_id_list, ts_list, all_states, ownship, dt)

            # move ship
            ownship.move(dt)
            ownship.set_opt_ctrl(0, 1)

            t = t + 1

    rate.sleep()

# Keep node alive until shutdown
rospy.spin()