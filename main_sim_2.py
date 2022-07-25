import time
import sys, os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,"StanfordQuadruped")


import math

from StanfordQuadruped.src.Controller import Controller
from StanfordQuadruped.src.State import State
from StanfordQuadruped.pupper.Kinematics import four_legs_inverse_kinematics
from StanfordQuadruped.pupper.Config import Configuration
from simulation.pupper_sim_2 import QuadEnv
from StanfordQuadruped.src.Command import Command
from StanfordQuadruped.src.State import BehaviorState


def euler_from_quaternion(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x,y,z,w = quat[0], quat[1], quat[2], quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians

def main():
    """ Main program
    """
    config = Configuration()
    controller = Controller(config, four_legs_inverse_kinematics)
    state = State()
    state.quat_orientation = np.array([1, 0, 0, 0])
    command = Command()
    path  = os.getcwd() + "/simulation/assets/pupper_out.xml"
    pupper = QuadEnv(path)

    pupper.reset()

    # fake pressing ps4 keys
    command.trot_event = True
    state.behavior_state = list(controller.trot_transition_mapping.keys())[1]


    # run the controller to get the joint angles.
    controller.run(state, command)


    # starting off with the trot gait.
    sim_state = pupper.get_state
    print(sim_state)

    ids = [pupper.sim.model.get_joint_qpos_addr(c) for c in pupper.sim.model.actuator_names]
    print(pupper.sim.model.actuator_names)
    initial_start_positions = state.joint_angles.flatten('F')
    print(initial_start_positions)
    
    pupper.data.ctrl[:] = initial_start_positions


    # setting the angle in the state
    for i in range(len(ids)):
        sim_state.qpos[ids[i]] = initial_start_positions[i]
    pupper.sim.set_state(sim_state)
    previous_command = state.joint_angles.flatten('F')
    act_time = pupper.data.time
    for i in range(3000):
        pupper.sim.step()
        pupper.render()
        pupper.data.time - act_time
        if any(previous_command != state.joint_angles.flatten('F')) and i > 20:
            # print(act_time - pupper.time)
            act_time = pupper.data.time

            pupper.data.ctrl[:] = state.joint_angles.flatten('F')
            # print(state.joint_angles.flatten('F'))
            previous_command = state.joint_angles.flatten('F')
            if i > 100:
                command.horizontal_velocity = np.array([0, 2])
                error = 0 - pupper.sim.data.body_xpos[1][1]
                # print('velcoity', pupper.sim.data.body_xvelp[2, :])
                # print('position', pupper.sim.data.get_body_xpos("torso"))
                angle = euler_from_quaternion(pupper.data.body_xquat[1])
                # print('angle', angle)   
                theta = angle[0] + math.pi 
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                global_vel = R @ pupper.sim.data.body_xvelp[1, :2].reshape(2, 1)
                print('global_vel', global_vel.flatten(), 'yaw', theta, 'position', pupper.sim.data.get_body_xpos("torso"))
                command.height = -0.2
                # command.height = -0.2

        controller.run(state, command)





    start_time = pupper.time
    start = True
    last_loop = pupper.time
    print("Summary of gait parameters:")
    print("overlap time: ", config.overlap_time)
    print("swing time: ", config.swing_time)
    print("z clearance: ", config.z_clearance)
    print("x shift: ", config.x_shift)
    print("dt:", config.dt)

    i = 0
    command.trot_event = True
    state.behavior_state = list(controller.trot_transition_mapping.keys())[0]
    store_data = state.joint_angles.flatten('F')
    plt.plot(store_data)
    plt.legend(pupper.model.actuator_names)
    plt.savefig("simulation/joint_angle.png")
    np.savetxt("simulation/joint_angles.csv", store_data)
    plt.show()
    # sys,exit()







if __name__ == "__main__":
    main()






