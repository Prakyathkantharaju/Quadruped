import time
import sys, os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,"StanfordQuadruped")




from StanfordQuadruped.src.Controller import Controller
from StanfordQuadruped.src.State import State
from StanfordQuadruped.pupper.Kinematics import four_legs_inverse_kinematics
from StanfordQuadruped.pupper.Config import Configuration
from simulation.pupper_sim import PupperBase
from StanfordQuadruped.src.Command import Command
from StanfordQuadruped.src.State import BehaviorState



def main():
    """ Main program
    """
    config = Configuration()
    controller = Controller(config, four_legs_inverse_kinematics)
    state = State()
    state.quat_orientation = np.array([1, 0, 0, 0])
    command = Command()
    pupper = PupperBase("simulation/assets/pupper_out.xml")


    # fake pressing ps4 keys
    command.trot_event = True
    state.behavior_state = list(controller.trot_transition_mapping.keys())[1]


    # run the controller to get the joint angles.
    controller.run(state, command)


    # starting off with the trot gait.
    sim_state = pupper.get_state()

    ids = [pupper.sim.model.get_joint_qpos_addr(c) for c in pupper.sim.model.actuator_names]
    initial_start_positions = state.joint_angles.flatten('F')
    pupper.data.ctrl[:] = initial_start_positions


    # setting the angle in the state
    for i in range(len(ids)):
        sim_state.qpos[ids[i]] = initial_start_positions[i]
    pupper.sim.set_state(sim_state)
    previous_command = state.joint_angles.flatten('F')
    act_time = pupper.time
    for i in range(3000):
        pupper.sim.step()
        pupper.render()
        pupper.time - act_time
        if any(previous_command != state.joint_angles.flatten('F')) and i > 20:
            # print(act_time - pupper.time)
            act_time = pupper.time

            pupper.data.ctrl[:] = state.joint_angles.flatten('F')
            # print(state.joint_angles.flatten('F'))
            previous_command = state.joint_angles.flatten('F')
            if i > 100:
                command.horizontal_velocity = np.array([0.2, 0])
                error = 0 - pupper.sim.data.body_xpos[1][1]
                print('velcoity', pupper.sim.data.body_xvelp[2, :])
                print('position', pupper.sim.data.get_body_xpos("torso"))
                # print(pupper.sim.data.body_xpos[1])
                # print("error:", error)
                #command.yaw_rate = np.clip(error, a_min=-0.3, a_max=0.3)
                # command.yaw_rate = 0.1
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






