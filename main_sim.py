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
	command = Command()
	pupper = PupperBase("simulation/assets/pupper_out.xml")
	state.quat_orientation = np.array([1, 0, 0, 0])

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


	#sys.exit()
	for _ in range(100):
		now = pupper.time
		if now - last_loop < config.dt:
			#pupper.render()
			pupper.sim.step()
			continue
		last_loop = pupper.time
		# time.sleep(0.1)
		# if last_loop - start_time > 5 and start == True:
		# 	state.behavior_state = BehaviorState.TROT
		# 	start = False

		#print(state.behavior_state)
		controller.run(state, command)

		store_data = np.vstack((store_data, state.joint_angles.flatten('F')))

		pupper.set_joint_angles(state.joint_angles)
		pupper.render()

	print(store_data.shape)
	plt.plot(store_data)
	plt.legend(pupper.model.actuator_names)
	plt.savefig("simulation/joint_angle.png")
	np.savetxt("simulation/joint_angles.csv", store_data)
	plt.show()



if __name__ == "__main__":
	main()






