import numpy as np
import mujoco as mj
from mujoco_py import load_model_from_path, MjSim, MjViewer
# from StanfordQuadruped.src.State import State
from dm_control import mujoco

class PupperBase:
	render_modes = {"human", "rgb_array"}
	def __init__(self, model_path: str):
		self.model = load_model_from_path(model_path)
		self.sim = MjSim(self.model)
		self.data = self.sim.data
		self.viewer = MjViewer(self.sim)
		self.sim.step()
		# information about the model
		if True:
			print(len(self.sim.data.ctrl))
			print(dir(self.sim.data))
			# print(self.model.actuator_names)
			print([print(c)  for c in dir(self.model) if 'quat' in c ])
			print(self.model.body_names)

	def set_joint_angles(self, action: np.ndarray):
		action = action.flatten('F')
		self.sim.data.ctrl[:] = action
		self.sim.step()


	def get_quat(self):
		main_body_quat = self.sim.data.body_xquat[1]

	@property
	def time(self):
		return self.sim.data.time





	def render(self, mode: str = "human"):
		if mode == "human":
			self.viewer.render()
		elif mode == "rgb_array":
			return self.viewer.get_image()
		else:
			raise ValueError(f"Unknown render mode: {mode}")



if __name__ == "__main__":
	pupper = PupperBase("assets/pupper_out.xml")
	# physics_engine.render()


	for i in range(100):
		pupper.get_state()
		pupper.render()
