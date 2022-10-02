# 2-D kinematics equations.

- Detailed derivation of the kinematics equations is [here](2dkinematics.pdf).
- The final equation is given as here.
- 
  $$
  0 = v \cdot cos(\psi + tan^{-1}(\frac{l_r}{l} tan(\delta))) - \dot{x(t)}  \\
  0 = v \cdot sin(\psi + tan^{-1}(\frac{l_r}{l} tan(\delta))) - \dot{y(t)}  \\
  0 = \frac{v}{l} \cdot cos(\beta) \cdot tan(\delta) - \dot{\psi(t)} 
  $$

 Given the $\dot{x(t)}, \dot{y(t)}, \dot{\psi(t)}$ and above equation, we can solve for the velocity ($v$) and steering angle ($\delta$).



 # In action. 


 ## Line follower robot controller using com velocity using PPO.


https://user-images.githubusercontent.com/34353557/179654395-5e33acfe-4c98-4276-8e11-ed2f4bb82b56.mp4



 ## Obstacle avoidance.

https://user-images.githubusercontent.com/34353557/179654498-de3be2b6-272b-43c2-955d-8af4689d759b.mp4



## Notes:

### old model.
- peanlize yaw. and maximize the speed .
$v$ -> maximize v.
$\omega$ -> mimize

### new model.


