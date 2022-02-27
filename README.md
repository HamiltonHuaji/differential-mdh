# Differentiable Modified Denavit–Hartenberg (MDH) Model (for Calibration of Robotic Arms)

Inspired by [mdh](https://github.com/MultipedRobotics/dh)

<img src="https://upload.wikimedia.org/wikipedia/commons/d/d8/DHParameter.png" width="600px">

[Modified Denavit-Hartenberg parameters](https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters)

## Example

```python
import mdh, mdh.robots
import torch as tf

param = mdh.KinematicChain.obj_to_tensor(mdh.robots.kuka_lbr_iiwa_7(), requires_grad=True)
chain = mdh.KinematicChain.from_tensor(param)
print(chain)

thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
T = chain.transform(thetas)

p = tf.tensor([0., 0., 0., 1.], dtype=tf.float64)
uncalibrated_end_effector = T@p
calibrated_end_effector = tf.tensor([1., -.2, 0., 1.], dtype=tf.float64)

r = uncalibrated_end_effector - calibrated_end_effector
loss = r@r

loss.backward()
print(param.grad.data)
```

## TODO

+ [ ] inverse kinematics
