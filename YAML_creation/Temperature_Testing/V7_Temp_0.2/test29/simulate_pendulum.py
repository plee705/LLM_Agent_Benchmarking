import mujoco
import numpy as np

# Load the model
model = mujoco.MjModel.from_xml_path('pendulum.xml')
data = mujoco.MjData(model)

# Set initial conditions: Pendulum starts horizontal
# Initial angle: +90 degrees from vertical
# Vertical is along +y (gravity is 0 -9.81 0).
# If the joint angle is 0, the bob is at (0.2485, 0, 0) relative to hinge.
# This is horizontal.
# So joint angle 0 is 90 degrees from vertical.

# Let's set it explicitly to pi/2 if we want to be sure, 
# but since 0 is already horizontal, let's just verify.
print(f"Initial joint angle: {data.qpos[0]}")

# The user asked for: "Initial angle: +90 degrees from vertical"
# and "Bob positioned along +x from hinge".
# If the joint angle is 0, it's along +x.

# We'll simulate for a few steps to see if it moves.
for _ in range(100):
    mujoco.mj_step(model, data)

print(f"Successfully simulated 100 steps.")
print(f"Final joint angle: {data.qpos[0]}")
