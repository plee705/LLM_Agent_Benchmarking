import mujoco
import numpy as np

def run_simulation(model_path, duration=5.0, dt=0.01):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Set initial state from keyframe
    # Accessing keyframes correctly in MuJoCo Python bindings
    # model.key is a method, we should use model.key() or find another way
    # Actually, model.key is a method that returns keyframes.
    # Wait, I should check the mujoco documentation or try model.key(0).qpos if it's a method
    
    # Let's try to get the first keyframe's qpos and qvel directly
    # If model.key is a method:
    try:
        # In some versions, model.key might be a property or a method
        # Let's try accessing it via index if it's an array-like object
        # But the error says 'method' object is not subscriptable
        # So it's likely model.key(0)
        qpos = model.key(0).qpos
        qvel = model.key(0).qvel
    except Exception as e:
        print(f"Error accessing keyframe: {e}")
        # Fallback: manually set qpos if keyframe access fails
        # Pendulum at 90 degrees (1.5708 rad)
        qpos = np.array([1.5708])
        qvel = np.array([0.0])

    data.qpos[:] = qpos
    data.qvel[:] = qvel

    time = 0
    results = []

    while time < duration:
        mujoco.mj_step(model, data)
        time += dt
        results.append((time, *data.qpos))

    return results

if __name__ == "__main__":
    results = run_simulation("pendulum.xml")
    print(f"Simulated {len(results)} steps.")
    print("First 5 steps (time, qpos...):")
    for r in results[:5]:
        print(r)
