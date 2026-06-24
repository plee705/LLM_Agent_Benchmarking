import time
import imageio.v3 as iio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

def camera_setup(azimuth, elevation, distance, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat = lookat

    return cam

def build_model_from_xml(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    return model, data

def save_single_render(
        model,
        data,
        output_file,
        height=480,
        width=640,
):
    with mujoco.Renderer(model, height, width) as renderer:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)

        pixels = renderer.render()
        iio.imwrite(output_file, pixels)

    print(f"Saved render to {output_file}")

def simulate_and_record(
    model,
    data,
    cam=None,
    n_seconds=5,
    framerate=30,
    height=480,
    width=640,
):
    n_frames = int(n_seconds * framerate)
    frames = []

    mujoco.mj_resetData(model, data)
    
    sim_time = 0.0
    render_time = 0.0
    n_steps = 0

    with mujoco.Renderer(model, height, width) as renderer:
        for i in range(n_frames):
            while data.time * framerate < i:
                tic = time.time()
                mujoco.mj_step(model, data)
                sim_time += time.time() - tic
                n_steps += 1

            tic = time.time()

            renderer.update_scene(data, camera=cam)

            scene = renderer.scene
            scene.lights[0].ambient[:] = [0.5, 0.5, 0.5] # Color

            frame = renderer.render()
            render_time += time.time() - tic

            frames.append(frame)

    print(f"Simulation time: {sim_time:.2f} seconds")
    print(f"Render time: {render_time:.2f} seconds")
    print(f"Number of steps: {n_steps}")

    return frames


def save_video(frames, output_file, framerate=30):
    iio.imwrite(output_file, frames, fps=framerate)
    print(f"Saved video to {output_file}")


# =================================================
# --------------------- MAIN ----------------------
# =================================================

def main():
    xml_path = "/home/ginger24/summer26/agentic-mujoco-master/llm_prompt_testing/YAML_creation/Temperature_Testing/V2_Temp_0.1/test01/pendulum.xml"  # Replace with your XML file path
    output_image_file = "/home/ginger24/summer26/agentic-mujoco-master/llm_prompt_testing/YAML_creation/Temperature_Testing/sandbox/V2_Temp_0.1/pendulum_image.png"
    output_video_file = "/home/ginger24/summer26/agentic-mujoco-master/llm_prompt_testing/YAML_creation/Temperature_Testing/sandbox/V2_Temp_0.1/pendulum_video.mp4"

    model, data = build_model_from_xml(xml_path)
    
    # Configure the Camera
    cam = camera_setup(
        azimuth=90,  # degrees
        elevation=-90,  # degrees
        distance=0.3,  # meters
        lookat=np.array([0.0, 0.8, 0.5]),  # Look at the origin
    )


    # Save a single render
    save_single_render(model, data, output_image_file)

    # Simulate and record video
    frames = simulate_and_record(model, data, cam=cam, n_seconds=5, framerate=30)
    save_video(frames, output_video_file, framerate=30)

if __name__ == "__main__":
    main()