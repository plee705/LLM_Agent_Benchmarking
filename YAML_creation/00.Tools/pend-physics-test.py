import time
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import pandas as pd

# =================================================
# --------------- PASSING CRITERIA ----------------
# =================================================

GRAVITY = -9.81 # m/s^2
G_MAG = abs(GRAVITY) # m/s^2
ROD = 0.2485 # m
MASS = 1.0 # kg
INERTIA = MASS * (ROD ** 2) # kg*m^2

N_SECONDS = 1.25 # seconds

TOL = 0.001 # 0.1 % Tolerance Criterion

def gravity_check(model):
    g = model.opt.gravity[1] # Assuming gravity is along the y-axis

    dir = False
    if g < 0:
        dir = True

    mag = False
    if abs((g - GRAVITY) / GRAVITY) <= TOL:
        mag = True

    return dir and mag

def mass_check(model):
    modelmass = np.sum(model.body_mass) # Add up total mass of the model

    mbool = False
    if abs((modelmass - MASS) / MASS) <= TOL:
        mbool = True

    return mbool
    
def inertia_check(model):
    # inertia = np.sum(model.body_inertia)    # Add up total inertia of the model
    inertia = model.body_inertia
    
    print(f"Calculated Inertia: {inertia}, Expected Inertia: {INERTIA}")
    ibool = False
    # if abs((inertia - INERTIA) / INERTIA) <= TOL:
    #     ibool = True

    return ibool

def run_physics_checks(model):
    gcheck = gravity_check(model)
    mcheck = mass_check(model)
    icheck = inertia_check(model)

    if gcheck and mcheck and icheck:
        return True
    
    return False

# =================================================
# ------------------ SIMULATIONS ------------------
# =================================================

def build_model_from_xml(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)

    return model, data

def simulate(model, data, n_seconds=2, timestep=0.001):

    mujoco.mj_resetData(model, data)
    
    n_steps = int(n_seconds / timestep)

    sim_time = np.zeros(int(n_steps))
    potential_e = np.zeros(int(n_steps))
    kinetic_e = np.zeros(int(n_steps))
    angle = np.zeros(int(n_steps))
    omega = np.zeros(int(n_steps))


    for i in range(int(n_steps)):

        mujoco.mj_step(model, data)
        sim_time[i] = i * timestep
        angle[i] = data.qpos[0]
        omega[i] = data.qvel[0]
        potential_e[i] = data.energy[0]
        kinetic_e[i] = data.energy[1]

    return sim_time, angle, potential_e, kinetic_e, omega

def get_oscillation_freq_from_KE(sim_time, kinetic_e):
    peaks, _ = find_peaks(kinetic_e)
    
    peak_times = sim_time[peaks]

    subperiods = np.diff(peak_times)
    mean_period = 2 * np.mean(subperiods)   # Pendulum reaches peak twice in one period

    if mean_period <= 0:
        raise ValueError("Mean period is non-positive, check the simulation data.")
    else:
        frequency = 1 / mean_period
    
    print(f"Oscillation Frequency: {frequency:.4f} Hz")


# =================================================
# --------------------- ODEs ----------------------
# =================================================

def ode_pendulum(t, y, g, l):
    theta, omega = y
    theta_dot = omega
    omega_dot = -(g/l)*np.sin(theta)
    return [theta_dot, omega_dot]

def ode_positive_to_neg_velocity(t, y, g, l):
    theta, omega = y
    return omega

ode_positive_to_neg_velocity.terminal = False
ode_positive_to_neg_velocity.direction = -1

def ode_large_angle_period(theta0, omega0, g=G_MAG, l=ROD):
    y0 = [theta0, omega0]

    sol = solve_ivp(
        ode_pendulum, 
        t_span=(0, N_SECONDS),
        y0=y0,
        args=(g, l),
        events=ode_positive_to_neg_velocity,
        max_step=0.001,
        rtol=1e-10,
        atol=1e-12
    )

    crossing_times = sol.t_events[0]
    crossing_times = crossing_times[crossing_times > 0] # Ensures it doesn't count first step

    if len(crossing_times) < 1:
        raise ValueError("No full-period velocity crossing detected.")
    
    period = 2*crossing_times[0]
    frequency = 1.0 / period

    return frequency


# =================================================
# --------------- PLOTTIE THOTTIES ----------------
# =================================================

def plot_angle_vs_time(sim_time, angle, output_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(sim_time, angle, label='Angle', color='blue')
    plt.title('Angle vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/angle_vs_time.png')

def plot_energy_fluctuations(sim_time, potential_e, kinetic_e, output_dir: str):
    total_e = potential_e + kinetic_e

    plt.figure(figsize=(10, 6))
    plt.plot(sim_time, total_e, label='Total Energy', color='green', linestyle='--')
    plt.title('Energy Fluctuations Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/energy_fluctuations.png')

def plot_energy_v_time(sim_time, potential_e, kinetic_e, output_dir: str):
    total_e = potential_e + kinetic_e

    plt.figure(figsize=(10, 6))
    plt.plot(sim_time, potential_e, label='Potential Energy', color='blue')
    plt.plot(sim_time, kinetic_e, label='Kinetic Energy', color='orange')
    plt.plot(sim_time, total_e, label='Total Energy', color='green', linestyle='--')
    plt.title('Energy vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/energy_vs_time.png')

def plot_energy_v_angle(angle, potential_e, kinetic_e, output_dir: str):
    total_e = potential_e + kinetic_e

    plt.figure(figsize=(10, 6))
    plt.plot(angle, potential_e, label='Potential Energy', color='blue')
    plt.plot(angle, kinetic_e, label='Kinetic Energy', color='orange')
    plt.plot(angle, total_e, label='Total Energy', color='green', linestyle='--')
    plt.title('Energy vs Angle')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/energy_vs_angle.png')

def plot_velocity_v_time(sim_time, omega, output_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(sim_time, omega, label='Angular Velocity', color='purple')
    plt.title('Angular Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/angular_velocity_vs_time.png')

def plot_velocity_v_angle(angle, omega, output_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(angle, omega, label='Angular Velocity', color='red')
    plt.title('Angular Velocity vs Angle')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/angular_velocity_vs_angle.png')


# =================================================
# -------------------- OUTPUT ---------------------
# =================================================








# =================================================
# --------------------- MAIN ----------------------
# =================================================

def main():
    xml_path = "/home/ginger24/summer26/agentic-mujoco-master/llm_prompt_testing/YAML_creation/Temperature_Testing/V2_Temp_0.1/test06/pendulum.xml"

    output_directory = "/home/ginger24/summer26/agentic-mujoco-master/llm_prompt_testing/YAML_creation/Temperature_Testing/sandbox/V2_Temp_0.1"

    model, data = build_model_from_xml(xml_path)

    sim_time, angle, potential_e, kinetic_e, omega = simulate(model, data, n_seconds=N_SECONDS, timestep=0.001)

    plot_energy_v_time(sim_time, potential_e, kinetic_e, output_directory)
    plot_energy_v_angle(angle, potential_e, kinetic_e, output_directory)
    plot_energy_fluctuations(sim_time, potential_e, kinetic_e, output_directory)
    plot_angle_vs_time(sim_time, angle, output_directory)
    plot_velocity_v_time(sim_time, omega, output_directory)
    plot_velocity_v_angle(angle, omega, output_directory)

    get_oscillation_freq_from_KE(sim_time, kinetic_e)

    ode_freq = ode_large_angle_period(theta0=angle[0], omega0=omega[0], g=G_MAG, l=ROD)
    print(f"Oscillation Frequency from ODE: {ode_freq:.4f} Hz")

    print(f"Gravity Check: {gravity_check(model)}")
    print(f"Mass Check: {mass_check(model)}")
    print(f"Inertia Check: {inertia_check(model)}")

if __name__ == "__main__":
    main()