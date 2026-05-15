'''
    Purpose: Compute monostatic RCS for a thin dielectric balloon-like shell.

    Parameters from user note:
      - Outer radius:      1.5 m
      - Shell thickness:   0.3 mm
      - Shell epsilon_r:   4.0
      - Shell loss tangent:0.005
      - Frequency range:   2 GHz to 18 GHz

    Model used here:
      - Core: air
      - Shell: dielectric coating
      - Background: air

    Output:
      - output/balloon_shell_rcs.png
      - output/balloon_shell_rcs.txt
'''

from getRCS import *
import numpy as np


if __name__ == "__main__":
    print("------\nBALLOON SHELL MONOSTATIC RCS TEST\n------\n")

    outer_radius = 1.5003
    shell_width = 0.3e-3  # 0.3 mm
    core_radius = outer_radius - shell_width

    frequency = np.linspace(2e9, 18e9, 401)
    sensor_location = [0, 0, -2000]

    air = DielectricMaterial(1.0, 0.0, name="Air")
    shell_material = DielectricMaterial(
        4.0,
        0.0,
        1.0,
        0.0,
        name="BalloonShell",
        loss_tangent=0.005,
    )

    coated = CoatedSphere(air, shell_material, core_radius=core_radius, shell_width=shell_width)

    c0 = 299792458.0
    ratio = coated.radius * frequency / c0

    (freq, mono_RCS) = RCS_vs_freq_shell(
        coated,
        ratio,
        air,
        sensor_location,
        save_file="output/balloon_shell_rcs",
        show_plot=1,
    )

    print(f"RCS min:  {np.min(np.abs(mono_RCS)):.6e}")
    print(f"RCS max:  {np.max(np.abs(mono_RCS)):.6e}")
    print(f"RCS mean: {np.mean(np.abs(mono_RCS)):.6e}")
    print("Finished Balloon Shell test.\n")
