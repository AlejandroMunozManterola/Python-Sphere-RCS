'''
    Purpose: Calculate and plot the monostatic RCS of a coated sphere —
    a PEC core (1.0 m radius) with a dielectric shell (3 mm thickness)
    in vacuum background.

    The shell modifies the RCS compared to a bare PEC sphere by
    introducing additional phase shifts and interference effects.

    Core:    PEC, radius = 1.0 m
    Shell:   ε_r = 2.56, σ = 0, width = 3 mm
    Background: vacuum
    Frequency sweep: 1 MHz to 1 GHz (301 points)
    Sensor location: [0, 0, -2000] (monostatic, 2 km away)

    Output:
        output/coated_sphere_rcs.png  — loglog plot
        output/coated_sphere_rcs.txt  — tab-separated frequency/RCS data
'''

from getRCS import *
import numpy as np


if __name__ == "__main__":
    print("------\nCOATED SPHERE MONOSTATIC RCS TEST\n------\n")

    radius = 1.0
    frequency = np.linspace(1e7, 4e8, 301)
    vacuum = DielectricMaterial(1.0, 0.0)
    pec = DielectricMaterial(1e8, 0.0, 1.0, 0.0, name="PEC")
    shell_mat = DielectricMaterial(2.56, 0.0, 1.0, 0.0, name="Dielectric")
    sensor_location = [0, 0, -2000]

    c0 = 299792458.0
    ratio = radius * frequency / c0

    coated = CoatedSphere(pec, shell_mat, core_radius=radius, shell_width=0.003)

    (freq, mono_RCS) = RCS_vs_freq_shell(
        coated, ratio, vacuum,
        sensor_location,
        save_file='output/coated_sphere_rcs',
        show_plot=1
    )

    print(f"RCS min: {np.min(np.abs(mono_RCS)):.6e}")
    print(f"RCS max: {np.max(np.abs(mono_RCS)):.6e}")
    print(f"RCS mean: {np.mean(np.abs(mono_RCS)):.6e}")
    print("Finished Coated Sphere test.\n")
