'''
    Purpose: Calculate and plot the monostatic RCS of an air sphere (epsilon_r=1, sigma=0)
    in vacuum background. Since the sphere and background are electromagnetically identical,
    the RCS should be zero across all frequencies — no scattering occurs.

    Sphere radius: 1.0 m
    Frequency sweep: 1 MHz to 1 GHz (301 points)
    Sensor location: [0, 0, -2000] (monostatic, 2 km away)

    Output:
        output/air_sphere_rcs.png  — loglog plot
        output/air_sphere_rcs.txt  — tab-separated frequency/RCS data
'''

from getRCS import *
import numpy as np


if __name__ == "__main__":
    print("------\nAIR SPHERE MONOSTATIC RCS TEST\n------\n")

    radius = 1.0
    frequency = np.linspace(1e7, 4e8, 301)
    vacuum = DielectricMaterial(1.0, 0.0)
    air = DielectricMaterial(1.0, 0.0, 1.0, 0.0, name="Air")
    sensor_location = [0, 0, -2000]

    c0 = 299792458.0
    ratio = radius * frequency / c0

    (freq, mono_RCS) = RCS_vs_freq(
        radius, ratio, vacuum, air,
        sensor_location,
        save_file='output/air_sphere_rcs',
        show_plot=1
    )

    print(f"RCS min: {np.min(np.abs(mono_RCS)):.6e}")
    print(f"RCS max: {np.max(np.abs(mono_RCS)):.6e}")
    print(f"RCS mean: {np.mean(np.abs(mono_RCS)):.6e}")
    print("Finished Air Sphere test.\n")
