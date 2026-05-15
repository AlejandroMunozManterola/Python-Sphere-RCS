'''
    Purpose: Calculate and plot the monostatic RCS of a Perfect Electric Conductor (PEC)
    sphere in vacuum background. A classical Mie scattering test case — the PEC reflects
    the incident wave, producing a frequency-dependent RCS with characteristic oscillations
    (Mie resonances) as the sphere size parameter sweeps through different regimes.

    Sphere radius: 1.0 m
    Frequency sweep: 1 MHz to 1 GHz (301 points)
    Sensor location: [0, 0, -2000] (monostatic, 2 km away)

    Output:
        output/pec_sphere_rcs.png  — loglog plot
        output/pec_sphere_rcs.txt  — tab-separated frequency/RCS data
'''

from getRCS import *
import numpy as np


if __name__ == "__main__":
    print("------\nPEC SPHERE MONOSTATIC RCS TEST\n------\n")

    radius = 1.0
    frequency = np.linspace(1e7, 1e9, 301)
    vacuum = DielectricMaterial(1.0, 0.0)
    pec = DielectricMaterial(1e8, 0.0, 1e-8, 0.0, name="PEC")
    sensor_location = [0, 0, -2000]

    c0 = 299792458.0
    ratio = radius * frequency / c0

    (freq, mono_RCS) = RCS_vs_freq(
        radius, ratio, vacuum, pec,
        sensor_location,
        save_file='output/pec_sphere_rcs',
        show_plot=1
    )

    print(f"RCS min: {np.min(np.abs(mono_RCS)):.6e}")
    print(f"RCS max: {np.max(np.abs(mono_RCS)):.6e}")
    print(f"RCS mean: {np.mean(np.abs(mono_RCS)):.6e}")
    print("Finished PEC Sphere test.\n")
