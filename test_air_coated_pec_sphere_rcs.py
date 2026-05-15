'''
    Purpose: Validate coated-sphere reduction for a PEC sphere with a
    transparent shell. When shell material equals background material,
    the shell should not scatter, so the result must match a bare PEC
    sphere of the core radius.

    Core:    PEC, radius = 1.0 m
    Shell:   air, width = 3 cm
    Outer radius: 1.03 m (should not affect scattering)
    Background: air
    Frequency sweep: 10 MHz to 1 GHz (301 points)
    Sensor location: [0, 0, -2000] (monostatic, 2 km away)

    Output:
        output/air_coated_pec_sphere_rcs.png  — loglog plot
        output/air_coated_pec_sphere_rcs.txt  — tab-separated frequency/RCS data
'''

from getRCS import *
import numpy as np


if __name__ == "__main__":
    print("------\nPEC CORE + AIR SHELL VALIDATION TEST\n------\n")

    core_radius = 1.0
    shell_width = 3e-2  # 3 cm
    frequency = np.linspace(1e7, 4e8, 301)
    air = DielectricMaterial(1.0, 0.0, name="Air")
    pec = DielectricMaterial(1e8, 0.0, 1.0, 0.0, name="PEC")
    sensor_location = [0, 0, -2000]

    c0 = 299792458.0

    # Coated sphere: PEC core + air shell (should = bare PEC core)
    coated = CoatedSphere(pec, air, core_radius=core_radius, shell_width=shell_width)
    ratio_coated = coated.radius * frequency / c0

    (freq_coated, rcs_coated) = RCS_vs_freq_shell(
        coated, ratio_coated, air,
        sensor_location,
        save_file='output/air_coated_pec_sphere_rcs',
        show_plot=1
    )

    # Bare PEC sphere of core radius (should match)
    ratio_bare = core_radius * frequency / c0
    (freq_bare, rcs_bare) = RCS_vs_freq(
        core_radius, ratio_bare, air, pec,
        sensor_location,
        save_file='output/air_coated_bare_comparison',
        show_plot=0
    )

    print(f"Coated RCS min: {np.min(np.abs(rcs_coated)):.6e}")
    print(f"Coated RCS max: {np.max(np.abs(rcs_coated)):.6e}")
    print(f"Coated RCS mean: {np.mean(np.abs(rcs_coated)):.6e}")
    print(f"Bare   RCS min: {np.min(np.abs(rcs_bare)):.6e}")
    print(f"Bare   RCS max: {np.max(np.abs(rcs_bare)):.6e}")
    print(f"Bare   RCS mean: {np.mean(np.abs(rcs_bare)):.6e}")

    # Compare (skip NaN entries)
    valid = ~(np.isnan(rcs_coated) | np.isnan(rcs_bare))
    if np.any(valid):
        max_abs_diff = np.max(np.abs(rcs_coated[valid] - rcs_bare[valid]))
        max_rel_diff = np.max(np.abs((rcs_coated[valid] - rcs_bare[valid]) / np.maximum(np.abs(rcs_bare[valid]), 1e-300)))
        print(f"\nMax abs difference (coated - bare): {max_abs_diff:.6e}")
        print(f"Max rel difference: {max_rel_diff:.6e}")
    else:
        print("\nAll values are NaN — comparison not possible.")

    print("Finished Coated Sphere validation test.\n")
