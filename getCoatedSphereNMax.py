import numpy as np
import math
from DielectricMaterial import DielectricMaterial as DM
from CoatedSphere import CoatedSphere


def getCoatedSphereNMax(coated_sphere, background, frequency):
    """
    Determines the appropriate number of Mie terms to evaluate for a
    coated (concentric shell) sphere.

    Based on the Wiscombe 1980 recommendation, using the OUTER radius
    (core_radius + shell_width) for the size parameter.

    Designed to work for single-layered (monolithic) sphere.
    """
    # check that frequency input is correct
    if (type(frequency) == int or type(frequency) == float):
        frequency = np.array([frequency])
    if (type(frequency) == list or type(frequency) == np.ndarray):
        frequency = np.array(frequency).flatten()
        M = len(frequency)
    else:
        print("wrong data type for frequency (in getCoatedSphereNMax)")

    # Use outer radius for size parameter
    outer_radius = coated_sphere.radius

    k_m = DM.getWaveNumber(background, frequency)
    x = abs(k_m * outer_radius)

    # Relative refractive index: shell / background
    n_shell = DM.getComplexRefractiveIndex(coated_sphere.shell_material, frequency)
    n_bg = DM.getComplexRefractiveIndex(background, frequency)
    m = n_shell / n_bg

    N_max = np.ones((M,))
    for k in range(0, M):
        if (x[k] < 0.02):
            print("WARNING: it is better to use Rayleigh Scattering models for low frequencies.")
            print("\tNo less than 3 Mie series terms will be used in this calculation")
            N_stop = 3
        elif (0.02 <= x[k] and x[k] <= 8):
            N_stop = x[k] + 4. * x[k] ** (1 / 3) + 1
        elif (8 < x[k] and x[k] < 4200):
            N_stop = x[k] + 4.05 * x[k] ** (1 / 3) + 2
        elif (4200 <= x[k] and x[k] <= 20000):
            N_stop = x[k] + 4. * x[k] ** (1 / 3) + 2
        else:
            print("WARNING: it is better to use Physical Optics models for high frequencies.")
            N_stop = 20000 + 4. * 20000 ** (1 / 3) + 2

        # KZHU formula: also consider the material contrast term
        n_kzhu = abs(m[k] * x[k]) + 15

        # Cap N_max to prevent explosion for PEC-like materials (n >> 1)
        N_max[k] = max(N_stop, min(n_kzhu, 500))

        # Cap based on x to avoid numerical issues with ric_bessely
        if x[k] < 1.0:
            N_max[k] = min(N_max[k], int(x[k]) + 150)

    # Use max of Wiscombe N_stop values, capped at 150
    wiscombe_max = max([x[k] + 4.*x[k]**(1/3) + 2 if x[k] >= 8 else x[k] + 4.*x[k]**(1/3) + 1 if x[k] >= 0.02 else 3 for k in range(M)])
    N_final = min(int(math.ceil(wiscombe_max)), 150)
    return max(N_final, 3)


if __name__ == "__main__":
    core = DM(2.56, 0.0)
    shell = DM(3.0, 0.0)
    background = DM(1.0, 0.0)
    coated = CoatedSphere(core, shell, core_radius=0.5, shell_width=0.01)
    frequency = np.logspace(5, 9, 5)
    print(frequency)
    print(getCoatedSphereNMax(coated, background, frequency))
