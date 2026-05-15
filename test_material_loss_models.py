'''
    Purpose: Validate new dielectric loss models in DielectricMaterial.

    This test verifies equivalence between the three supported electric-loss
    definitions at a chosen frequency f0:
      1) conductivity model      : epsilon_r + sigma/(j*omega*eps0)
      2) loss tangent model      : epsilon_r * (1 - j*tan_delta)
      3) complex permittivity    : epsilon_r_complex directly

    For equivalence at f0:
        sigma_eq = omega0 * eps0 * epsilon_r * tan_delta

    The test also checks coated-sphere RCS consistency for the three models
    when configured to be equivalent at f0.
'''

import numpy as np

from CoatedSphere import CoatedSphere
from DielectricMaterial import DielectricMaterial
from getCoatedSphereFieldUnderPlaneWave import getCoatedSphereFieldUnderPlaneWave
from src import norm


def _rcs_from_fields(fields, sensor_location):
    E_r, E_theta, E_phi, _, _, _ = fields
    E = np.stack((E_r, E_theta, E_phi), axis=0)
    return 4 * np.pi * (norm(sensor_location) ** 2) * np.sum((E * np.conj(E)), 0)


def _max_rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.maximum(np.abs(a), 1e-30)
    return float(np.max(np.abs(a - b) / denom))


if __name__ == "__main__":
    print("------\nMATERIAL LOSS MODEL EQUIVALENCE TEST\n------\n")

    eps_r = 4.0
    tan_delta = 0.15
    f0 = 1.0e8
    omega0 = 2 * np.pi * f0
    eps0 = 8.8541878176e-12
    sigma_eq = omega0 * eps0 * eps_r * tan_delta
    eps_complex = eps_r * (1 - 1j * tan_delta)

    mat_sigma = DielectricMaterial(eps_r, sigma_eq, 1.0, 0.0, name="sigma-model")
    mat_tan = DielectricMaterial(eps_r, 0.0, 1.0, 0.0, name="tan-model", loss_tangent=tan_delta)
    mat_complex = DielectricMaterial(eps_r, 0.0, 1.0, 0.0, name="complex-model", epsilon_r_complex=eps_complex)

    eps_sigma = mat_sigma.getComplexPermittivity(np.array([f0]))[0]
    eps_tan = mat_tan.getComplexPermittivity(np.array([f0]))
    eps_complex_direct = mat_complex.getComplexPermittivity(np.array([f0]))

    print(f"sigma_eq at f0: {sigma_eq:.6e} S/m")
    print(f"eps_sigma(f0):   {eps_sigma}")
    print(f"eps_tan(f0):     {eps_tan}")
    print(f"eps_complex(f0): {eps_complex_direct}")

    rel_eps_tan = _max_rel(np.array([eps_sigma]), np.array([eps_tan]))
    rel_eps_complex = _max_rel(np.array([eps_sigma]), np.array([eps_complex_direct]))

    print(f"\nPermittivity relative error (sigma vs tan):     {rel_eps_tan:.3e}")
    print(f"Permittivity relative error (sigma vs complex): {rel_eps_complex:.3e}")

    assert rel_eps_tan < 1e-12
    assert rel_eps_complex < 1e-12

    # Coated-sphere equivalence at f0
    background = DielectricMaterial(1.0, 0.0, name="Air")
    pec_core = DielectricMaterial(1e8, 0.0, 1.0, 0.0, name="PEC")
    sensor_location = [0, 0, -2000]

    coated_sigma = CoatedSphere(pec_core, mat_sigma, core_radius=1.0, shell_width=0.05)
    coated_tan = CoatedSphere(pec_core, mat_tan, core_radius=1.0, shell_width=0.05)
    coated_complex = CoatedSphere(pec_core, mat_complex, core_radius=1.0, shell_width=0.05)

    fields_sigma = getCoatedSphereFieldUnderPlaneWave(coated_sigma, background, sensor_location, np.array([f0]))
    fields_tan = getCoatedSphereFieldUnderPlaneWave(coated_tan, background, sensor_location, np.array([f0]))
    fields_complex = getCoatedSphereFieldUnderPlaneWave(coated_complex, background, sensor_location, np.array([f0]))

    rcs_sigma = _rcs_from_fields(fields_sigma, sensor_location)
    rcs_tan = _rcs_from_fields(fields_tan, sensor_location)
    rcs_complex = _rcs_from_fields(fields_complex, sensor_location)

    rel_rcs_tan = _max_rel(rcs_sigma, rcs_tan)
    rel_rcs_complex = _max_rel(rcs_sigma, rcs_complex)

    print(f"\nRCS relative error (sigma vs tan):     {rel_rcs_tan:.3e}")
    print(f"RCS relative error (sigma vs complex): {rel_rcs_complex:.3e}")

    assert rel_rcs_tan < 1e-9
    assert rel_rcs_complex < 1e-9

    print("\nPASS: Loss-model variants are equivalent at f0 when parameter-matched.\n")
