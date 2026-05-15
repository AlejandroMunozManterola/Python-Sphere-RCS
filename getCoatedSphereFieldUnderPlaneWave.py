import numpy as np
from DielectricMaterial import DielectricMaterial as DM
from getDielectricSphereFieldUnderPlaneWave import getDielectricSphereFieldUnderPlaneWave
from getCoatedPECSphereFieldUnderPlaneWave import getCoatedPECSphereFieldUnderPlaneWave
from getCoatedDielectricSphereFieldUnderPlaneWave import getCoatedDielectricSphereFieldUnderPlaneWave


def _material_equivalent(mat_a, mat_b, frequency, rel_tol=1e-10):
    k_a = DM.getWaveNumber(mat_a, frequency)
    k_b = DM.getWaveNumber(mat_b, frequency)
    mu_a = DM.getComplexPermeability(mat_a, frequency)
    mu_b = DM.getComplexPermeability(mat_b, frequency)

    k_scale = max(np.max(np.abs(k_a)), np.max(np.abs(k_b)), 1.0)
    mu_scale = max(np.max(np.abs(mu_a)), np.max(np.abs(mu_b)), 1.0)

    k_match = np.max(np.abs(k_a - k_b)) <= rel_tol * k_scale
    mu_match = np.max(np.abs(mu_a - mu_b)) <= rel_tol * mu_scale
    return k_match and mu_match


def _is_pec(material):
    if material.name == "PEC":
        return True
    if material.epsilon_r > 1e5 and abs(material.mu_r - 1.0) <= 1e-2:
        return True
    return False


def getCoatedSphereFieldUnderPlaneWave(coated_sphere, background, sensor_location, frequency):
    """
    Dispatcher for coated-sphere scattering.

    Uses exact reductions for degenerate limits and routes to the dedicated
    PEC-core or dielectric-core coated solvers for strict two-layer cases.
    """

    if isinstance(frequency, (int, float)):
        frequency = np.array([frequency])
    frequency = np.asarray(frequency, dtype=complex).flatten()

    core_material = coated_sphere.core_material
    shell_material = coated_sphere.shell_material
    core_radius = coated_sphere.core_radius
    outer_radius = coated_sphere.radius
    shell_width = coated_sphere.shell_width

    # Exact reduction: transparent shell.
    if _material_equivalent(shell_material, background, frequency):
        return getDielectricSphereFieldUnderPlaneWave(
            core_radius,
            core_material,
            background,
            sensor_location,
            frequency,
        )

    # Exact reduction: uniform material.
    if _material_equivalent(core_material, shell_material, frequency):
        return getDielectricSphereFieldUnderPlaneWave(
            outer_radius,
            shell_material,
            background,
            sensor_location,
            frequency,
        )

    if _is_pec(core_material):
        return getCoatedPECSphereFieldUnderPlaneWave(
            outer_radius,
            shell_material,
            background,
            sensor_location,
            frequency,
            coating_thickness=shell_width,
            inner_radius=core_radius,
        )

    return getCoatedDielectricSphereFieldUnderPlaneWave(
        outer_radius,
        shell_material,
        core_material,
        background,
        sensor_location,
        frequency,
        coating_thickness=shell_width,
        inner_radius=core_radius,
    )


if __name__ == "__main__":
    bg = DM(1.0, 0.0)
    core = DM(1e8, 0.0, 1.0, 0.0, name="PEC")
    shell = DM(4.0, 3.0, 1.0, 0.0, name="Shell")
    class _C:
        pass
    coated = _C()
    coated.core_material = core
    coated.shell_material = shell
    coated.core_radius = 1.0
    coated.shell_width = 0.05
    coated.radius = 1.05
    getCoatedSphereFieldUnderPlaneWave(coated, bg, [0, 0, 100], [1e7, 5e7, 1e8])
