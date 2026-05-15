'''
    Purpose: Automated near-limit continuity checks for coated-sphere physics.

    Why this test matters:
    - Exact limit: if shell == background, coated solution must equal bare core sphere.
    - Near limit: as shell -> background (epsilon_r -> epsilon_bg), coated response
      should converge smoothly to bare-core response.

    Current repository behavior:
    - The exact-limit check should pass.
    - The near-limit check is expected to fail until full coated-shell formulas are fixed.

    This script is intentionally diagnostic: it prints a convergence table and raises
    AssertionError when continuity criteria are violated.
'''

import numpy as np

from CoatedSphere import CoatedSphere
from DielectricMaterial import DielectricMaterial
from getRCS import RCS_vs_freq, RCS_vs_freq_shell


def _max_relative_error(reference, candidate):
    valid = ~(np.isnan(reference) | np.isnan(candidate))
    if not np.any(valid):
        return np.inf
    denom = np.maximum(np.abs(reference[valid]), 1e-30)
    return float(np.max(np.abs(candidate[valid] - reference[valid]) / denom))


def run_near_limit_continuity_test():
    core_radius = 1.0
    shell_width = 0.03
    frequency = np.linspace(1.0e7, 1.0e9, 121)
    sensor_location = [0, 0, -2000]

    c0 = 299792458.0

    background = DielectricMaterial(1.0, 0.0, name="Air")
    pec = DielectricMaterial(1e8, 0.0, 1.0, 0.0, name="PEC")

    # Bare-core reference (physical target for transparent shell limit)
    ratio_bare = core_radius * frequency / c0
    _, rcs_bare = RCS_vs_freq(
        core_radius,
        ratio_bare,
        background,
        pec,
        sensor_location,
        show_plot=0,
    )

    # 1) Exact transparent-shell limit: should pass (uses exact reduction branch)
    shell_exact = DielectricMaterial(1.0, 0.0, name="Air")
    coated_exact = CoatedSphere(pec, shell_exact, core_radius=core_radius, shell_width=shell_width)
    ratio_exact = coated_exact.radius * frequency / c0
    _, rcs_exact = RCS_vs_freq_shell(
        coated_exact,
        ratio_exact,
        background,
        sensor_location,
        show_plot=0,
    )

    exact_rel_err = _max_relative_error(rcs_bare, rcs_exact)
    print("Exact limit check (shell == background):")
    print(f"  max relative error = {exact_rel_err:.3e}")
    if exact_rel_err > 1e-8:
        raise AssertionError(
            "Exact transparent-shell reduction failed. "
            f"max_rel_err={exact_rel_err:.3e} exceeds 1e-8"
        )

    # 2) Near-limit continuity: error should decrease as delta -> 0
    # Shell epsilon values approaching background from above
    deltas = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4], dtype=float)
    rel_errors = []

    print("\nNear-limit continuity sweep (shell epsilon_r = 1 + delta):")
    for delta in deltas:
        shell = DielectricMaterial(1.0 + delta, 0.0, name=f"Air+{delta:.0e}")
        coated = CoatedSphere(pec, shell, core_radius=core_radius, shell_width=shell_width)
        ratio_coated = coated.radius * frequency / c0
        _, rcs_coated = RCS_vs_freq_shell(
            coated,
            ratio_coated,
            background,
            sensor_location,
            show_plot=0,
        )
        rel_err = _max_relative_error(rcs_bare, rcs_coated)
        rel_errors.append(rel_err)
        print(f"  delta={delta:>7.1e}  max_rel_err={rel_err:>12.3e}")

    rel_errors = np.array(rel_errors, dtype=float)

    # For a physically consistent implementation, continuity implies a downward trend
    # as delta shrinks. We require substantial improvement from first to last point.
    improvement_factor = rel_errors[0] / max(rel_errors[-1], 1e-300)
    print("\nContinuity score:")
    print(f"  improvement factor (largest delta / smallest delta) = {improvement_factor:.3e}")

    if improvement_factor < 10.0:
        raise AssertionError(
            "Near-limit continuity failed: coated solution does not sufficiently "
            "approach bare-core result as shell->background. "
            f"improvement_factor={improvement_factor:.3e} < 1e1"
        )

    print("\nPASS: Near-limit continuity behavior is consistent.")


if __name__ == "__main__":
    print("------\nNEAR-LIMIT CONTINUITY TEST\n------\n")
    run_near_limit_continuity_test()
    print("\nFinished Near-Limit Continuity test.\n")
