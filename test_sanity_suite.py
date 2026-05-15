'''
    Purpose: Broad sanity-suite for sphere/coated-sphere solvers.

    This script bundles many independent checks and is intended as a
    regression gate while iterating on formulas and material models.
'''

import numpy as np

from CoatedSphere import CoatedSphere
from DielectricMaterial import DielectricMaterial
from getDielectricSphereFieldUnderPlaneWave import getDielectricSphereFieldUnderPlaneWave
from getCoatedSphereFieldUnderPlaneWave import getCoatedSphereFieldUnderPlaneWave
from getCoatedPECSphereFieldUnderPlaneWave import getCoatedPECSphereFieldUnderPlaneWave
from getCoatedDielectricSphereFieldUnderPlaneWave import getCoatedDielectricSphereFieldUnderPlaneWave
from getRCS import RCS_vs_freq, RCS_vs_freq_shell
from src import norm, sphToCart


def _rcs_from_fields(fields, sensor_location):
    E_r, E_theta, E_phi, _, _, _ = fields
    E = np.stack((E_r, E_theta, E_phi), axis=0)
    return 4 * np.pi * (norm(sensor_location) ** 2) * np.sum((E * np.conj(E)), 0)


def _max_rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.maximum(np.abs(a), 1e-30)
    return float(np.max(np.abs(a - b) / denom))


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _sigma_from_tandelta(eps_r, tan_delta, f_hz, eps0=8.8541878176e-12):
    omega = 2 * np.pi * f_hz
    return omega * eps0 * eps_r * tan_delta


def _print_check(name, value, threshold, mode="<="):
    if mode == "<=":
        ok = value <= threshold
    else:
        ok = value >= threshold
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {value:.3e} (threshold {mode} {threshold:.3e})")
    return ok


if __name__ == "__main__":
    print("------\nSANITY SUITE\n------\n")
    failures = []

    def check(name, value, threshold, mode="<="):
        ok = _print_check(name, value, threshold, mode)
        if not ok:
            failures.append((name, value, threshold, mode))

    c0 = 299792458.0
    freq = np.linspace(2e9, 6e9, 81)

    air = DielectricMaterial(1.0, 0.0, name="Air")
    pec = DielectricMaterial(1e8, 0.0, 1.0, 0.0, name="PEC")
    diel = DielectricMaterial(2.5, 0.0, 1.0, 0.0, name="Diel")

    sensor_2km = [0, 0, -2000]
    sensor_4km = [0, 0, -4000]

    # 1) Air sphere must not scatter.
    ratio = 1.0 * freq / c0
    _, rcs_air = RCS_vs_freq(1.0, ratio, air, air, sensor_2km, show_plot=0)
    check("Air sphere zero scatter (max abs)", float(np.max(np.abs(rcs_air))), 1e-8)

    # 2) Transparent shell reduction (PEC core + air shell == bare PEC core).
    coated_transparent = CoatedSphere(pec, air, core_radius=1.0, shell_width=0.01)
    ratio_coated = coated_transparent.radius * freq / c0
    _, rcs_coated_t = RCS_vs_freq_shell(coated_transparent, ratio_coated, air, sensor_2km, show_plot=0)

    ratio_bare = 1.0 * freq / c0
    _, rcs_bare_pec = RCS_vs_freq(1.0, ratio_bare, air, pec, sensor_2km, show_plot=0)
    check("Transparent shell reduction (max rel)", _max_rel(rcs_bare_pec, rcs_coated_t), 1e-8)

    # 3) Uniform material reduction (core==shell -> bare outer sphere).
    coated_uniform = CoatedSphere(diel, diel, core_radius=1.0, shell_width=0.02)
    ratio_uniform = coated_uniform.radius * freq / c0
    _, rcs_coated_u = RCS_vs_freq_shell(coated_uniform, ratio_uniform, air, sensor_2km, show_plot=0)

    _, rcs_bare_u = RCS_vs_freq(coated_uniform.radius, ratio_uniform, air, diel, sensor_2km, show_plot=0)
    check("Uniform material reduction (max rel)", _max_rel(rcs_bare_u, rcs_coated_u), 1e-8)

    # 4) Distance invariance for monostatic RCS.
    _, rcs_d2 = RCS_vs_freq(1.0, ratio_bare, air, pec, sensor_2km, show_plot=0)
    _, rcs_d4 = RCS_vs_freq(1.0, ratio_bare, air, pec, sensor_4km, show_plot=0)
    check("Distance invariance (max rel)", _max_rel(rcs_d2, rcs_d4), 1e-2)

    # 5) Backscatter-direction phi degeneracy:
    # theta=pi maps to the same Cartesian point for any phi.
    distance = 2000
    theta = np.pi
    s_phi0 = sphToCart(distance, theta, 0.0)
    s_phi1 = sphToCart(distance, theta, np.pi / 3.0)

    fields_phi0 = getCoatedSphereFieldUnderPlaneWave(coated_uniform, air, s_phi0, freq)
    fields_phi1 = getCoatedSphereFieldUnderPlaneWave(coated_uniform, air, s_phi1, freq)
    rcs_phi0 = _rcs_from_fields(fields_phi0, s_phi0)
    rcs_phi1 = _rcs_from_fields(fields_phi1, s_phi1)
    check("Backscatter phi degeneracy (max rel)", _max_rel(rcs_phi0, rcs_phi1), 1e-8)

    # 6) Frequency ordering invariance.
    freq_rev = freq[::-1]
    ratio_rev = 1.0 * freq_rev / c0
    _, rcs_rev = RCS_vs_freq(1.0, ratio_rev, air, pec, sensor_2km, show_plot=0)
    check("Frequency order invariance (max rel)", _max_rel(rcs_d2, rcs_rev[::-1]), 1e-10)

    # 7) Frequency input type invariance for single-frequency call.
    f0 = 3.5e9
    f_scalar = f0
    f_list = [f0]
    f_array = np.array([f0])
    E1 = getDielectricSphereFieldUnderPlaneWave(1.0, pec, air, sensor_2km, f_scalar)
    E2 = getDielectricSphereFieldUnderPlaneWave(1.0, pec, air, sensor_2km, f_list)
    E3 = getDielectricSphereFieldUnderPlaneWave(1.0, pec, air, sensor_2km, f_array)
    check("Frequency type invariance (scalar/list)", _max_abs(E1[0], E2[0]), 1e-12)
    check("Frequency type invariance (scalar/array)", _max_abs(E1[0], E3[0]), 1e-12)

    # 8) Dispatcher consistency for PEC-core branch.
    shell_lossy = DielectricMaterial(4.0, 3.0, 1.0, 0.0, name="Shell")
    coated_pec = CoatedSphere(pec, shell_lossy, core_radius=1.0, shell_width=0.05)
    fields_dispatch_pec = getCoatedSphereFieldUnderPlaneWave(coated_pec, air, sensor_2km, freq)
    fields_direct_pec = getCoatedPECSphereFieldUnderPlaneWave(
        coated_pec.radius,
        coated_pec.shell_material,
        air,
        sensor_2km,
        freq,
        coating_thickness=coated_pec.shell_width,
        inner_radius=coated_pec.core_radius,
    )
    rcs_dispatch_pec = _rcs_from_fields(fields_dispatch_pec, sensor_2km)
    rcs_direct_pec = _rcs_from_fields(fields_direct_pec, sensor_2km)
    check("Dispatcher PEC branch consistency (max rel)", _max_rel(rcs_dispatch_pec, rcs_direct_pec), 1e-12)

    # 9) Dispatcher consistency for dielectric-core branch.
    core_d = DielectricMaterial(1.6, 0.0, 1.0, 0.0, name="CoreD")
    shell_d = DielectricMaterial(3.5, 0.1, 1.0, 0.0, name="ShellD")
    coated_d = CoatedSphere(core_d, shell_d, core_radius=1.0, shell_width=0.05)
    fields_dispatch_d = getCoatedSphereFieldUnderPlaneWave(coated_d, air, sensor_2km, freq)
    fields_direct_d = getCoatedDielectricSphereFieldUnderPlaneWave(
        coated_d.radius,
        coated_d.shell_material,
        coated_d.core_material,
        air,
        sensor_2km,
        freq,
        coating_thickness=coated_d.shell_width,
        inner_radius=coated_d.core_radius,
    )
    rcs_dispatch_d = _rcs_from_fields(fields_dispatch_d, sensor_2km)
    rcs_direct_d = _rcs_from_fields(fields_direct_d, sensor_2km)
    check("Dispatcher dielectric branch consistency (max rel)", _max_rel(rcs_dispatch_d, rcs_direct_d), 1e-12)

    # 10) Finite outputs in representative scenarios.
    balloon_shell = DielectricMaterial(4.0, 0.0, 1.0, 0.0, name="BalloonShell", loss_tangent=0.005)
    coated_balloon = CoatedSphere(air, balloon_shell, core_radius=1.5 - 0.3e-3, shell_width=0.3e-3)
    _, rcs_balloon = RCS_vs_freq_shell(
        coated_balloon,
        coated_balloon.radius * freq / c0,
        air,
        sensor_2km,
        show_plot=0,
    )
    finite_ratio = np.sum(np.isfinite(rcs_balloon)) / len(rcs_balloon)
    check("Finite outputs ratio", finite_ratio, 1.0, mode=">=")

    # 11) Complex epsilon and tan(delta) equivalence on solver output.
    shell_tan = DielectricMaterial(4.0, 0.0, 1.0, 0.0, loss_tangent=0.005)
    shell_cmp = DielectricMaterial(4.0, 0.0, 1.0, 0.0, epsilon_r_complex=4.0 - 1j * 0.02)
    coated_tan = CoatedSphere(air, shell_tan, core_radius=1.0, shell_width=0.05)
    coated_cmp = CoatedSphere(air, shell_cmp, core_radius=1.0, shell_width=0.05)
    _, rcs_tan = RCS_vs_freq_shell(coated_tan, coated_tan.radius * freq / c0, air, sensor_2km, show_plot=0)
    _, rcs_cmp = RCS_vs_freq_shell(coated_cmp, coated_cmp.radius * freq / c0, air, sensor_2km, show_plot=0)
    check("Complex epsilon vs tan(delta) equivalence (max rel)", _max_rel(rcs_tan, rcs_cmp), 1e-8)

    # 12) Dielectric-core near-limit continuity shell->background.
    core_for_limit = DielectricMaterial(2.0, 0.0, 1.0, 0.0)
    coated_ref = CoatedSphere(core_for_limit, air, core_radius=1.0, shell_width=0.03)
    _, rcs_ref = RCS_vs_freq(
        1.0,
        1.0 * freq / c0,
        air,
        core_for_limit,
        sensor_2km,
        show_plot=0,
    )

    deltas = [1e-1, 1e-2, 1e-3, 1e-4]
    errs = []
    for d in deltas:
        shell = DielectricMaterial(1.0 + d, 0.0, 1.0, 0.0)
        coated = CoatedSphere(core_for_limit, shell, core_radius=1.0, shell_width=0.03)
        _, rcs_val = RCS_vs_freq_shell(coated, coated.radius * freq / c0, air, sensor_2km, show_plot=0)
        errs.append(_max_rel(rcs_ref, rcs_val))

    improvement = errs[0] / max(errs[-1], 1e-30)
    check("Near-limit continuity improvement factor", improvement, 10.0, mode=">=")

    # 13) RCS should be real and non-negative within numerical tolerance.
    real_part = np.real(rcs_balloon)
    imag_part = np.imag(rcs_balloon)
    check("RCS imaginary leakage (max abs)", float(np.max(np.abs(imag_part))), 1e-8)
    check("RCS non-negativity (min real)", float(np.min(real_part)), -1e-10, mode=">=")

    # 14) Multi-material shell matrix checks (finite outputs + passivity behavior).
    print("\nMaterial matrix checks:")
    f_ref = 8e9
    material_cases = [
        {
            "name": "LosslessLowPerm",
            "mat": DielectricMaterial(2.2, 0.0, 1.0, 0.0),
            "passive_lossy": False,
        },
        {
            "name": "ConductiveModerate",
            "mat": DielectricMaterial(4.0, 0.02, 1.0, 0.0),
            "passive_lossy": True,
        },
        {
            "name": "HighPermLossless",
            "mat": DielectricMaterial(10.0, 0.0, 1.0, 0.0),
            "passive_lossy": False,
        },
        {
            "name": "LossTangentSmall",
            "mat": DielectricMaterial(4.0, 0.0, 1.0, 0.0, loss_tangent=0.005),
            "passive_lossy": True,
        },
        {
            "name": "ComplexPermittivity",
            "mat": DielectricMaterial(3.2, 0.0, 1.0, 0.0, epsilon_r_complex=3.2 - 0.15j),
            "passive_lossy": True,
        },
        {
            "name": "MagneticLossless",
            "mat": DielectricMaterial(3.0, 0.0, 1.3, 0.0),
            "passive_lossy": False,
        },
    ]

    for case in material_cases:
        name = case["name"]
        shell_mat = case["mat"]
        coated_case = CoatedSphere(air, shell_mat, core_radius=1.5 - 0.3e-3, shell_width=0.3e-3)
        _, rcs_case = RCS_vs_freq_shell(
            coated_case,
            coated_case.radius * freq / c0,
            air,
            sensor_2km,
            show_plot=0,
        )

        finite_ratio_case = np.sum(np.isfinite(rcs_case)) / len(rcs_case)
        check(f"{name} finite outputs ratio", finite_ratio_case, 1.0, mode=">=")

        imag_leak_case = float(np.max(np.abs(np.imag(rcs_case))))
        check(f"{name} RCS imaginary leakage (max abs)", imag_leak_case, 1e-8)

        min_real_case = float(np.min(np.real(rcs_case)))
        check(f"{name} RCS non-negativity (min real)", min_real_case, -1e-10, mode=">=")

        if case["passive_lossy"]:
            min_att_case = float(np.min(shell_mat.getAbsorptionDepth(freq)))
            check(f"{name} attenuation metric (min)", min_att_case, 0.0, mode=">=")

    # 15) Integrate complex-permittivity verification into this suite for
    # multiple representative loss tangents.
    print("\nComplex permittivity equivalence matrix:")
    epsr_candidates = [2.5, 4.0, 8.0]
    tan_candidates = [0.001, 0.005, 0.02]
    for eps_r in epsr_candidates:
        for tan_delta in tan_candidates:
            eps_complex = eps_r * (1 - 1j * tan_delta)
            sigma_eq = _sigma_from_tandelta(eps_r, tan_delta, f_ref)

            mat_tan = DielectricMaterial(eps_r, 0.0, 1.0, 0.0, loss_tangent=tan_delta)
            mat_cmp = DielectricMaterial(eps_r, 0.0, 1.0, 0.0, epsilon_r_complex=eps_complex)
            mat_sig = DielectricMaterial(eps_r, sigma_eq, 1.0, 0.0)

            # Material-level equivalence at reference frequency.
            eps_tan = np.atleast_1d(mat_tan.getComplexPermittivity(np.array([f_ref]))).astype(np.complex128)
            eps_cmp = np.atleast_1d(mat_cmp.getComplexPermittivity(np.array([f_ref]))).astype(np.complex128)
            eps_sig = np.atleast_1d(mat_sig.getComplexPermittivity(np.array([f_ref]))).astype(np.complex128)

            tag = f"eps{eps_r:g}_tan{tan_delta:g}"
            check(f"{tag} permittivity tan vs complex (max rel)", _max_rel(eps_tan, eps_cmp), 1e-12)
            check(f"{tag} permittivity tan vs sigma (max rel)", _max_rel(eps_tan, eps_sig), 1e-12)

            # Solver-level equivalence in a coated-shell scenario.
            coated_tan_case = CoatedSphere(air, mat_tan, core_radius=1.0, shell_width=0.05)
            coated_cmp_case = CoatedSphere(air, mat_cmp, core_radius=1.0, shell_width=0.05)
            coated_sig_case = CoatedSphere(air, mat_sig, core_radius=1.0, shell_width=0.05)

            ratio_case = coated_tan_case.radius * freq / c0
            _, rcs_tan_case = RCS_vs_freq_shell(coated_tan_case, ratio_case, air, sensor_2km, show_plot=0)
            _, rcs_cmp_case = RCS_vs_freq_shell(coated_cmp_case, ratio_case, air, sensor_2km, show_plot=0)
            ratio_ref = coated_sig_case.radius * np.array([f_ref]) / c0
            _, rcs_tan_ref = RCS_vs_freq_shell(coated_tan_case, ratio_ref, air, sensor_2km, show_plot=0)
            _, rcs_sig_ref = RCS_vs_freq_shell(coated_sig_case, ratio_ref, air, sensor_2km, show_plot=0)

            check(f"{tag} solver tan vs complex (max rel)", _max_rel(rcs_tan_case, rcs_cmp_case), 1e-8)
            check(f"{tag} solver tan vs sigma at f_ref (max rel)", _max_rel(rcs_tan_ref, rcs_sig_ref), 1e-6)

    if failures:
        print("\nFailed checks:")
        for name, value, threshold, mode in failures:
            print(f"  - {name}: {value:.3e} (threshold {mode} {threshold:.3e})")
        raise AssertionError(f"{len(failures)} sanity checks failed.")

    print("\nAll sanity checks passed.\n")
