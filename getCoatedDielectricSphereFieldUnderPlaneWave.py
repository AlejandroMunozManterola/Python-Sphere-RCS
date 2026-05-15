import numpy as np
from DielectricMaterial import DielectricMaterial as DM
from src import *
from bessel import *
from getNMax import getNMax


def _safe_divide(numerator, denominator, eps=1e-300):
    den = np.where(np.abs(denominator) < eps, eps + 0j, denominator)
    return numerator / den


def getCoatedDielectricSphereFieldUnderPlaneWave(
    outer_radius,
    coating_material,
    core_material,
    background,
    sensor_location,
    frequency,
    coating_thickness=None,
    inner_radius=None,
):
    """
    Scattered field for a coated dielectric sphere (general core + shell).
    """

    if coating_thickness is not None and inner_radius is None:
        inner_radius = outer_radius - coating_thickness
    elif inner_radius is None and coating_thickness is None:
        raise ValueError("Provide either coating_thickness or inner_radius")

    if inner_radius <= 0 or inner_radius >= outer_radius:
        raise ValueError("inner_radius must be in (0, outer_radius)")

    if isinstance(frequency, (int, float)):
        frequency = np.array([frequency])
    frequency = np.asarray(frequency, dtype=complex).flatten()
    M = len(frequency)

    eta_m = DM.getIntrinsicImpedance(background, frequency)
    k_m = DM.getWaveNumber(background, frequency)
    k_1 = DM.getWaveNumber(core_material, frequency)
    k_2 = DM.getWaveNumber(coating_material, frequency)

    m1 = k_1 / k_m
    m2 = k_2 / k_m

    N_shell = getNMax(outer_radius, coating_material, background, frequency)
    N_core = getNMax(inner_radius, core_material, background, frequency)
    N = max(N_shell, N_core)
    nu = np.arange(1, N + 1, dtype=int)

    [r, theta, phi] = cartToSph(sensor_location[0], sensor_location[1], sensor_location[2])

    inc_n = np.empty((M, N), dtype=complex)
    for c, n in enumerate(nu):
        inc_n[:, c] = (1j ** (-n)) * (2 * n + 1) / (n * (n + 1))

    aux0 = np.zeros((N, 1), dtype=complex)
    aux0[0] = -1.0
    if N >= 2:
        aux0[1] = -3.0 * np.cos(theta)
    for n in range(2, N):
        aux0[n] = ((2 * n + 1) / n) * np.cos(theta) * aux0[n - 1] - ((n + 1) / n) * aux0[n - 2]

    aux1 = np.zeros((N, 1), dtype=complex)
    aux1[0] = np.cos(theta)
    for n in range(2, N + 1):
        aux1[n - 1] = (n + 1) * aux0[n - 2] - n * np.cos(theta) * aux0[n - 1]

    aux0 = np.ones((M, 1)) @ aux0.reshape(1, N)
    aux1 = np.ones((M, 1)) @ aux1.reshape(1, N)

    y = k_m * inner_radius
    x = k_m * outer_radius
    m1y = m1 * y
    m2y = m2 * y
    m2x = m2 * x

    psi_m1y = ric_besselj(nu, m1y)
    psi_m1y_p = ric_besselj_derivative(nu, m1y)

    psi_m2y = ric_besselj(nu, m2y)
    psi_m2y_p = ric_besselj_derivative(nu, m2y)
    chi_m2y = -ric_bessely(nu, m2y)
    chi_m2y_p = -ric_bessely_derivative(nu, m2y)

    m1_row = m1.reshape(1, M)
    m2_row = m2.reshape(1, M)

    A_num = m2_row * psi_m2y * psi_m1y_p - m1_row * psi_m2y_p * psi_m1y
    A_den = m2_row * chi_m2y * psi_m1y_p - m1_row * chi_m2y_p * psi_m1y
    A_n = _safe_divide(A_num, A_den)

    B_num = m2_row * psi_m2y_p * psi_m1y - m1_row * psi_m2y * psi_m1y_p
    B_den = m2_row * chi_m2y_p * psi_m1y - m1_row * chi_m2y * psi_m1y_p
    B_n = _safe_divide(B_num, B_den)

    psi_x = ric_besselj(nu, x)
    psi_x_p = ric_besselj_derivative(nu, x)
    xi_x = ric_besselh(nu, x, 2)
    xi_x_p = ric_besselh_derivative(nu, x, 2)

    psi_m2x = ric_besselj(nu, m2x)
    psi_m2x_p = ric_besselj_derivative(nu, m2x)
    chi_m2x = -ric_bessely(nu, m2x)
    chi_m2x_p = -ric_bessely_derivative(nu, m2x)

    hat_a_1 = psi_m2x_p - A_n * chi_m2x_p
    hat_a_0 = psi_m2x - A_n * chi_m2x
    num_a = m2_row * psi_x * hat_a_1 - psi_x_p * hat_a_0
    den_a = m2_row * xi_x * hat_a_1 - xi_x_p * hat_a_0
    scat_a_n = _safe_divide(num_a, den_a).T * inc_n

    hat_b_1 = psi_m2x_p - B_n * chi_m2x_p
    hat_b_0 = psi_m2x - B_n * chi_m2x
    num_b = psi_x * hat_b_1 - m2_row * psi_x_p * hat_b_0
    den_b = xi_x * hat_b_1 - m2_row * xi_x_p * hat_b_0
    scat_b_n = _safe_divide(num_b, den_b).T * inc_n

    for i in range(1, M):
        num_zeros = 0
        for j in range(N):
            if abs(scat_a_n[i, j]) < 1e-300:
                num_zeros += 1
            if num_zeros > 4:
                scat_a_n[i, j:] = 0
                break

    for i in range(1, M):
        num_zeros = 0
        for j in range(N):
            if abs(scat_b_n[i, j]) < 1e-300:
                num_zeros += 1
            if num_zeros > 4:
                scat_b_n[i, j:] = 0
                break

    x_obs = k_m * r

    alpha00 = np.transpose(ric_besselh_derivative(nu, x_obs, 2, 2))
    alpha01 = np.transpose(ric_besselh(nu, x_obs, 2))
    alpha10 = np.array(get_legendre(nu, 1, np.cos(theta)))
    alpha11 = np.transpose(np.reshape(alpha10, (N, 1)) @ np.ones((1, M)))
    alpha = (alpha00 + alpha01) * alpha11

    E_r = -1j * np.cos(phi) * np.sum(scat_a_n * alpha, axis=1)
    H_r = -1j * np.sin(phi) * np.sum(scat_b_n * alpha, axis=1) / eta_m

    alpha_e = np.transpose(ric_besselh_derivative(nu, x_obs, 2)) * aux1
    beta_e = np.transpose(ric_besselh(nu, x_obs, 2)) * aux0
    E_theta = (np.cos(phi) / x_obs) * np.sum(1j * scat_a_n * alpha_e - scat_b_n * beta_e, axis=1)
    H_theta = (np.sin(phi) / x_obs) * np.sum(1j * scat_b_n * alpha_e - scat_a_n * beta_e, axis=1) / eta_m

    alpha_p = np.transpose(ric_besselh_derivative(nu, x_obs, 2)) * aux0
    beta_p = np.transpose(ric_besselh(nu, x_obs, 2)) * aux1
    E_phi = (np.sin(phi) / x_obs) * np.sum(1j * scat_a_n * alpha_p - scat_b_n * beta_p, axis=1)
    H_phi = (-np.cos(phi) / x_obs) * np.sum(1j * scat_b_n * alpha_p - scat_a_n * beta_p, axis=1) / eta_m

    return [E_r, E_theta, E_phi, H_r, H_theta, H_phi]
