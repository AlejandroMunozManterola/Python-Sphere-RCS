import numpy as np
from DielectricMaterial import DielectricMaterial as DM
from src import *
from bessel import *
from getNMax import getNMax


def getCoatedPECSphereFieldUnderPlaneWave(
    outer_radius,
    coating_material,
    background,
    sensor_location,
    frequency,
    coating_thickness=None,
    inner_radius=None,
):
    """
    Scattered field for a PEC core with dielectric shell.

    Geometry:
      r < inner_radius              : PEC core
      inner_radius < r < outer_radius : coating
      r > outer_radius              : background
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
    k_1 = DM.getWaveNumber(coating_material, frequency)

    m1 = k_1 / k_m

    N = getNMax(outer_radius, coating_material, background, frequency)
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

    y_inner = k_1 * inner_radius
    x_outer = k_m * outer_radius

    psi_y = ric_besselj(nu, y_inner)
    psi_y_p = ric_besselj_derivative(nu, y_inner)
    chi_y = -ric_bessely(nu, y_inner)
    chi_y_p = -ric_bessely_derivative(nu, y_inner)

    # PEC inner boundary condition.
    A_n = -psi_y_p / chi_y_p
    B_n = -psi_y / chi_y

    m1_row = m1.reshape(1, M)
    m1x = m1 * x_outer

    psi_x = ric_besselj(nu, x_outer)
    psi_x_p = ric_besselj_derivative(nu, x_outer)
    xi_x = ric_besselh(nu, x_outer, 2)
    xi_x_p = ric_besselh_derivative(nu, x_outer, 2)

    psi_m1x = ric_besselj(nu, m1x)
    psi_m1x_p = ric_besselj_derivative(nu, m1x)
    chi_m1x = -ric_bessely(nu, m1x)
    chi_m1x_p = -ric_bessely_derivative(nu, m1x)

    hat_a_psi = psi_m1x_p - A_n * chi_m1x_p
    hat_a_psi0 = psi_m1x - A_n * chi_m1x

    num_a = m1_row * psi_x * hat_a_psi - psi_x_p * hat_a_psi0
    den_a = m1_row * xi_x * hat_a_psi - xi_x_p * hat_a_psi0
    scat_a_n = (num_a / den_a).T * inc_n

    hat_b_psi = psi_m1x_p - B_n * chi_m1x_p
    hat_b_psi0 = psi_m1x - B_n * chi_m1x

    num_b = psi_x * hat_b_psi - m1_row * psi_x_p * hat_b_psi0
    den_b = xi_x * hat_b_psi - m1_row * xi_x_p * hat_b_psi0
    scat_b_n = (num_b / den_b).T * inc_n

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

    x = k_m * r

    alpha00 = np.transpose(ric_besselh_derivative(nu, x, 2, 2))
    alpha01 = np.transpose(ric_besselh(nu, x, 2))
    alpha10 = np.array(get_legendre(nu, 1, np.cos(theta)))
    alpha11 = np.transpose(np.reshape(alpha10, (N, 1)) @ np.ones((1, M)))
    alpha = (alpha00 + alpha01) * alpha11

    E_r = -1j * np.cos(phi) * np.sum(scat_a_n * alpha, axis=1)
    H_r = -1j * np.sin(phi) * np.sum(scat_b_n * alpha, axis=1) / eta_m

    alpha_e = np.transpose(ric_besselh_derivative(nu, x, 2)) * aux1
    beta_e = np.transpose(ric_besselh(nu, x, 2)) * aux0
    E_theta = (np.cos(phi) / x) * np.sum(1j * scat_a_n * alpha_e - scat_b_n * beta_e, axis=1)
    H_theta = (np.sin(phi) / x) * np.sum(1j * scat_b_n * alpha_e - scat_a_n * beta_e, axis=1) / eta_m

    alpha_p = np.transpose(ric_besselh_derivative(nu, x, 2)) * aux0
    beta_p = np.transpose(ric_besselh(nu, x, 2)) * aux1
    E_phi = (np.sin(phi) / x) * np.sum(1j * scat_a_n * alpha_p - scat_b_n * beta_p, axis=1)
    H_phi = (-np.cos(phi) / x) * np.sum(1j * scat_b_n * alpha_p - scat_a_n * beta_p, axis=1) / eta_m

    return [E_r, E_theta, E_phi, H_r, H_theta, H_phi]
