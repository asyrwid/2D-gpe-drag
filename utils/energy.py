from readline import get_begidx
import numpy as np

from utils.interactions import intra_component_int
from utils.structures import (
    InteractionParameters,
    TrapParameters,
)


def compute_kinetic_energy(f: np.array, N: int, dx: float) -> float:
    e_kin = 0.
    for i in range(N):
        for j in range(N):

            indices = periodic_idxs(N, i, j)
            ip1 = indices['i_plus_1']
            im1 = indices['i_minus_1']
            jp1 = indices['j_plus_1']
            jm1 = indices['j_minus_1']
            ip2 = indices['i_plus_2']
            im2 = indices['i_minus_2']
            jp2 = indices['j_plus_2']
            jm2 = indices['j_minus_2']

            left_x = - f[ip2, j] + 16*f[ip1, j]
            left_y = - f[i, jp2] + 16*f[i, jp1]
            right_x = 16*f[im1, j] - f[im2, j]
            right_y = 16*f[i, jm1] - f[i, jm2]

            ddxf = 1./(12*(dx**2)) * (left_x - 30*f[i, j] + right_x)
            ddyf = 1./(12*(dx**2)) * (left_y - 30*f[i, j] + right_y)

            e_kin += -np.conjugate(f[i, j]) * ddxf/2.
            e_kin += -np.conjugate(f[i, j]) * ddyf/2.

    return e_kin*(dx**2)


def compute_int_energy(
    fa: np.array,
    fb: np.array,
    N: int,
    dx: float,
    interaction_parameters: InteractionParameters,
) -> float:

    potential_range = interaction_parameters.potential_range
    ga = interaction_parameters.ga
    gb = interaction_parameters.gb
    gab = interaction_parameters.gab
    long_ga = interaction_parameters.long_ga
    long_gb = interaction_parameters.long_gb

    intra_a = 0.
    intra_b = 0.
    contact_ab = 0.
    for i in range(N):
        for j in range(N):
            intra_a += 0.5 * intra_component_int(
                potential_range, fa, dx, i, j, ga, long_ga) * np.absolute(fa[i, j])**2
            intra_b += 0.5 * intra_component_int(
                potential_range, fb, dx, i, j, gb, long_gb) * np.absolute(fb[i, j])**2

            contact_ab += gab * \
                (np.absolute(fa[i, j])**2)*(np.absolute(fb[i, j])**2)

    return intra_a*(dx**2), intra_b*(dx**2), contact_ab*(dx**2)


def compute_current_current_energy(
    fa: np.array,
    fb: np.array,
    N: int,
    dx: float,
    g_parallel,
    g_perpendicular,
) -> float:
    jaxjax = 0.
    jayjay = 0.
    jbxjbx = 0.
    jbyjby = 0.
    jaxjbx = 0.
    jayjby = 0.
    jaxjby = 0.
    jayjbx = 0.
    for i in range(N):
        for j in range(N):
            # take into account pbc
            indices = periodic_idxs(N, i, j)
            ip1 = indices['i_plus_1']
            im1 = indices['i_minus_1']
            jp1 = indices['j_plus_1']
            jm1 = indices['j_minus_1']
            ip2 = indices['i_plus_2']
            im2 = indices['i_minus_2']
            jp2 = indices['j_plus_2']
            jm2 = indices['j_minus_2']

            left_ax = - fa[ip2, j] + 8*fa[ip1, j]
            right_ax = - 8*fa[im1, j] + fa[im2, j]
            left_ay = - fa[i, jp2] + 8*fa[i, jp1]
            right_ay = - 8*fa[i, jm1] + fa[i, jm2]

            left_bx = - fb[ip2, j] + 8*fb[ip1, j]
            right_bx = - 8*fb[im1, j] + fb[im2, j]
            left_by = - fb[i, jp2] + 8*fb[i, jp1]
            right_by = - 8*fb[i, jm1] + fb[i, jm2]

            dax = 1/(12*dx)*(left_ax + right_ax)
            day = 1/(12*dx)*(left_ay + right_ay)
            dbx = 1/(12*dx)*(left_bx + right_bx)
            dby = 1/(12*dx)*(left_by + right_by)

            jax = 1./(2.*1j) * \
                (np.conjugate(fa[i, j])*dax - fa[i, j]*np.conjugate(dax))
            jay = 1./(2.*1j) * \
                (np.conjugate(fa[i, j])*day - fa[i, j]*np.conjugate(day))
            jbx = 1./(2.*1j) * \
                (np.conjugate(fb[i, j])*dbx - fb[i, j]*np.conjugate(dbx))
            jby = 1./(2.*1j) * \
                (np.conjugate(fb[i, j])*dby - fb[i, j]*np.conjugate(dby))

            jaxjax += jax*jax*(dx**2)
            jayjay += jay*jay*(dx**2)
            jbxjbx += jbx*jbx*(dx**2)
            jbyjby += jby*jby*(dx**2)
            jaxjbx += jax*jbx*(dx**2)
            jayjby += jay*jby*(dx**2)
            jaxjby += jax*jby*(dx**2)
            jayjbx += jay*jbx*(dx**2)

    e_jaja = 0.5*(np.absolute(g_parallel) +
                  np.absolute(g_perpendicular))*(jaxjax + jayjay)
    e_jbjb = 0.5*(np.absolute(g_parallel) +
                  np.absolute(g_perpendicular))*(jbxjbx + jbyjby)
    e_ABdrag = g_parallel*(jaxjbx + jayjby)
    e_vecdrag = g_perpendicular*(jaxjby - jayjbx)

    current_energies = {
        "e_jaja": e_jaja,
        "e_jbjb": e_jbjb,
        "e_ABdrag": e_ABdrag,
        "e_vecdrag": e_vecdrag,
    }

    return current_energies


def compute_energy(
    fa: np.array,
    fb: np.array,
    N: int,
    dx: float,
    interaction_parameters: InteractionParameters,
    trap_parameters_a: TrapParameters,
    trap_parameters_b: TrapParameters,
) -> float:
    e_kin_a = compute_kinetic_energy(fa, N, dx)
    e_kin_b = compute_kinetic_energy(fb, N, dx)
    e_kin = e_kin_a + e_kin_b

    e_intr_a, e_intr_b, e_cont_ab = compute_int_energy(
        fa, fb, N, dx, interaction_parameters)
    e_int = e_intr_a + e_intr_b + e_cont_ab

    g_parallel = interaction_parameters.g_parallel
    g_perpendicular = interaction_parameters.g_perpendicular
    current_energies = compute_current_current_energy(
        fa, fb, N, dx, g_parallel, g_perpendicular)
    e_jaja = current_energies["e_jaja"]
    e_jbjb = current_energies["e_jbjb"]
    e_ABdrag = current_energies["e_ABdrag"]
    e_vecdrag = current_energies["e_vecdrag"]
    e_current_current = e_jaja + e_jbjb + e_ABdrag + e_vecdrag

    # and compute the energy due to harmonic trap
    trap_a_shift_x = trap_parameters_a.xshift
    trap_a_shift_y = trap_parameters_a.yshift
    trap_b_shift_x = trap_parameters_b.xshift
    trap_b_shift_y = trap_parameters_b.yshift
    wa_x = trap_parameters_a.wx
    wa_y = trap_parameters_a.wy
    wb_x = trap_parameters_b.wx
    wb_y = trap_parameters_b.wy

    n0 = (N - 1)/2.
    e_harm_a = 0.
    e_harm_b = 0.
    for i in range(N):
        for j in range(N):
            xa2 = (dx*(i - n0) - trap_a_shift_x)**2
            ya2 = (dx*(j - n0) - trap_a_shift_y)**2
            xb2 = (dx*(i - n0) - trap_b_shift_x)**2
            yb2 = (dx*(j - n0) - trap_b_shift_y)**2
            e_harm_a += (np.absolute(fa[i, j])**2)*(dx**2) * \
                ((wa_x**2)*xa2 + (wa_y**2)*ya2)/2.
            e_harm_b += (np.absolute(fb[i, j])**2)*(dx**2) * \
                ((wb_x**2)*xb2 + (wb_y**2)*yb2)/2.

    e_harm = e_harm_a + e_harm_b
    return e_kin + e_int + e_current_current + e_harm


def periodic_idxs(N: int, i: int, j: int) -> dict:
    ip1 = i+1
    jp1 = j+1
    im1 = i-1
    jm1 = j-1

    ip2 = i+2
    jp2 = j+2
    im2 = i-2
    jm2 = j-2

    if i == 0:
        im1 = N - 1
        im2 = N - 2
    if j == 0:
        jm1 = N - 1
        jm2 = N - 2
    if i == N - 1:
        ip1 = 0
        ip2 = 1
    if j == N - 1:
        jp1 = 0
        jp2 = 1
    if i == 1:
        im2 = N - 1
    if j == 1:
        jm2 = N - 1
    if i == N - 2:
        ip2 = 0
    if j == N - 2:
        jp2 = 0

    pbc_idxs2 = {
        'i_plus_1': ip1,
        'j_plus_1': jp1,
        'i_plus_2': ip2,
        'j_plus_2': jp2,
        'i_minus_1': im1,
        'j_minus_1': jm1,
        'i_minus_2': im2,
        'j_minus_2': jm2,
    }

    return pbc_idxs2
