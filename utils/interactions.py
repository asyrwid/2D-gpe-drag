import numpy as np
from numba import jit

from utils.structures import InteractionParameters


@ jit(nopython=True)
def intra_component_int(
    potential_range: float,
    f: np.array,
    dx: float,
    pos_i: int,
    pos_j: int,
    g_core: float,
    g_long: float
) -> float:
    N = len(f)
    R = int(potential_range/dx)
    sides = np.arange(-R - 1, R + 2)
    intra_int = g_core * np.absolute(f[pos_i, pos_j]) ** 2
    for s in sides:
        for p in sides:
            dist = dx*(s**2 + p**2)**(0.5)
            if (dist <= potential_range + dx) and (dist > 0):
                i_s = ((pos_i + s) % N)
                j_s = ((pos_j + p) % N)
                r = 0.025 + dx*((s**2 + p**2)**0.5)
                factor = g_long*dx*dx/(r**3)
                intra_int += factor * np.absolute(f[i_s, j_s]) ** 2
    return intra_int


@ jit(nopython=True)
def nonlinearity(
        fa_extr: np.array,
        fb_extr: np.array,
        dx: float,
        ga,
        gb,
        gab,
        gpar,
        gperp,
        potential_range,
        long_ga,
        long_gb,
):
    gg = np.absolute(gpar) + np.absolute(gperp)
    N = len(fb_extr)
    fa_nonl = np.zeros((N, N), dtype=np.complex128)
    fb_nonl = np.zeros((N, N), dtype=np.complex128)
    for i in range(0, N):
        for j in range(0, N):
            # take into account pbc
            ip = i+1
            jp = j+1
            im = i-1
            jm = j-1
            if i == 0:
                im = N - 1
                im = N - 2
            if j == 0:
                jm = N - 1
                jm = N - 2
            if i == N - 1:
                ip = 0
                ip = 1
            if j == N - 1:
                jp = 0
                jp = 1

            # first derivatives
            dax = 1./(2.*dx)*(fa_extr[ip, j] - fa_extr[im, j])
            day = 1./(2.*dx)*(fa_extr[i, jp] - fa_extr[i, jm])
            dbx = 1./(2.*dx)*(fb_extr[ip, j] - fb_extr[im, j])
            dby = 1./(2.*dx)*(fb_extr[i, jp] - fb_extr[i, jm])
            # second derivatives
            lapax = 1./(dx*dx) * \
                (-2.*fa_extr[i, j] + fa_extr[ip, j] + fa_extr[im, j])
            lapay = 1./(dx*dx) * \
                (-2.*fa_extr[i, j] + fa_extr[i, jp] + fa_extr[i, jm])
            lapbx = 1./(dx*dx) * \
                (-2.*fb_extr[i, j] + fb_extr[ip, j] + fb_extr[im, j])
            lapby = 1./(dx*dx) * \
                (-2.*fb_extr[i, j] + fb_extr[i, jp] + fb_extr[i, jm])
            lapa = lapax + lapay
            lapb = lapbx + lapby
            # currents
            jax = 1. / \
                (2.*1j)*(np.conjugate(fa_extr[i, j]) *
                         dax - fa_extr[i, j]*np.conjugate(dax))
            jay = 1. / \
                (2.*1j)*(np.conjugate(fa_extr[i, j]) *
                         day - fa_extr[i, j]*np.conjugate(day))
            jbx = 1. / \
                (2.*1j)*(np.conjugate(fb_extr[i, j]) *
                         dbx - fb_extr[i, j]*np.conjugate(dbx))
            jby = 1. / \
                (2.*1j)*(np.conjugate(fb_extr[i, j]) *
                         dby - fb_extr[i, j]*np.conjugate(dby))
            # regularizing terms related to ja.ja & jb.jb
            jja1 = (2./1j)*(jax*dax + jay*day)
            jja2 = (-1./2.)*fa_extr[i, j]*(np.conjugate(fa_extr[i, j])
                                           * lapa - fa_extr[i, j]*np.conjugate(lapa))
            jjb1 = (2./1j)*(jbx*dbx + jby*dby)
            jjb2 = (-1./2.)*fb_extr[i, j]*(np.conjugate(fb_extr[i, j])
                                           * lapb - fb_extr[i, j]*np.conjugate(lapb))
            jaja = gg/2.*(jja1 + jja2)
            jbjb = gg/2.*(jjb1 + jjb2)
            # parallel drag
            jABa1 = (1./1j)*(jbx*dax + jby*day)
            jABa2 = -(1./4.)*fa_extr[i, j]*(np.conjugate(fb_extr[i, j])
                                            * lapb - fb_extr[i, j]*np.conjugate(lapb))
            jABb1 = (1./1j)*(jax*dbx + jay*dby)
            jABb2 = -(1./4.)*fb_extr[i, j]*(np.conjugate(fa_extr[i, j])
                                            * lapa - fa_extr[i, j]*np.conjugate(lapa))
            jABa = gpar*(jABa1 + jABa2)
            jABb = gpar*(jABb1 + jABb2)
            # vector drag
            jva1 = (1./1j)*(jby*dax - jbx*day)
            jva2 = -(1./2.)*fa_extr[i, j] * \
                (np.conjugate(dbx)*dby - dbx*np.conjugate(dby))
            jvb1 = -(1./1j)*(jay*dbx - jax*dby)
            jvb2 = (1./2.)*fb_extr[i, j] * \
                (np.conjugate(dax)*day - dax*np.conjugate(day))
            jva = gperp*(jva1 + jva2)
            jvb = gperp*(jvb1 + jvb2)
            # density-density interactions
            contact_ba = gab*fb_extr[i, j]*np.absolute(fa_extr[i, j])**2
            contact_ab = gab*fa_extr[i, j]*np.absolute(fb_extr[i, j])**2

            intra_a = intra_component_int(
                potential_range, fa_extr, dx, i, j, ga, long_ga)*fa_extr[i, j]
            intra_b = intra_component_int(
                potential_range, fb_extr, dx, i, j, gb, long_gb)*fb_extr[i, j]

            fa_nonl[i, j] = jaja + jABa + jva + contact_ab + intra_a
            fb_nonl[i, j] = jbjb + jABb + jvb + contact_ba + intra_b

    return fa_nonl, fb_nonl
