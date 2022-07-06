import numpy as np
from scipy import sparse
from tqdm import tqdm

from utils.matrices import (
    laplacian2D_pbc,
    inv_matrix1,
    matrix2,
)
from utils.interactions import nonlinearity
from utils.structures import (
    InteractionParameters,
    TrapParameters,
    SystemParameters,
)

from utils.energy import compute_energy, periodic_idxs


class Evolution():
    def __init__(
        self,
        system_parameters: SystemParameters,
        interaction_parameters: InteractionParameters,
        trap_parameters_a: TrapParameters,
        trap_parameters_b: TrapParameters,
    ):

        self.system_parameters = system_parameters
        self.interaction_parameters = interaction_parameters
        self.trap_parameters_a = trap_parameters_a
        self.trap_parameters_b = trap_parameters_b
        self.N = system_parameters.N
        self.L = system_parameters.L
        # finite difference along 'square lattice'
        dx = self.L*1./(self.N-1.)
        self.dx = dx  # here I assume that dy = dx
        self.dt = system_parameters.dt
        # generate matrices
        self.laplacian2d = laplacian2D_pbc(self.N)  # sparse array

        trap_params_arr_a = np.array([
            trap_parameters_a.wx,
            trap_parameters_a.wy,
            trap_parameters_a.xshift,
            trap_parameters_a.yshift,
        ])
        trap_params_arr_b = np.array([
            trap_parameters_b.wx,
            trap_parameters_b.wy,
            trap_parameters_b.xshift,
            trap_parameters_b.yshift,
        ])

        dif = np.sum((trap_params_arr_a - trap_params_arr_b)**2)
        different_traps = True
        if dif**0.5 < 0.0001:
            different_traps = False
        self.different_traps = different_traps
        if different_traps is True:
            print(
                'trap parameters are different: generate separate matrices for a and b components')
            self.O2a = sparse.csr_matrix(
                matrix2(self.laplacian2d, self.dx, self.dt, trap_parameters_a))
            self.O2b = sparse.csr_matrix(
                matrix2(self.laplacian2d, self.dx, self.dt, trap_parameters_b))
            self.invO1a = inv_matrix1(
                self.laplacian2d, self.dx, self.dt, trap_parameters_a)
            self.invO1b = inv_matrix1(
                self.laplacian2d, self.dx, self.dt, trap_parameters_b)
        else:
            print(
                'trap parameters are identical: generate the same matrices for both components')
            self.O2 = sparse.csr_matrix(
                matrix2(self.laplacian2d, self.dx, self.dt, trap_parameters_a))
            self.invO1 = inv_matrix1(
                self.laplacian2d, self.dx, self.dt, trap_parameters_a)

        self.ga = interaction_parameters.ga
        self.gb = interaction_parameters.gb
        self.gab = interaction_parameters.gab
        self.g_parallel = interaction_parameters.g_parallel
        self.g_perpendicular = interaction_parameters.g_perpendicular
        self.potential_range = interaction_parameters.potential_range
        self.long_ga = interaction_parameters.long_ga
        self.long_gb = interaction_parameters.long_gb
        self.wa_x = trap_parameters_a.wx
        self.wa_y = trap_parameters_a.wy
        self.trap_a_shift_x = trap_parameters_a.xshift
        self.trap_a_shift_y = trap_parameters_a.yshift
        self.wb_x = trap_parameters_b.wx
        self.wb_y = trap_parameters_b.wy
        self.trap_b_shift_x = trap_parameters_b.xshift
        self.trap_b_shift_y = trap_parameters_b.yshift

    def compute_momentum(self, f: np.array) -> float:
        px = 0.
        py = 0.
        for i in range(self.N):
            for j in range(self.N):
                # take into account pbc
                indices = periodic_idxs(self.N, i, j)
                ip1 = indices['i_plus_1']
                im1 = indices['i_minus_1']
                jp1 = indices['j_plus_1']
                jm1 = indices['j_minus_1']
                ip2 = indices['i_plus_2']
                im2 = indices['i_minus_2']
                jp2 = indices['j_plus_2']
                jm2 = indices['j_minus_2']

                left_x = - f[ip2, j] + 8*f[ip1, j]
                right_x = - 8*f[im1, j] + f[im2, j]
                left_y = - f[i, jp2] + 8*f[i, jp1]
                right_y = - 8*f[i, jm1] + f[i, jm2]

                dfx = 1/(12*self.dx)*(left_x + right_x)
                dfy = 1/(12*self.dx)*(left_y + right_y)

                px += -1j*(self.dx**2)*np.conjugate(f[i, j])*dfx
                py += -1j*(self.dx**2)*np.conjugate(f[i, j])*dfy
        return np.array([px, py])

    def crank_nicolson_evolution(self,
                                 fa0: np.array,
                                 fb0: np.array,
                                 Nsaved: int,
                                 Nsteps: int
                                 ) -> dict:
        # fa0 and fb0 are initial states for the evolution
        # they should be given in the form of 2d arrays
        # Nsaved = number of saved outcomes in the whole evolution
        # Nsteps = number of dt time steps per single outcome
        # Each outcome is saved after Delta_t = Nsteps*dt

        # ensure that the states are normalized
        norm_a0 = (np.sum(np.absolute(fa0)**2)*(self.dx**2))**(-0.5)
        norm_b0 = (np.sum(np.absolute(fb0)**2)*(self.dx**2))**(-0.5)
        fa0 = fa0*norm_a0
        fb0 = fb0*norm_b0
        # create arrays for storing final results
        evol_fa = np.zeros((Nsaved+1, self.N, self.N), dtype=np.complex128)
        evol_fb = np.zeros((Nsaved+1, self.N, self.N), dtype=np.complex128)
        energies = np.zeros((Nsaved+1), dtype=np.complex128)
        norms_a = np.zeros((Nsaved+1), dtype=np.float64)
        norms_b = np.zeros((Nsaved+1), dtype=np.float64)
        momenta_a = np.zeros((Nsaved+1, 2), dtype=np.complex128)
        momenta_b = np.zeros((Nsaved+1, 2), dtype=np.complex128)
        times = np.zeros((Nsaved+1), dtype=np.complex128)
        # and set the first elements
        evol_fa[0] = fa0
        evol_fb[0] = fb0
        energies[0] = compute_energy(
            fa0,
            fb0,
            self.N,
            self.dx,
            self.interaction_parameters,
            self.trap_parameters_a,
            self.trap_parameters_b,
        )
        norms_a[0] = (np.sum(np.absolute(fa0.flatten())**2)*self.dx**2)
        norms_b[0] = (np.sum(np.absolute(fb0.flatten())**2)*self.dx**2)
        momenta_a[0] = self.compute_momentum(fa0)
        momenta_b[0] = self.compute_momentum(fb0)
        times[0] = 0
        # set extrapolation for the initial state
        fa1 = fa0
        fb1 = fb0
        fa_extrapolated = (3.*fa1 - fa0)/2.
        fb_extrapolated = (3.*fb1 - fb0)/2.
        # and flatten the arrays to vectors
        flat_fa1 = fa1.flatten()
        flat_fb1 = fb1.flatten()

        pbar = tqdm(range(1, Nsaved + 1), total=Nsaved)

        for save in pbar:  # range(1, Nsaved+1):
            for step in range(0, Nsteps):
                # caclutate linear part: O2 matrix action
                if self.different_traps is True:
                    flat_fa12 = self.O2a.dot(flat_fa1)
                    flat_fb12 = self.O2b.dot(flat_fb1)
                else:
                    flat_fa12 = self.O2.dot(flat_fa1)
                    flat_fb12 = self.O2.dot(flat_fb1)
                # calculate nonlinearities
                fa_nonl, fb_nonl = nonlinearity(
                    fa_extrapolated,
                    fb_extrapolated,
                    self.dx,
                    self.ga,
                    self.gb,
                    self.gab,
                    self.g_parallel,
                    self.g_perpendicular,
                    self.potential_range,
                    self.long_ga,
                    self.long_gb,
                    # self.interaction_parameters,
                )

                flat_fa12 = flat_fa12 + fa_nonl.flatten()
                flat_fb12 = flat_fb12 + fb_nonl.flatten()
                # calculate final vector: inverse of O1 action
                if self.different_traps is True:
                    flat_fa2 = self.invO1a.dot(flat_fa12)
                    flat_fb2 = self.invO1b.dot(flat_fb12)
                else:
                    flat_fa2 = self.invO1.dot(flat_fa12)
                    flat_fb2 = self.invO1.dot(flat_fb12)

                # normalize if imaginary time evolution
                if np.absolute(np.imag(self.dt)) > 0.:
                    norm_a = (np.sum(np.absolute(flat_fa2)**2)
                              * (self.dx**2))**(-0.5)
                    flat_fa2 = flat_fa2*norm_a
                    norm_b = (np.sum(np.absolute(flat_fb2)**2)
                              * (self.dx**2))**(-0.5)
                    flat_fb2 = flat_fb2*norm_b

                fa2 = flat_fa2.reshape(self.N, self.N)
                fb2 = flat_fb2.reshape(self.N, self.N)

                # recalculate wave functions for the next step
                fa0 = fa1
                fb0 = fb1
                fa1 = fa2
                fb1 = fb2
                fa_extrapolated = (3*fa1 - fa0)/2.
                fb_extrapolated = (3*fb1 - fb0)/2.

                flat_fa1 = fa1.flatten()
                flat_fb1 = fb1.flatten()

            evol_fa[save] = fa1
            evol_fb[save] = fb1

            energy = compute_energy(
                fa1,
                fb1,
                self.N,
                self.dx,
                self.interaction_parameters,
                self.trap_parameters_a,
                self.trap_parameters_b,
            )
            norm_a = (np.sum(np.absolute(fa1.flatten())**2)*self.dx**2)
            norm_b = (np.sum(np.absolute(fb1.flatten())**2)*self.dx**2)
            momentum_a = self.compute_momentum(fa1)
            momentum_b = self.compute_momentum(fb1)
            time = save * Nsteps * self.dt

            prt_e = round(np.real(energy), 5)
            prt_na = round(norm_a, 5)
            prt_nb = round(norm_b, 5)
            prt_pa = [round(np.real(momentum_a[0]), 5),
                      round(np.real(momentum_a[1]), 5)]
            prt_pb = [round(np.real(momentum_b[0]), 5),
                      round(np.real(momentum_b[1]), 5)]

            print_line = f"E = {prt_e} norm_a = {prt_na} norm_b = {prt_nb} p_a = {prt_pa} p_b = {prt_pb}"
            desc = print_line
            pbar.set_description(desc)

            energies[save] = energy
            norms_a[save] = norm_a
            norms_b[save] = norm_b
            momenta_a[save] = momentum_a
            momenta_b[save] = momentum_b
            times[save] = time

            results = {
                "evol_fa": evol_fa,
                "evol_fb": evol_fb,
                "energies": energies,
                "norms_a": norms_a,
                "norms_b": norms_b,
                "momenta_a": momenta_a,
                "momenta_b": momenta_b,
                "times": times,
            }

        return results
