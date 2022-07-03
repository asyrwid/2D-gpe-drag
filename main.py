import numpy as np

from utils.structures import (
    SystemParameters,
    InteractionParameters,
    TrapParameters,
    EvolutionConfig,
)
from evolution import Evolution


def run():

    system_parameters = SystemParameters(
        N=50,
        L=0.5,
        dt=-0.0001*1j,
    )
    interaction_parameters = InteractionParameters(
        ga=1,
        gb=1,
        gab=0,
        g_parallel=0,
        g_perpendicular=0,
        potential_range=0.05,
        long_ga=-0.175,
        long_gb=-0.175,
    )
    trap_parameters_a = TrapParameters(
        wx=0,
        wy=0,
        xshift=0,
        yshift=0,
    )
    trap_parameters_b = TrapParameters(
        wx=0,
        wy=0,
        xshift=0,
        yshift=0,
    )

    config = EvolutionConfig(
        Nsaved=10,
        Nsteps=200,
        system_parameters=system_parameters,
        interaction_parameters=interaction_parameters,
        trap_parameters_a=trap_parameters_a,
        trap_parameters_b=trap_parameters_b,
    )

    return evolve(config)


def evolve(config: EvolutionConfig) -> dict:

    evolution = Evolution(
        config.system_parameters,
        config.interaction_parameters,
        config.trap_parameters_a,
        config.trap_parameters_b,
    )

    fa0, fb0 = initial_states(
        config.system_parameters.N, config.system_parameters.L)

    results = evolution.crank_nicolson_evolution(
        fa0, fb0, config.Nsaved, config.Nsteps)

    return results


def initial_states(N: int, L: float):

    dx = L/(N-1)
    sigma = L/3
    a = 1./sigma**2

    fa0 = np.ones((N, N))
    fb0 = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            xa2 = (i*dx - L/2)**2
            xb2 = (i*dx - L/2)**2
            ya2 = (j*dx - L/2)**2
            yb2 = (j*dx - L/2)**2

            fa0[i, j] = np.exp(-a*xa2 - a*ya2)
            fb0[i, j] = np.exp(-a*xb2 - a*yb2)

    return fa0, fb0
