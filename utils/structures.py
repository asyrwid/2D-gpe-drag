from dataclasses import dataclass
import numpy as np


@dataclass
class SystemParameters:
    N: int
    L: float
    dt: float


@dataclass
class InteractionParameters:
    ga: float
    gb: float
    gab: float
    g_parallel: float
    g_perpendicular: float
    potential_range: float
    long_ga: float
    long_gb: float


@dataclass
class TrapParameters:
    wx: float
    wy: float
    xshift: float
    yshift: float


@dataclass
class EvolutionResults:
    evolved_fa: np.array
    evolved_fb: np.array
    energies: np.array
    norms_a: np.array
    norms_b: np.array
    momenta_a: np.array
    momenta_b: np.array


@dataclass
class EvolutionConfig:
    Nsaved: int
    Nsteps: int
    system_parameters: SystemParameters
    interaction_parameters: InteractionParameters
    trap_parameters_a: TrapParameters
    trap_parameters_b: TrapParameters
