from main import evolve
from utils.structures import (
    SystemParameters,
    InteractionParameters,
    TrapParameters,
    EvolutionConfig,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"})


def prepare_config() -> EvolutionConfig:
    system_parameters = SystemParameters(
        N=70,
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
        long_ga=-0.16,
        long_gb=-0.2,
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
        Nsaved=100,
        Nsteps=10,
        system_parameters=system_parameters,
        interaction_parameters=interaction_parameters,
        trap_parameters_a=trap_parameters_a,
        trap_parameters_b=trap_parameters_b,
    )

    return config


def generate_imag_evol_gif():

    config = prepare_config()
    results = evolve(config)

    max_val_a = np.absolute(results['evol_fa']).max()
    max_val_b = np.absolute(results['evol_fb']).max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi = 32)
    fig.set_facecolor('white')

    im1 = ax1.imshow(
        np.absolute(results['evol_fa'][0]),
        animated=True,
        vmin=0,
        vmax=max_val_a,
        cmap='gnuplot2',
    )
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    ax1.set_title(r"$|\psi_a|$", size=20)

    im2 = ax2.imshow(
        np.absolute(results['evol_fb'][0]),
        animated=True,
        vmin=0,
        vmax=max_val_b,
        cmap='gnuplot2',
    )
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    ax2.set_title(r"$|\psi_b|$", size=20)

    for ax in [ax1, ax2]:
        ax.set_xticks([0, 34, 69])
        ax.set_xticklabels([0, 0.25, 0.5])
        ax.set_yticks([0, 34, 69])
        ax.set_yticklabels([0, 0.25, 0.5])
        ax.set_xlabel(r"$x$", fontsize=15)
        ax.set_ylabel(r"$y$", fontsize=15)
        ax.tick_params(labelsize=15)

    energy = np.real(results['energies'][0])
    tau = np.absolute(results['times'][0])
    text = fig.text(
        0.025, 0.92, f"Imaginary time evolution: $\\tau = {'{:04.3f}'.format(tau)} \quad E = {'{:04.3f}'.format(energy)}$", size=20)

    def animate_func(j):
        im1.set_array(np.absolute(results['evol_fa'][j]))
        ax1.set_title(r"$|\psi_a|$", size=20)
        im2.set_array(np.absolute(results['evol_fb'][j]))
        ax2.set_title(r"$|\psi_b|$", size=20)
        en = np.real(results['energies'][j])
        tau = np.absolute(results['times'][j])
        text.set_text(
            f"Imaginary time evolution: $\\tau = {'{:04.3f}'.format(tau)} \quad E = {'{:04.3f}'.format(en)}$")
        fig.tight_layout()
        return [im1, im2]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(results['evol_fb'])-1,
        # interval=100,
        repeat=True,
    )
    anim.save('tmp/imag_time_evolution.gif', writer='ffmpeg', fps=20)


if __name__ == "__main__":
    generate_imag_evol_gif()
