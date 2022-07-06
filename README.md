# 2D-gpe-drag
numerical evolution of a binary mixture of bosons with drag in 2D 

<img src="/tmp/imag_time_evolution.gif" width="50%" height="50%"/>

## Simulated system

Here, we study dynamics of the binary superfluid Bose system in 2D described by the following energy functional

$ \mathcal{E}=\int\mathrm{d}{\bf r} [\varepsilon_0 +  \varepsilon_\text{b-c} + \varepsilon_\text{d-d} + \varepsilon_\text{drag-reg} + \varepsilon_\text{AB-drag} + \varepsilon_\text{vector-drag}], $

where standard kinetic, harmonic trapping potential and intra-component contact interaction terms are included in 

$\varepsilon_0 =  \sum_\alpha\left[ \frac{1}{2}|\nabla \psi_\alpha|^2 + \frac{1}{2}\omega_\alpha^2 {\bf r}^2|\psi_\alpha|^2 + g_\alpha |\psi_\alpha|^4 \right]$

with $\alpha\in\{a, b\}$ denoting components. In addition, we assume beyond contact intra-component interactions decaying as $r^{-3}$

$\varepsilon_\text{b-c}({\bf r}) = \sum_\alpha\int \mathrm{d} {\bf r}' g_{\alpha}^{\text{(b-c)}}|\psi_\alpha({\bf r})|^2 |\psi_\alpha({\bf r}')|^2/|{\bf r} - {\bf r}'|^3,$

standard contact density-density inter-component interactions

$\varepsilon_\text{d-d} = g_{ab}|\psi_a|^2 |\psi_b|^2,$

and dissipationless drag-related terms:

1. regularization required for $\mathcal{E}$ to be bounded from below
  $\varepsilon_\text{drag-reg} = \frac{1}{2}(|g_\parallel|+|g_\perp|)({\bf j}_a^2 + {\bf j}_b^2) $,

2. standard collinear Andree-Bashkin drag $\varepsilon_\text{AB-drag} = g_{\parallel} {\bf j}_a \cdot {\bf j}_b$, 

3. novel vector-drag effect $\varepsilon_\text{vector-drag} = g_{\perp} {\bf j}_a \times {\bf j}_b \cdot {\bf e}_z$  (see: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.100403). 


