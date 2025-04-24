# Data Assimilation for Orszag-Tang MHD Vortex Problem
## Introduction
This is a github repostory for the research project "Data assimilation for Orszag-Tang Magnetohydrodynamic Vortex Problem". Here, FDM based 2D MHD simulation code and EnKF are developed to estimate the nonlinear dynamics for 2D MHD environment where the measurement noise and the process noise exist in the system.

## Background
### MHD and Orszag-Tnag MHD vortex
A plasma is an ionized gas with quasi-neutrality, which behaves like an electrically conducting fluid. Charge neutrality and small gyro-radius approximation allow the plasma to be interpreted as a single fluid model linearly combining the average motions of individual species. This is called Magnetohydrodynamics (MHD)

In the 2D periodic system, a supersonic MHD turbulence model problem called the Orszag-Tang MHD vortex problem can be observed when large sinusoidal perturbations in the velocity and magnetic field are applied. Below figure shows the evolution of the plasma density with the initial perturbation given. 

<div>
    <p float = 'left'>
        <img src="/results/simulation/density_evolution.gif"  width="50%">
    </p>
</div>

### Ensemble Kalman Filter
Ensemble Kalman Filter(EnKF) is a Monte Carlo method based Kalman filtering, replacing the covariance required for Kalman updating by the sample covariance from the ensemble. EnKF can be a good choice for MHD since it does not require linearization for nonlinear dynamics approximation by ensemble sampling. In this project, Gaussian noise is added when sensing measurement points and computing the MHD equation for time integration for representing the measurement noise.

## Numerical results
### Density estimation

<div>
    <p float = 'left'>
        <img src="/results/rho/density_evolution_comparison.gif"  width="100%">
    </p>
</div>

### Magnetic field estimation

<div>
    <p float = 'left'>
        <img src="/results/B/Bx_evolution_comparison.gif"  width="100%">
    </p>
</div>

<div>
    <p float = 'left'>
        <img src="/results/B/By_evolution_comparison.gif"  width="100%">
    </p>
</div>

### Pressure estimation

<div>
    <p float = 'left'>
        <img src="/results/P/pressure_evolution_comparison.gif"  width="100%">
    </p>
</div>